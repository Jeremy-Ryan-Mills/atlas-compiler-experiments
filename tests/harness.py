"""
Before/after equivalence harness for Atlas kernel optimizations.

Runs a kernel from `third_party/npu_model` as written, runs it again after an
optimizer has rewritten its assembly, and compares the architectural state both
runs leave behind. Both runs use the model's default mode (no scoreboard), so an
optimized kernel only passes if its own ordering and `delay`s are sufficient.
"""

from __future__ import annotations

import contextlib
import io
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
NPU_MODEL_ROOT = REPO_ROOT / "third_party" / "npu_model"
ASM_DIR = NPU_MODEL_ROOT / "npu_model" / "configs" / "programs" / "asm"

_asm_sources: dict[int, tuple[list, Path]] = {}
"""id(instruction list) -> (that list, the .S file it was parsed from)."""


def bootstrap_npu_model() -> None:
    """
    Make `npu_model` importable from this repo. Must run before anything imports
    `npu_model.configs.programs`.

    - `ASM_FOLDER` is relative to the npu_model checkout, so it only resolves when
      the cwd is that checkout. Point it at the absolute path instead.
    - Wrap `load_asm` to remember which `.S` file each program's instruction list
      came from, so the optimizer receives the original source (labels, `li`,
      comments) rather than a flattened re-print with numeric branch offsets.
    """
    if not (NPU_MODEL_ROOT / "npu_model").is_dir():
        raise RuntimeError(
            f"{NPU_MODEL_ROOT} is empty. Run `git submodule update --init`."
        )
    sys.path.insert(0, str(NPU_MODEL_ROOT))

    import npu_model.software.program as program_module
    import npu_model.util.converter as converter

    program_module.ASM_FOLDER = ASM_DIR

    original_load_asm = converter.load_asm

    def recording_load_asm(source: Path):
        instructions = original_load_asm(source)
        _asm_sources[id(instructions)] = (instructions, Path(source))
        return instructions

    converter.load_asm = recording_load_asm


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Kernel:
    name: str
    program_cls: type

    def program(self):
        return self.program_cls()

    @property
    def asm_path(self) -> Path | None:
        entry = _asm_sources.get(id(self.program_cls.instructions))
        if entry is None or entry[0] is not self.program_cls.instructions:
            return None
        return entry[1]

    def source(self) -> str:
        """Assembly text to hand to the optimizer."""
        if self.asm_path is not None:
            return self.asm_path.read_text()
        from npu_model.util.converter import program_to_asm

        return program_to_asm(self.program())


def discover_kernels() -> list[Kernel]:
    import npu_model.configs.programs as program_configs

    names = sorted(getattr(program_configs, "__all__", []))
    return [Kernel(name, getattr(program_configs, name)) for name in names]


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    cycles: int
    finished: bool
    regions: dict[str, bytes] = field(default_factory=dict)
    """Label -> final bytes of each compared memory region."""


def compared_dram_regions(program) -> list[tuple[str, int, int]]:
    """(label, DRAM base, length) for the golden output and every input region."""
    regions = []
    golden = getattr(program, "golden_result", None)
    if golden:
        base, tensor = golden
        regions.append(("dram_output", base, tensor.numel() * tensor.element_size()))
    for base, tensor in program.memory_regions:
        regions.append(
            (f"dram_input@{base:#x}", base, tensor.numel() * tensor.element_size())
        )
    return regions


def run_program(
    program,
    hardware_config,
    *,
    dram_regions: list[tuple[str, int, int]],
    max_cycles: int,
    compare_vmem: bool,
) -> RunResult:
    """
    Run `program` without the scoreboard and snapshot `dram_regions` (from
    `compared_dram_regions` on the *original* program, so both runs capture the
    same bytes) plus, optionally, all of VMEM.
    """
    from npu_model.logging import LoggerConfig
    from npu_model.simulation import Simulation

    with tempfile.TemporaryDirectory() as trace_dir:
        sim = Simulation(
            hardware_config=hardware_config,
            logger_config=LoggerConfig(filename=str(Path(trace_dir) / "trace.json")),
            program=program,
            verbose=False,
        )
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                sim.run(max_cycles=max_cycles)
            state = sim.core.arch_state
            result = RunResult(cycles=sim.cycle_count, finished=sim.core.is_finished())
            # Index DRAM directly: `read_dram` offsets by the final `dma.base`, which
            # the kernel may have left pointing somewhere else.
            for label, base, length in dram_regions:
                result.regions[label] = state.dram[base : base + length].numpy().tobytes()
            if compare_vmem:
                result.regions["vmem"] = state.vmem.numpy().tobytes()
            return result
        finally:
            sim.close()


def diff_regions(before: RunResult, after: RunResult) -> list[str]:
    """Human-readable description of every region that differs."""
    problems = []
    for label, expected in before.regions.items():
        actual = after.regions.get(label)
        if actual is None:
            problems.append(f"{label}: missing from optimized run")
            continue
        if actual == expected:
            continue
        mismatched = [i for i, (a, b) in enumerate(zip(expected, actual)) if a != b]
        first = mismatched[0]
        problems.append(
            f"{label}: {len(mismatched)}/{len(expected)} bytes differ, "
            f"first at +{first:#x} (before={expected[first]:#04x}, "
            f"after={actual[first]:#04x})"
        )
    return problems


def program_from_asm(text: str, memory_regions):
    from npu_model.software.program import InstantiableProgram
    from npu_model.util.converter import input_to_program

    program = input_to_program(io.StringIO(text))
    optimized = InstantiableProgram(program.instructions)
    optimized.memory_regions = memory_regions
    return optimized


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

Optimizer = Callable[[str, Path], str]
"""(input assembly, scratch dir) -> optimized assembly."""


def identity_optimizer(source: str, workdir: Path) -> str:
    return source


_DELAY_LINE = re.compile(r"^\s*delay\b", re.IGNORECASE)


def strip_delays_optimizer(source: str, workdir: Path) -> str:
    """Deliberately unsafe: drops every `delay`. Used to prove the harness catches breakage."""
    return "".join(
        line for line in source.splitlines(keepends=True) if not _DELAY_LINE.match(line)
    )


BUILTIN_OPTIMIZERS: dict[str, Optimizer] = {
    "identity": identity_optimizer,
    "strip-delays": strip_delays_optimizer,
}


def external_optimizer(command: list[str]) -> Optimizer:
    """
    Wrap a CLI with the `atlas-opt` interface (PLAN.md §5.5):
        <command...> in.S -o out.S
    """

    def run(source: str, workdir: Path) -> str:
        in_path = workdir / "before.S"
        out_path = workdir / "after.S"
        in_path.write_text(source)
        completed = subprocess.run(
            [*command, str(in_path), "-o", str(out_path)],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise OptimizerError(
                f"`{shlex.join(command)}` exited with {completed.returncode}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        if not out_path.exists():
            raise OptimizerError(f"`{shlex.join(command)}` did not write {out_path}")
        return out_path.read_text()

    return run


class OptimizerError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


class EquivalenceError(AssertionError):
    pass


class BaselineError(AssertionError):
    """The unoptimized kernel itself is broken, so there is nothing to compare to."""


def _check_golden(kernel: Kernel, program, result: RunResult) -> None:
    import torch

    golden = getattr(program, "golden_result", None)
    if not golden:
        return
    _, expected = golden
    actual = (
        torch.frombuffer(bytearray(result.regions["dram_output"]), dtype=torch.uint8)
        .view(expected.dtype)
        .reshape(expected.shape)
    )
    rtol, atol = getattr(program, "kernel_tolerance", (1e-2, 1e-2))
    if not torch.allclose(actual.float(), expected.float(), rtol=rtol, atol=atol):
        raise BaselineError(f"{kernel.name} does not match its golden output")


def check_equivalence(
    kernel: Kernel,
    optimizer: Optimizer,
    hardware_config,
    *,
    workdir: Path,
    max_cycles: int,
    compare_vmem: bool = True,
) -> tuple[RunResult, RunResult]:
    """
    Run `kernel`, optimize it, run the result, and return (before, after).

    Raises `BaselineError` if the original kernel is broken, `OptimizerError` if
    the optimizer fails, and `EquivalenceError` if the optimized kernel doesn't
    assemble, errors in the simulator, doesn't finish, or leaves different bytes
    in any compared region. `workdir` keeps before.S / after.S for debugging.
    """
    program = kernel.program()
    max_cycles = getattr(program, "kernel_max_cycles", max_cycles)
    dram_regions = compared_dram_regions(program)

    def run(prog) -> RunResult:
        return run_program(
            prog,
            hardware_config,
            dram_regions=dram_regions,
            max_cycles=max_cycles,
            compare_vmem=compare_vmem,
        )

    before = run(program)
    if not before.finished:
        raise BaselineError(f"{kernel.name} did not finish in {max_cycles} cycles")
    _check_golden(kernel, program, before)

    source = kernel.source()
    (workdir / "before.S").write_text(source)
    optimized_source = optimizer(source, workdir)
    after_path = workdir / "after.S"
    after_path.write_text(optimized_source)

    try:
        optimized = program_from_asm(optimized_source, program.memory_regions)
    except Exception as exc:
        raise EquivalenceError(f"does not assemble ({after_path}): {exc!r}") from exc
    try:
        after = run(optimized)
    except Exception as exc:
        raise EquivalenceError(
            "raised during simulation, usually a missing delay (unit backpressure or "
            f"bank conflict) ({after_path}): {exc!r}"
        ) from exc
    if not after.finished:
        raise EquivalenceError(
            f"did not finish in {max_cycles} cycles (before: {before.cycles}) ({after_path})"
        )

    problems = diff_regions(before, after)
    if problems:
        raise EquivalenceError(
            f"ends in a different state ({after_path}):\n  " + "\n  ".join(problems)
        )
    return before, after
