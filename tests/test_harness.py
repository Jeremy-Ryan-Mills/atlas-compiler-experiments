"""
Self-tests: prove the equivalence check passes a no-op and catches both ways an
optimizer can break a kernel. These run regardless of --atlas-opt.
"""

import re

import pytest

from tests import harness

KERNELS = {k.name: k for k in harness.discover_kernels()}
MATMUL = KERNELS["ParameterizedMatmul32x32x32Program"]


def _check(kernel, optimizer, hardware_config_cls, tmp_path, live_state="all"):
    return harness.check_equivalence(
        kernel,
        optimizer,
        hardware_config_cls(),
        workdir=tmp_path,
        max_cycles=100000,
        live_state=live_state,
    )


def _append(*lines):
    """Optimizer that runs `lines` after the kernel's last instruction."""

    def optimizer(source, workdir):
        return source.rstrip("\n") + "\n" + "\n".join(lines) + "\n"

    return optimizer


def test_optimizer_receives_original_source() -> None:
    assert MATMUL.asm_path is not None
    assert MATMUL.asm_path.name == "parameterized_matmul32x32x32.S"
    assert "loop_1:" in MATMUL.source()


def test_identity_is_equivalent(hardware_config_cls, tmp_path) -> None:
    before, after = _check(MATMUL, harness.identity_optimizer, hardware_config_cls, tmp_path)
    assert before.cycles == after.cycles
    assert {"dram_output", "vmem"} <= set(before.regions)
    assert any(label.startswith("dram_input@") for label in before.regions)
    assert before.registers == after.registers


def test_initial_state_is_randomized(hardware_config_cls, tmp_path) -> None:
    """Registers and VMEM the kernel never writes keep their seeded random values,
    except the x registers, which start at 0 as atlas-opt assumes."""
    before, _ = _check(MATMUL, harness.identity_optimizer, hardware_config_cls, tmp_path)
    assert before.registers["x30"] == 0  # matmul never writes x30
    assert any(before.registers["m63"])  # nor m63
    assert any(before.regions["vmem"][-1024:])  # nor the top of VMEM


def test_register_only_change_is_caught(hardware_config_cls, tmp_path) -> None:
    """A dead register write never reaches memory, but `all` must still catch it."""
    clobber = _append("addi x30, x0, 123")
    with pytest.raises(harness.EquivalenceError, match=r"registers differ: x30 \(0x0 -> 0x7b\)"):
        _check(MATMUL, clobber, hardware_config_cls, tmp_path)
    _check(MATMUL, clobber, hardware_config_cls, tmp_path, live_state="dram")


def test_missing_delays_are_caught(hardware_config_cls, tmp_path) -> None:
    with pytest.raises(harness.EquivalenceError, match="raised during simulation"):
        _check(MATMUL, harness.strip_delays_optimizer, hardware_config_cls, tmp_path)


def test_silently_wrong_output_is_caught(hardware_config_cls, tmp_path) -> None:
    """A kernel that runs cleanly but never writes its result must still fail."""

    def drop_stores(source, workdir):
        return re.sub(r"^\s*dma\.store\.ch\d.*$", "", source, flags=re.MULTILINE)

    with pytest.raises(harness.EquivalenceError, match=r"dram_output: \d+/2048 bytes differ"):
        _check(MATMUL, drop_stores, hardware_config_cls, tmp_path)
