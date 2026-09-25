# atlas-compiler-experiments

`atlas-opt` reads Atlas NPU assembly, builds a dependency graph for every basic
block, and reschedules the code so the engines overlap while every dependence and
hardware rule of npu_model's `rtl-match` branch still holds. It then writes the new
program with the minimum `delay`s. [PLAN.md](.agents/PLAN.md) has the design and roadmap;
[OPEN_QUESTIONS.md](.agents/OPEN_QUESTIONS.md) lists what waits until the model is
confirmed RTL accurate.

```sh
git submodule update --init -- third_party/npu_model
cmake -S . -B build -G Ninja && cmake --build build
build/atlas-opt kernel.S -o kernel.opt.S --viz kernel.html
```

Initialize `third_party/atlas-npu` separately for RTL reference.

| Option | What it does |
|---|---|
| `-o FILE` | write the optimized program (otherwise it is printed) |
| `--viz FILE.html` | before/after dependency graph viewer (open it in a browser) |
| `--passes a,b,c` / `--list-passes` | run only some passes / list them |
| `--check` | only simulate the input: cycle count and any broken timing rules |
| `--dma-timing model` | trust npu_model's DMA latency instead of staying valid for any latency |

After optimizing, atlas-opt simulates the result (at npu_model's DMA speed and with
slower DMA) and exits with an error if any timing rule is broken. The viewer shows
each block's dependency graph before and after, with every instruction placed at the
cycle it issues (one lane per engine). Select an instruction to see what it waits
for and why.

Optimization supports conditional branches and `jal x0, label`. It rejects
`jalr`, `jal` with a nonzero link register, and `auipc`, including in delay slots,
because address relocation is unsupported. Ordinary slot delays require
`strip-artifacts` (enabled by default); slot delays marked `# keep` and slot
halts are rejected. `--check` can inspect these instructions without rewriting.

Scheduled halts use a NOP guard after delays, including across labels;
`# keep` preserves the delay immediate. The checker reports unfinished work at
halt. Robust DMA scheduling reserves remaining port use across waits and gaps.

The default `insert-dma-waits` pass adds matching waits before dependent VMEM accesses, changes to DMA operand registers, channel reuse, marked completion signals, and program exits. It tracks pending commands through branches and loops, preserves existing waits, and leaves independent work free to overlap. Unknown addresses are treated conservatively. Programs must start with no earlier DMA work running.

Mark a completion CSR with `# atlas.release`:

```asm
vstore m0, 0(x0)
csrrwi x0, x1, 0xC10 # atlas.release
```

Prior fixed-latency work must finish before the CSR executes. Each possibly pending DMA channel needs a matching wait on every path to release and before reuse; the default pipeline inserts missing waits. Releases in delay slots or pipelines without `schedule` are rejected. Preserve the exact, case-sensitive, whitespace-delimited token during preprocessing; printing and reoptimization retain it.

Releases assume idle entry; unmarked CSR writes do not signal completion to the compiler. A release provides no host acknowledgment, buffer ownership, or IMEM-slot exit proof. `--check` only checks modeled timing and never inserts waits. A custom pipeline without `insert-dma-waits` must still supply the waits required by annotated releases. Wait insertion requires `schedule`; DMA commands retained in branch delay slots also require `strip-artifacts`.

## Layout

| Folder | Contents |
|---|---|
| `src/core/` | Everything about programs and the machine: assembly parsing and printing (`asm`), basic blocks (`blocks`), known register values (`values`), the rtl-match timing rules (`machine`, `reservations`), dependency graphs (`depgraph`), and a timing simulator whose cycle counts match npu_model's (`simulator`) |
| `src/passes/` | The optimization passes, one file each, plus `registry.cpp` listing them in order. **See [src/passes/README.md](src/passes/README.md) to add a pass.** |
| `src/tool/` | The `atlas-opt` command line and the HTML viewer |
| `tests/` | C++ timing tests and Python equivalence and regression tests |

## Equivalence tests

`tests/` also has a pytest harness that runs every kernel registered in
`third_party/npu_model` (a submodule on its `rtl-match` branch) before and after
optimization. It fails if the optimized kernel errors (e.g. breaks a timing rule),
doesn't finish, or leaves different bytes in the kernel's DRAM output, its DRAM
inputs, or VMEM.

```sh
python -m pytest                              # uses build/atlas-opt if it exists
python -m pytest --atlas-opt=identity         # compare unchanged kernels
python -m pytest --atlas-opt=strip-delays     # exercise missing-delay detection
```

Run it with a Python that has npu_model's dependencies: `uv run pytest` (using this
repo's `pyproject.toml`), or `~/Projects/npu_model/.venv/bin/python -m pytest`.
The optimizer is invoked as `<cmd> in.S -o out.S` (settable via `--atlas-opt` /
`$ATLAS_OPT`, extra flags via `--atlas-opt-args` / `$ATLAS_OPT_ARGS`).
`--artifacts-dir DIR` keeps each kernel's `before.S`/`after.S`, and `--max-cycles`
sets the cycle budget. A before→after cycle table is printed at the end.
`SmolVLARmsNormProgram` has a known baseline golden-output failure. Built-in
optimizers skip compiler-specific tests; harness self-tests run independently
of `--atlas-opt`.

`tests/test_regressions.py` covers relocation, both branch paths, halt guards,
and variable DMA latency. `tests/test_publication*.py` checks data and engine
state at the expected completion signal, including MXU source reuse, branches,
loops, and DMA channel reuse. `tests/test_dma_wait_insertion.py` compares automatically repaired programs with explicit-wait references at varied DMA latency. C++ tests check timing, metadata, reservations, wait placement, and repeated optimization.
