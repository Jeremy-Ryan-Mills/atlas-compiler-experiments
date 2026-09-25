# atlas-compiler-experiments

`atlas-opt` reads Atlas NPU assembly, builds a dependency graph for every basic
block, and reschedules the code so the engines overlap while every dependence and
hardware rule of npu_model's `rtl-match` branch still holds. It then writes the new
program with the minimum `delay`s. [PLAN.md](PLAN.md) has the design and roadmap;
[OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) lists what waits until the model is
confirmed RTL accurate.

```sh
git submodule update --init          # third_party/npu_model at rtl-match
cmake -S . -B build -G Ninja && cmake --build build
build/atlas-opt kernel.S -o kernel.opt.S --viz kernel.html
```

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

Optimization supports conditional branches and direct `jal x0, label` jumps.
It rejects `jalr`, `jal` with a nonzero link register, and `auipc` before running
any passes: changing instruction addresses would require relocating indirect
targets, observable link values, and PC-relative values. These checks also cover
delay slots. An ordinary `delay` in a branch/jump delay slot is supported when
`strip-artifacts` runs (as it does by default), because that pass removes it.
A retained slot delay, including one marked `# keep`, and `ecall` or `ebreak` in
a delay slot are unsupported. The `--check` timing simulator can still inspect
these instructions without rewriting the input.

Scheduled halts use a no-op guard after preceding delays, including across
labeled block boundaries. Delays marked `# keep` retain their immediate. The
checker reports unfinished work at the actual halt cycle instead of assuming
that halt drains the engines. Robust DMA scheduling conservatively reserves
remaining port use across waits, including gaps between streamed accesses.

## Layout

| Folder | Contents |
|---|---|
| `src/core/` | Everything about programs and the machine: assembly parsing and printing (`asm`), basic blocks (`blocks`), known register values (`values`), the rtl-match timing rules (`machine`, `reservations`), dependency graphs (`depgraph`), and a timing simulator whose cycle counts match npu_model's (`simulator`) |
| `src/passes/` | The optimization passes, one file each, plus `registry.cpp` listing them in order. **See [src/passes/README.md](src/passes/README.md) to add a pass.** |
| `src/tool/` | The `atlas-opt` command line and the HTML viewer |
| `tests/` | `tests.cpp` (C++ unit tests, `build/atlas-tests`) and the Python equivalence harness |

## Equivalence tests

`tests/` also has a pytest harness that runs every kernel registered in
`third_party/npu_model` (a submodule on its `rtl-match` branch) before and after
optimization. It fails if the optimized kernel errors (e.g. breaks a timing rule),
doesn't finish, or leaves different bytes in the kernel's DRAM output, its DRAM
inputs, or VMEM.

```sh
python -m pytest                              # uses build/atlas-opt if it exists
python -m pytest --atlas-opt=identity         # harness sanity check: everything passes
python -m pytest --atlas-opt=strip-delays     # harness sanity check: everything fails
```

Run it with a Python that has npu_model's dependencies: `uv run pytest` (using this
repo's `pyproject.toml`), or `~/Projects/npu_model/.venv/bin/python -m pytest`.
The optimizer is invoked as `<cmd> in.S -o out.S` (settable via `--atlas-opt` /
`$ATLAS_OPT`, extra flags via `--atlas-opt-args` / `$ATLAS_OPT_ARGS`).
`--artifacts-dir DIR` keeps each kernel's `before.S`/`after.S`, and `--max-cycles`
sets the cycle budget. A before→after cycle table is printed at the end.
`SmolVLARmsNormProgram` fails as a BASELINE error: on rtl-match, the unmodified
kernel already misses its golden output.

The C++ tests check timing and resource reservations directly.
