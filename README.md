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
doesn't finish, or ends in a different architectural state: any byte of the
kernel's DRAM output, its DRAM inputs or VMEM, or any register (`x`, `e`, matrix,
MXU weight and accumulator registers, flags, `dma.base`, the halt status and the
scratch CSRs). The PC and the cycle/instret counters are left out, since
rescheduling changes them. Both runs start from the same seeded random registers
and VMEM (`x` registers stay 0, the reset state atlas-opt assumes), so reading a
value before it is written is caught. `--live-state=dram` compares only the DRAM regions, the PLAN.md §0 contract, for
passes that may leave dead registers or VMEM different.

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
