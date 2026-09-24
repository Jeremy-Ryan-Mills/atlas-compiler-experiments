# atlas-opt: Dependency-Graph Scheduler for Atlas NPU Assembly

Plan of record for a C++ tool that takes Atlas assembly, builds a dependency graph,
runs optimization passes, and emits a statically scheduled program that:

1. computes the same DRAM result (every data dependency and every hardware-asserted
   scheduling rule honored),
2. keeps the long-chime engines (MXU0, MXU1, VPU, XLU, LSU, DMA) as busy as possible, and
3. spends the fewest cycles on `delay` / idle issue slots.

Reference model: **`npu_model` branch `rtl-match`** (commits `aa95c78`, `bf0f018`). Its
scalar core, IMEM, LSU, MXUs, VPU and XLU are cycle- and bit-matched against Verilator
traces of the Chisel RTL. DMA is still an approximation. Where `npu_spec/` and that
model disagree, the model wins.

Sources surveyed on `rtl-match`: `npu_spec/00–06`, `docs/rtl-timing.md`,
`npu_model/hardware/{core,idu,ifu,exu,lsu,mxu,vpu,xlu,dma,bank_conflict}.py`,
`configs/isa_definition.py`, `util/converter.py`, `tests/test_*rtl*.py`,
`tests/test_mreg_ports.py`, `tests/rtl/README.md`, and all kernels in
`configs/programs/asm/` (5764 lines, 1159 of them `delay`).

---

## Status (2026-09-23)

Built, using only the conservative choices from [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md):
parser/printer, row-timed access profiles for every rtl-match instruction,
dependency graph, reservation table, list scheduler with the robust `dma.wait`
rule, passes P0 (`strip-artifacts`), P3 (`fill-delay-slots`) and P2 (`schedule`),
a timing simulator/checker, and the before/after HTML viewer.

Validation (the pytest equivalence harness in `tests/`, on the rtl-match submodule):
79 of 80 kernels leave identical DRAM output, DRAM inputs and VMEM after
optimization, with no assertion. The 80th, `SmolVLARmsNormProgram`, already misses its
golden output unmodified on rtl-match. atlas-opt's predicted cycles matched
npu_model's for every original and optimized kernel. Total ≈ 596k → 543k cycles
(1.10×); the fused-attention kernels gain 1.32–1.37×.

Changes from the plan below: the code is organized as `src/core` (everything about
programs and the machine), `src/passes` (one file per pass plus a registry; see
`src/passes/README.md`) and `src/tool` (CLI and viewer). The opcode table is a
plain C++ table with an operand list per opcode, not an X-macro, and the tests use a
small built-in runner instead of GoogleTest, to keep the code easy to follow.
Where later sections mention `validate.py`, read "the equivalence harness". The
roadmap is now ordered by the measurements below.

## Roadmap from measurements (2026-09-23)

After the current passes, the 80 kernels are limited by the off-chip DMA link:

| Measure | Value |
|---|---|
| DMA engine busy | 484,516 of 543,363 cycles (89%); VPU 8%, LSU 6%, MXU0 3%, MXU1 ≈ 0% |
| Kernels ≥ 85% DMA-bound | 60 of 80 |
| DMA bytes | 654 KB loaded + 309 KB stored ≈ the whole DMA time at 2 B/cycle |
| Loads of data already brought in | 12% of load bytes (≈ 38k cycles); fused attention 39–42%, batch matmul 50% |
| DMA idle time (ceiling for any reordering) | 58,847 cycles (11%) |

So the order of work is: move fewer bytes, then keep the DMA queue full, then the
compute side. "Blocked on" names the open question or sign-off each idea needs;
"—" means it can be built now.

**1. Move fewer bytes**

| Idea | Upper-bound gain | Blocked on |
|---|---|---|
| Redundant-load elimination: skip a `dma.load` when VMEM still holds that DRAM range unchanged | ≈ 38k cycles (7%); ≈ 40% of attention loads | — for untouched VMEM slots (only 7 KB today); keeping data at other VMEM addresses needs Q4 |
| DRAM round-trip forwarding: drop a load of data an earlier kernel just stored (stores stay; DRAM is live at exit) | up to ⅓ of traffic in chained programs | Q3 |
| Dead-store elimination: drop a store overwritten before anything reads it | small (scratch buffers) | Q3 |
| Transfer coalescing | ≈ 4 cycles per merge, negligible | — |

**2. Keep the DMA queue full**

| Idea | Blocked on |
|---|---|
| P7 timing across blocks: stop draining all engines at block ends; carry in-flight state across loop back-edges | — |
| Fully unroll short loops (many run 2–4 times): removes branch, slot and drain, exposes cross-iteration overlap | — |
| P12 software pipelining: iteration i+1's loads during iteration i's compute | double buffering needs a second VMEM buffer (Q4) |
| Cross-kernel overlap: the next kernel's loads under the previous kernel's tail | Q3 |

**3. Compute side** (the 20 kernels at 60–85% DMA, and larger future kernels)

| Idea | Blocked on |
|---|---|
| P9 MXU binding: 37 matmuls run on MXU0, 16 on MXU1; MXU1 is ≈ 60 cycles faster per matmul and both together double throughput | — (approved) |
| P4/P5 scalar renaming + VLS offset folding: 489 `srli x31` chain every `vload`/`vstore` in program order | — |
| Loop-invariant hoisting / reuse: `vli` constants (104), repeated `vtrpose` + weight push of one tile, reloads of registers that already hold the value, re-pushes of weights already in a slot | — |
| P8 MREG reallocation: break false reuse of m0–m6, avoid mN/m(N+32) bank conflicts | — |
| Dead-code elimination of work that never reaches a `dma.store` | — |
| Bias into the accumulator (`vmatpush.acc` instead of pop + `vadd`) | numerics sign-off |

**4. Scheduler quality:** exact branch-and-bound for small blocks, delay-slot
filling from the branch target, superblocks across branch diamonds, and (after Q1)
a guaranteed minimum DMA latency to drop delays after waits.

**Next, in order:** (1) P7 and short-loop unrolling, (2) loop-invariant hoisting
and redundant-load elimination for untouched slots, (3) P9, P4 and P5. Answering
Q4 early unlocks the largest single gain (redundant-load elimination with VMEM
relocation).

---

## 0. Decisions recorded

| Decision | Consequence for the design |
|---|---|
| Target the `rtl-match` model | All timing below comes from it. The scoreboard / `annotate_delays` work lives on `main` and models every unit as "busy until latency", which is stale here: RAR is now legal, MXUs pipeline, and the VPU has two slots. `atlas-opt` does not depend on it |
| **Only DRAM is live at program exit** | Registers, VMEM, weight slots and accumulators are dead at exit. Renaming is unrestricted at exit, and work whose results never reach a `dma.store` is dead code. Final `dma.store`s and their waits stay |
| **MXU0 and MXU1 are interchangeable** | MXU binding is a default pass, not an opt-in. Both engines have the same throughput (one compute per 32 cycles), and MXU1 has lower latency, so splitting work across them roughly doubles matmul throughput |
| **Eventually, the whole program, not one kernel** | The IR is a whole-program module from day one. Kernel boundaries are ordinary code. Algorithms must be near-linear in program size, and there is a cross-kernel pass family (§5.3, P13) |

---

## 1. Machine model (rtl-match)

### 1.1 Shape

- Two-stage core: **S0** is synchronous IMEM fetch. **S1** is decode + register read +
  scalar execute/writeback + engine launch, all in one cycle. At most one instruction
  issues per cycle.
- Only `delay` and `dma.wait` stall the frontend. **Every other resource conflict is an
  assertion, not a stall.** The emitted schedule must be exact.
- Engines: Scalar (in S1), LSU (three independent paths: scalar, VLOAD, VSTORE), MXU0
  (systolic), MXU1 (inner-product tree), VPU (two slots), XLU (transpose), DMA (FIFO).
- State: `x0..x31` (`x0` = 0), `m0..m63` (1 KiB each; BF16 tiles are even-aligned
  pairs), `e0..e31`, `mxu{0,1}.w{0,1}`, `mxu{0,1}.acc{0,1}`, `dma.base`, 8 DMA channel
  flags, VMEM = 1.5 MiB as **six 256 KiB banks**, IMEM = 128 KiB.
- MREG physical banks: 32 banks, each 1R1W. **`mN` and `m(N+32)` share a bank.** At most
  one read and one write per bank per cycle, and never a read and a write to the same
  physical row in the same cycle.

### 1.2 Frontend timing

- Scalar ALU ops: result visible to the next instruction (RAW distance 1).
- `delay N` issues in its own slot, then holds the next instruction for N cycles. If
  A issues at `t`, then `delay N` issues at `t+1` and the next instruction B issues at
  `t+N+2`. To put B at `t+g`: use no delay if `g=1`, otherwise emit `delay (g−2)`, or
  fill the gap with useful instructions.
- `dma.wait.chN` holds S1 until channel N's flag clears. The flag clears on the tick
  after the transfer completes.
- **Taken branches and jumps have exactly one delay slot. A not-taken branch has
  none.** A control-transfer instruction in the slot is illegal. A `delay` in the slot
  holds the target. Branch/JAL offsets in assembly are **instruction words**. Links are
  `pc+1` (words).
- Existing kernels still end loops with two `addi x0,x0,0` (left over from the
  two-slot spec). The second one runs only on fall-through.

### 1.3 Per-instruction access profiles

The scheduler's timing input. For an instruction issued at cycle `T` (age 0), each
row `r ∈ 0..31` of each operand is read or written at a fixed age. A **reservation
release** is the age at which the logical MREG reservation is dropped; the release
becomes visible on the next cycle.

| Instruction | Engine occupancy | Reads | Writes | MREG release (R / W) |
|---|---|---|---|---|
| scalar ALU, `lui`, `auipc`, `seli`, branches, `jal(r)`, CSR | S1 only | x at T | x/e at T | — |
| `lb/lh/lw/lbu/lhu`, `seld` | LSU scalar. Next scalar load at ≥ T+3 | x[rs1] at T, VMEM at T+1 | x[rd]/e[rd] at T+3, visible at **T+4** | — |
| `sb/sh/sw` | LSU scalar | x at T (latched) | VMEM at T+1 | — |
| `vload` | VLOAD path, next at ≥ T+35 | x[rs1] at T, VMEM row r at T+1+r | m[vd] row r at T+3+r | W 34 (may write during another op's read) |
| `vstore` | VSTORE path, next at ≥ T+35 | x[rs1] at T, m[vd] row r at T+1+r | VMEM row r at T+3+r | R 34 |
| `vmatpush.weight` | MXU read port 1 (else 0), T..T+31 | m[vs1] row r at T+r | w row r at T+1+r | R 32 |
| `vmatpush.acc.fp8` | read port 1 (else 0) | m[vs1] row r at T+r | acc row r at T+1+r | R 32 |
| `vmatpush.acc.bf16` | read ports 0+1 | {m, m+1} row r at T+r | acc row r at T+1+r | R 32 |
| `vmatpop.bf16.acc` | write ports 2+3 | acc row r at T+r | {m, m+1} row r at T+1+r | W 32 |
| `vmatpop.fp8.acc` | write port 2 | acc row r at T+r, e at T | m row r at T+1+r | W 32 |
| `vmatmul[.acc].mxu0` | port 0 T..T+31. **≤ 3 in flight** until T+94 | m[vs1] row r at T+r (+ acc row r at T+r for `.acc`) | acc row r at **T+63+r** | R 32 |
| `vmatmul[.acc].mxu1` | port 0 T..T+31. **≤ 2 in flight** until T+34 | same as MXU0 | acc row r at **T+3+r** | R 32 |
| VPU elementwise (binary, unary, `vmov`, transcendentals) | 1 slot (binary: both). Slot free at T+65 | pair: low row r at T+r, high row r at T+32+r | pair T+2..T+65 | R 63 / W 65 |
| `vpack.bf16.fp8` | 1 slot | pair (vs2) as above, e at T | m[vd] T+3..T+65, every other cycle | R 63 / W 65 |
| `vunpack.fp8.bf16` | 1 slot | m[vs2] row r at T+r, e at T | pair T+3..T+66 | R 31 / W 66 |
| `vred{min,max}.row` / `vredsum.row` | **both slots** | pair row r at T+r (both halves) | pair T+2..T+33 / T+7..T+38 | R 31 / W 33 or 38 |
| `vred{sum,min,max}` (column) | 1 slot | pair, streamed twice, T..T+127 | pair T+66..T+129 | R 127 / W 129 |
| `vli.all/row` · `vli.col/one` | 1 slot | — | pair T+1..T+64 · m[vd] T+1..T+32 | W 64 · W 32 |
| `vtrpose.xlu` | XLU, next at ≥ T+66 | m[vs1] row r at T+1+r | m[vd] row r at T+34+r | R 33 / W 65 |
| `dma.load/store.chN` | DMA FIFO (§1.4) | x regs **at completion**. Store: VMEM at completion | load: VMEM at completion. store: DRAM | — |
| `dma.config.chN` | DMA FIFO, 1 cycle, **sets flag N** | x[rs1] at completion | `dma.base` | — |
| `dma.wait.chN` · `delay N` | S1 hold | flag N | — | — |

The machine description stores exactly these profiles, one row per mnemonic (X-macro
`isa.def`). Every number is checked by tests against the Verilator fixtures that ship
in `rtl-match/tests/rtl/*.json` (§7.1).

### 1.4 Rules the schedule must satisfy

**A. Logical MREG reservations** (ScalarCore issue assertions). Issuing `b` fails if
it conflicts with a reservation still held by an earlier `a`:

- `b` reads or writes a register `a` is writing (RAW, WAW)
- `b` writes a register `a` is reading (WAR), **except** that `vload` may write during a
  read
- **RAR is legal**

**B. Physical ports** (per cycle, not precedence):

- MREG: 1 read + 1 write per physical bank (`reg % 32`), and no read + write to the
  same physical row in one cycle. Inside the VPU, identical (bank, row) reads are shared.
- VMEM: among the LSU paths (scalar at T+1, VLOAD at T+1..T+32, VSTORE at T+3..T+34),
  at most one access per 256 KiB bank per cycle. VLOAD/VSTORE addresses must be 1 KiB
  aligned and must not cross a bank.

**C. Engine rules:**

| Engine | Rule |
|---|---|
| Scalar / LSU | Scalar load at T: no scalar op writing x[rd≠0] or `seli` may issue at T+3 (writeback-port collision). No second scalar load before T+3 |
| VPU | ≤ 2 live ops (live = age < write-last). Binary and row-reduce ops need the VPU empty and block everything else. Ops in the same logic group cannot overlap: {add, sub, redsum.row}, {exp, exp2}, {sin, cos}, {square, cube}, {max, redmax}, {min, redmin}, {vli.*}, and any op with itself (e.g. two `vmov`s). Pair operands must be even |
| MXU (both) | Any non-weight command touching acc `a` waits until row 0 of an in-flight compute on `a` is written: age > 63 (MXU0) / > 3 (MXU1). A compute and a pop may not read the same acc in the same cycle. A compute and a push may not write the same acc in the same cycle. At most one weight push and one acc push stream at a time (the port-1/port-0 output mux) |
| MXU0 | Weight push to slot s ≥ T+63 after a compute using s. Compute may start right after a weight push to the same slot (the wavefront trails the push) |
| MXU1 | No compute while a weight push to its slot (age < 32) or an acc push to its acc (age ≤ 32) is active. No weight push while a push or compute on that slot is active |
| XLU | One transpose at a time |
| DMA | Head-of-line FIFO, depth 8, one transfer progressing at a time across all channels. Latency = `max(ceil((B+8)/4)·2, ceil(B/64))`, i.e. 516 cycles for 1 KiB. Issuing to a channel whose flag is set is illegal. Queued transfers must touch disjoint 32 B VMEM ranges |

**D. Data correctness** (row-level). Every read of row r must see the intended write
of row r. This is what the logical reservations *don't* fully cover: accumulators,
weight slots, VMEM, and the `vload`-during-read exception.

### 1.5 Known model limitations (treated conservatively)

| # | Issue | Policy |
|---|---|---|
| L1 | DMA timing is an approximation. Real TileLink latency varies | Default `--dma-timing=robust`: never assume *when* a `dma.wait` releases (§5.2). `--dma-timing=model` uses the model's exact FIFO timing, for experiments |
| L2 | The model's DMA reads its `x` operands and `dma.base` at **completion** | Keep DMA operand registers unmodified until the matching `dma.wait` retires. Scalar renaming (P4) makes this nearly free |
| L3 | LSU-vs-DMA VMEM conflicts are not checked | Order them through `dma.wait` plus VMEM alias analysis (a correctness requirement anyway) |
| L4 | `npu_spec` numbers differ from the model (e.g. MXU0 96 vs 95 inclusive, `vli.col/one` 65 vs 33, `vunpack` 66 vs 67). Some `HardwareConfig` latency fields are unused | Use the model and ignore those config fields. `check_isa_sync.py` (§3.1) catches drift |
| L5 | VLS bases are **word** addresses, while DMA and scalar addresses are bytes. The kernels emit `srli x31, xN, 2` before every `vload`/`vstore` | Value analysis tracks both domains (§4.2). Passes P4/P5 remove the `x31` serialization |

---

## 2. Where the wins are

Current kernels are hand-serialized: `op; delay L; op; delay L`. With the `rtl-match`
engine model:

- **MXU pipelining + binding.** Each MXU accepts a compute every 32 cycles (MXU0 keeps
  up to 3 in flight, MXU1 up to 2). A pop can start as soon as row 0 of the result is
  written (T+64 on MXU0, T+4 on MXU1). Today, push → `delay 32` → matmul → `delay 35/96`
  → pop runs one tile at a time on one MXU. The target is both MXUs streaming a tile
  each per 32 cycles, with pushes on port 1 overlapping computes on port 0.
- **VPU dual-issue.** Two single-input ops from different logic groups overlap
  (e.g. `vexp` + `vrecip`). Sources are released at T+64, so the next op that writes a
  source can start before the current one's results drain.
- **LSU overlap.** VLOAD, VSTORE and scalar paths run concurrently when they touch
  different VMEM banks. A `vload` into a register that a `vstore` is still reading is
  legal right away (the write trails the read).
- **Scalar false dependencies.** The shared `x31` word-address temporary chains every
  VLS op together. Renaming and folding offsets into the VLS immediate (32 words per
  unit) removes those chains.
- **DMA-bound kernels.** Example: `elementwise_add64x64` spends about 3 × 1028 = 3084
  cycles per iteration in the DMA FIFO, versus about 270 of compute. The roof there is
  keeping the FIFO full: issue iteration i+1's loads during iteration i (software
  pipelining), and across kernel boundaries, the next kernel's loads during the
  previous kernel's tail (P13). The report prints this bound so we know when a kernel
  has hit it.
- **Delay slot.** One slot per taken branch can do useful work, and the legacy second
  nop goes away.

---

## 3. Architecture

```
  .S (one or many files)
    │  Lexer/Parser: labels, word-offset branches, li/nop, # comments, all mnemonics incl. CSR
    ▼
  Module IR (whole program; kernel files become regions)
    │  strip scheduling artifacts (delay, filler nops)
    ├─ CFG (1-slot taken branches), loops, regions
    ├─ Scalar value analysis (byte + word address domains, induction variables)
    ├─ Access-profile analysis (per insn: resources × row-timed R/W, reservations, ports)
    ├─ Liveness (x, m, e, w, acc, VMEM ranges; only DRAM live at exit)
    ▼
  Dependency graph per scheduling region, edge distances derived from profiles (§5.1)
    │
    ├─ Pass pipeline (§5.3)
    ▼
  Schedule (issue cycle per insn) ──► internal timing checker (mirrors §1.4 exactly)
    │  Emitter: gaps → `delay N` or fillers, delay slots, labels
    ▼
  .S  +  report.json (cycles, per-engine utilization, lower bounds)  +  graph.dot
```

### 3.1 Source layout

```
atlas-compiler-experiments/
  CMakeLists.txt
  src/
    core/     asm (opcodes, parser, printer), blocks, values, machine (timing rules),
              reservations, depgraph, simulator
    passes/   pass.h, registry.cpp, one .cpp per pass, README.md (how to add a pass)
    tool/     main.cpp (atlas-opt), viewer (HTML)
  tests/      tests.cpp (C++ unit tests) + the pytest equivalence harness
  third_party/npu_model   submodule at rtl-match (the oracle)
```

Toolchain: C++20, CMake ≥ 3.20, no runtime dependencies, `-Wall -Wextra -Werror`,
ASan/UBSan in debug. `check_isa_sync.py` imports `npu_model` (rtl-match) and diffs
mnemonics, engine assignments and latency tables against `isa.def`.
`check_profiles.py` diffs our access profiles against `tests/rtl/*_traces.json`.

---

## 4. IR and analyses

### 4.1 Instruction IR

```cpp
enum class Engine : uint8_t { Scalar, LsuScalar, LsuVload, LsuVstore, Mxu0, Mxu1, Vpu, Xlu, Dma, S1Hold };

struct Resource {        // one dependency-carrying location
  enum Kind : uint8_t { XReg, MReg, EReg, WSlot, Acc, DmaBase, DmaChan, Vmem, VmemUnknown };
  Kind kind; uint32_t index;          // Vmem: 32 B line (or a symbolic range)
};

struct RowAccess {       // a row-timed stream: row r touched at age first + r*stride
  Resource res; bool write; uint16_t first, stride, rows;
};

struct Instr {
  const OpInfo* op;                    // row from isa.def
  std::array<int32_t, 4> operands;
  SymbolRef target;                    // branch label
  SmallVector<RowAccess, 6> accesses;  // resolved per instance (pairs, VMEM ranges)
  Reservation mreg;                    // read/write sets + release ages (§1.3)
  PortUse ports;                       // engine slots/ports with age windows
  SourceLoc loc;
};
```

Pair expansion (`{m[r], m[r+1]}`), even-alignment checks, implicit operands
(`dma.base`, weight slots, accumulators, `e`) and VMEM ranges are all resolved into
`accesses` in one place. The graph and scheduler never special-case mnemonics.

### 4.2 Scalar value analysis

- Lattice per `x`: `⊥ | Const(c) | Affine(base + stride·iv) | ⊤`, with RV32 semantics
  (`lui/addi/add/slli/srli/…`).
- Tracks two address domains: **byte** (DMA, scalar LSU) and **word** (VLS, via
  `srli 2`), so the `srli x31` idiom resolves to exact VMEM lines and 256 KiB banks.
- Loops: basic induction variables and trip counts from `blt/bge/bne` exits. Each
  memory op gets a VMEM range that is exact, affine per iteration, or ⊤ (may alias
  everything: conservative but still correct).
- DMA byte count resolves to a constant, giving DMA latency and FIFO occupancy. If it
  doesn't resolve, that DMA is not moved.

### 4.3 CFG and regions

- Blocks split at labels and after branches. The one delay-slot instruction is pinned
  to its branch.
- **Regions** are the scheduling units: basic blocks → superblocks along the hot path
  → loop bodies (for modulo scheduling). Region size is capped (e.g. 2k instructions)
  so whole-program runs stay near-linear.

---

## 5. Dependency graph, scheduler, passes

### 5.1 Edges derived from profiles

Construction uses per-resource last-writer / live-readers tables, so it is O(n · k),
not O(n²). For `a` before `b` on a shared resource, the minimum issue distance
`d(a,b)` is the maximum of:

| Hazard | Reservation term (MREG only) | Row-data term |
|---|---|---|
| RAW | `d ≥ W_a + 1` | `∀r: d + readAge_b(r) > writeAge_a(r)` |
| WAR | `d ≥ R_a + 1` (skipped when `b` is `vload`) | `∀r: d + writeAge_b(r) > readAge_a(r)` |
| WAW | `d ≥ W_a + 1` | `∀r: d + writeAge_b(r) > writeAge_a(r)` |
| RAR | none | none (port exclusion only) |

These terms are plus explicit **engine-rule edges** from §1.4C (e.g. MXU0 compute on
acc a → next command on a: 64. MXU0 compute → weight push on the same slot: 63.
MXU1 weight push → compute on the same slot: 32. Scalar load → consumer: 4). Examples
the formula reproduces: `vload m0` → `vadd` reading m0: 35. `vstore m0` → `vload m0`:
1. VPU op → `vload` into its high source register: 30. MXU0 `vmatmul` → `vmatpop`: 64.

**Exclusion constraints** are *not* edges. They are checked per cycle in a reservation
table: VPU slots and logic groups, MXU ports and in-flight limits, LSU paths, VMEM bank
ports, MREG physical ports, the scalar writeback port, and the XLU. Modulo scheduling
uses a modulo reservation table.

**DMA:** transfer → `dma.wait` on the same channel → consumers of the VMEM range (RAW
through VMEM). Producers of a store's VMEM source → transfer. Anything writing a
store's source or a load's destination waits for that channel's `dma.wait`. Channel
reuse needs the wait in between.

Graph utilities: critical-path height, slack, Graphviz export (colored by engine and
edge kind), and the lower bound

```
LB = max(critical path, max over engines of Σ occupancy, Σ DMA FIFO cycles, #instructions)
```

### 5.2 Scheduler core

- **List scheduler** (default): cycle-driven, one issue per cycle. Picks the
  highest-priority ready instruction whose reservation-table checks pass. Priority is
  critical-path height, then DMA-first (keep the FIFO fed), then source order.
- **Exact scheduler** for small regions (≤ ~25 instructions): branch-and-bound seeded
  with the list schedule and pruned by LB.
- **Modulo scheduler** for counted loops (§5.3 P12).
- **Timing checker**: re-simulates the emitted program with the §1.4 rules
  (reservations, ports, engine rules, row-level data). It is the internal oracle and
  runs on every pass result in debug and `--verify` builds.
- **`dma.wait` is a timing re-sync point.** Under `--dma-timing=robust`, the wait may
  take any time ≥ 1 cycle. Precedence edges across a wait are safe for any duration
  (a longer wait only increases distances), but **exclusion constraints are not
  monotone**. A later post-wait instruction can collide with a later window of a
  pre-wait instruction still in flight. So for exclusion checks, pre-wait in-flight
  instructions occupy the hull of their remaining windows. This matters for
  correctness on silicon, and the tests cover it explicitly.

### 5.3 Pass pipeline

| # | Pass | What it does | Level |
|---|---|---|---|
| P0 | `strip-artifacts` | Remove `delay N` and filler `addi x0,x0,0`, including the legacy second post-branch nop. `delay N # keep` survives | O1 |
| P1 | `cleanup` | Drop `x0`-writing no-ops, `dma.wait`s on channels known idle, and `dma.config`s that rewrite an identical `dma.base` | O1 |
| P2 | `local-schedule` | List/exact scheduling per block. Engines drain at block boundaries (safe baseline) | O1 |
| P3 | `delay-slot-fill` | The word after a branch runs on **both** paths: as the slot if taken, as the next instruction if not. Fill it with an instruction from above the branch that the branch condition doesn't depend on, or with one from either successor that is dead or harmless on the other path (liveness makes this checkable) | O1 |
| P4 | `xreg-rename` | Rename scalar registers to break WAR/WAW (above all the `x31` chain). Respects DMA operand lifetimes (L2) and 31 usable registers | O2 |
| P5 | `vls-addr-fold` | Compute each buffer's word base once and fold offsets into the VLS `imm12` (32 words per unit). Deletes most `srli` instructions | O2 |
| P6 | `dma-hoist-sink` | Issue `dma.load` as early as its operands, channel and VMEM range allow. Sink `dma.wait` to just before the first reader/writer of the range | O2 |
| P7 | `interblock-timing` | Replace drain-at-boundary with propagated engine/resource state (join = per-resource max). Loop headers iterate to a fixed point | O2 |
| P8 | `mreg-rename` | Reallocate `m` registers: even pairs, avoid `mN`/`m(N+32)` port clashes, break WAR/WAW. Needs liveness (dead at exit) | O3 |
| P9 | `mxu-bind` | Default on. Assign each matmul group (weight push → computes → pop, connected through w/acc) to MXU0 or MXU1, and allocate weight slots and accumulators, to balance load. Numerics are treated as equal per project decision | O3 |
| P10 | `dma-chan-assign` | Recolor DMA channels (interval-graph coloring over issue → wait lifetimes) so more transfers can be queued | O3 |
| P11 | `vmem-place` | *Later:* move buffers across the six 256 KiB banks so VLOAD/VSTORE overlap, rewriting the address constants found by value analysis. Invasive, so behind a flag (open question 4) | flag |
| P12 | `modulo-schedule` / `unroll` | Software-pipeline counted loops (iterative modulo scheduling, ResMII from the modulo reservation table, where DMA usually dominates). Prologue/epilogue, trip-count fixup, rotating buffers via P8/P10. Unroll by 2 or 4 as a fallback | O3 |
| P13 | whole-program family | Cross-kernel scheduling (next kernel's DMA loads under the previous kernel's tail). Redundant `dma.config`/constant elimination across kernels. **DRAM round-trip forwarding**: a load of a range the program just stored, while VMEM still holds it, becomes nothing or a VMEM copy. Stores stay, because DRAM is live at exit | O3 |
| P14 | `emit` | Gaps become `delay (g−2)` or fillers. Labels re-resolved. Optional `# c=<cycle> e=<engine>` annotations | always |

After every pass, debug builds check: acyclic region graphs, zero timing-checker
violations, every DMA transfer matched by a wait before channel reuse, and every
instruction's even-pair and alignment rules.

### 5.4 CLI

```
atlas-opt a.S [b.S …] -o out.S          # several files = one program, in order
          [-O0..-O3] [--passes=…] [--disable=…] [--enable=vmem-place]
          [--dma-timing=robust|model] [--verify]
          [--report=report.json] [--dot=graph.dot] [--annotate]
```

---

## 6. Correctness argument

For every pair of instructions sharing a resource, the emitted issue distance
satisfies the derived edge, and every per-cycle exclusion constraint holds. Three
layers enforce this:

1. **Construction.** Every shared resource gets an edge. Unresolved VMEM addresses
   (⊤) alias everything.
2. **Internal timing checker.** It replays the emitted program with the §1.4 rules,
   including the `dma.wait` robustness rule.
3. **External oracle (the equivalence harness in `tests/`).** Runs the output on `npu_model@rtl-match`,
   which *asserts* every RTL scheduling violation (MREG reservations and ports, VPU
   issue-busy, MXU ports and ordering, LSU paths, VMEM banks, delay-slot legality).
   Requirements: no assertion, golden output matches, and cycle count ≤ the original.
   The rtl-match model is itself the RTL-derived checker, so the `main`-branch
   scoreboard is not needed here.

---

## 7. Verification and measurement

### 7.1 Profile validation (first thing to build)

- `check_profiles.py` extracts every MREG read/write cycle from
  `tests/rtl/{vector,sa,ipt,memory,scalar}_traces.json` (real Verilator output) and
  diffs them against `isa.def` profiles.
- Micro-kernels (2–3 instructions each) run on the rtl-match model to find the
  **minimum legal distance** for each edge kind and engine pair. They should match
  §5.1's derived distances exactly. Off-by-one in either direction fails the test.
- Both suites stay as regression tests. A timing change in `npu_model` fails CI
  instead of silently producing wrong schedules.

### 7.2 Test layers

- **Unit:** parser round-trip of every rtl-match kernel (parse → print → parse gives
  identical IR). Profile expansion per mnemonic. Value analysis (byte/word domains,
  induction variables). Edge derivation on hand-built snippets. Schedulers on synthetic
  DAGs with known optimal makespan. The `dma.wait` robustness rule.
- **Property:** random legal programs generated from `isa.def`, scheduled and
  cross-checked by the timing checker and the rtl-match model.
- **Golden:** all registered rtl-match kernels pass the equivalence harness.
- **Whole-program:** `compose.py` builds multi-kernel programs by relocating each
  kernel's DRAM regions and concatenating them. Golden = each kernel's golden at its
  relocated address. There are also producer→consumer chains (e.g. matmul → bias →
  SiLU) that exercise P13 forwarding. No such corpus exists in `npu_model` today.
- **Performance report:** per kernel and per composed program: original cycles,
  optimized cycles, LB, and per-engine utilization. It also tracks compile time
  against program size (target: 10⁵ instructions in seconds).

---

## 8. Milestones

| M | Deliverable | Exit criterion |
|---|---|---|
| M0 | CMake skeleton, `isa.def` with profiles, parser/printer, `check_isa_sync.py`, `check_profiles.py`, micro-kernel suite, baseline cycle table | All kernels round-trip. Profiles match the RTL fixtures. Baseline recorded |
| M1 | IR, value analysis (straight-line), liveness, dep graph, `.dot`, LB | Graphs and lower bounds for every kernel |
| M2 | Timing checker, P0–P3, P14, `validate.py` | `-O1` passes golden with no assertions on every kernel |
| M3 | P4, P5, P6, P7, loop-aware value analysis | `-O2` passes. The `x31` chains are gone. DMA-bound straight-line kernels within ~5% of the DMA roof |
| M4 | P8, P9, P10, exact scheduler | `-O3` (without P12/P13) passes. Matmul/attention kernels keep both MXUs and the VPU overlapped |
| M5 | P12 modulo scheduling + unroll fallback | Loop kernels approach `ResMII × trips` |
| M6 | Whole-program: multi-file input, `compose.py` corpus, P13, scalability work | Composed programs pass. Cross-kernel overlap shows up in the report. 10⁵-instruction inputs compile in seconds |
| M7 | P11 `vmem-place` (if approved) | LSU overlap on kernels that were bank-limited |

Correctness infrastructure (M0–M2) comes before the aggressive passes on purpose.
Every later pass is only as trustworthy as the profiles and the oracle.

---

## 9. Open questions

Tracked in [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md). They are deferred until the
`rtl-match` model is confirmed RTL accurate. Until then, each one uses the
conservative choice listed there.
