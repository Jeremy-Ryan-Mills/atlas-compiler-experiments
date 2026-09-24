# atlas-rtlgraph: Deriving the Scheduling Graph from Atlas RTL via CIRCT

Planning document. No code yet.

Companion to `PLAN.md` (`atlas-opt`). That plan builds a dependency graph over **kernel
instructions** and schedules it. Its edge weights and resource rules come from a
machine description that is hand-copied from `npu_model` and then calibrated against
`npu_model` (PLAN.md §1.2, §1.4, §7.1). This project asks a different question:

> Given the Atlas Chisel RTL lowered into CIRCT, can an agentic framework **derive**
> that machine description, meaning the resource graph, the per-instruction timing,
> and the pairwise dependency distances, from the hardware itself? The result would
> make the RTL, not the Python model, the source of truth for `atlas-opt`.

Sources surveyed: `npu_model/npu_spec/00–06`; `npu_model/hardware/{scoreboard,
bank_conflict,idu}.py`; `npu_model/npu_model/out.scala` (the generated Chisel decode
table); `PLAN.md`; and in Merlin (`ucb-bar/merlin@refactor/merlin-phase-architecture-clean`)
the README, `docs/design/rtl_derived_compiler_tooling.md`, `docs/guides/targetgen.md`,
`docs/design/{command_stream_reorder_emitter,macro_scheduling,compiler_plane}.md`,
`merlin/contract/schemas/rtl_facts.schema.json`, `build_tools/scripts/check_isa_matches_rtl.py`,
and `examples/atlas/**` (including `target/descriptor.yaml` and
`phase1/contracts/hwbringup_atlas_v0/schedule_contract.yaml`).

---

## 1. Two graphs, and which one this project produces

"A graph similar to the other agent's" can mean two things. This project produces
the first graph and projects it onto the second.

| | **Hardware Resource-Timing Graph (RTG)** | **Instruction Interaction Graph (IIG)** |
|---|---|---|
| Nodes | Architectural storage (x/m/e RFs and their SRAM banks, weight slots, acc buffers, VMEM banks, `dma.base`, channel-busy bits) and functional units (MXU0/1 sequencers, VPU, XLU, LSU, DMA engines, IDU) | ISA mnemonics (op classes) |
| Edges | Datapaths and control paths in the netlist, annotated with sequential depth, port counts, and FSM occupancy | For a pair `(A, B)` sharing a resource: the minimum issue distance `d(A,B)` and its kind (RAW/WAR/WAW/bank/unit) |
| Derived from | CIRCT `hw`/`comb`/`seq` IR, plus simulation | Projecting the RTG through the decoder |
| Consumer | Humans, the agent, and discrepancy reports | `atlas-opt`: it **is** PLAN.md's §1.2 latency table plus the §5.1 edge-distance table |

`atlas-opt` then instantiates the IIG over a concrete kernel to get the per-kernel
dependency graph. That part is unchanged. The deliverable here is the **machine
description**, built from RTL evidence instead of transcription.

A stretch goal (§6, S7) observes the **dynamic** per-kernel dependency graph directly in
RTL simulation and diffs it against the static graph `atlas-opt` builds.

---

## 2. Merlin, and how this project differs

### 2.1 What Merlin does

Merlin is a *compiler-generation* framework. Its phases are Phase 0 (derive test
"capsules" from hardware), Phase 1 (an agent authors a functional backend compiler,
graded by oracles), and Phase 2 (optimize that compiler). TargetGen ingests docs and
Chisel and deterministically emits target dialect and lowering scaffolding. It has no
LLM; the agent is the compiler author in Phase 1.

Its RTL path (`rtl_derived_compiler_tooling.md`) works like this:

- It elaborates with `firtool --ir-hw`, walks the `circt-opt` HW graph, and finds the
  decoder through `comb.icmp eq` fan-out.
- It extracts **static structural facts**: mesh DIM, scratchpad/accumulator capacity,
  datapath dtypes, and the legal opcode/funct set. These are schema'd in
  `rtl_facts.schema.json` (`family: circt_static`).
- It uses those facts as **preconditions for lowering passes above the ISA**: tile to
  DIM, fit capacity, emit only legal functs, and configure before use. The facts are
  also compiled into FileCheck assertions that pre-screen agent-emitted kernels.
- It uses the arcilator model built from the same RTL (`libatlas_model.so`) as the
  cycle-level oracle.

### 2.2 The gap this project targets

Merlin's own accounting names the gap: RTL-derived checks catch legality and structure,
while "bank conflicts, DMA backpressure, pipeline interlocks … are verilator-only."
Merlin extracts *what the hardware is*, not *when things happen*.

For Atlas specifically, Merlin's `schedule_contract.yaml` already has the right shape:
`minimum_issue_gap` and `register_dependency_gap` entries per producer/consumer class,
with cycle counts. However, its `evidence.sources` fields cite
`npu_model/hardware/{lsu,vpu,mxu}.py`, not the RTL. The timing contract is still
transcribed from the Python model. The same descriptor also records RTL-vs-model
disagreements: `AtlasMemMap.VMEM_SIZE` is 1.5 MiB in RTL but 1 MiB in the model config,
and the MRF capacity obligation is "undecidable". That is direct evidence that the
model and the RTL drift.

### 2.3 Side-by-side

| Axis | Merlin (RTL, then compiler generation) | This project (RTL, then scheduling graph) |
|---|---|---|
| Position in the stack | **Above** the ISA: frontend → target dialect → lowering → encoding | **Below** the ISA: instruction order, issue timing, and `delay` insertion for a fixed ISA program |
| Kind of fact | Static and structural: DIM, capacity, legal encodings, config-before-use | Temporal and microarchitectural: operand read/commit cycles, unit occupancy, initiation interval, port/bank conflicts, interlock vs. no-interlock |
| Why facts matter | A wrong DIM or capacity gives aliasing and wrong results | On a statically scheduled machine without hazard checking, a wrong latency or WAR window **also** gives silently wrong results, and an over-conservative one costs cycles |
| Main CIRCT analysis | Decoder fan-out; memory/SRAM census; parameter constants | Decoder → control-signal → unit mapping; sequential-depth and FSM analysis on datapaths from RF ports to RF ports; ready/valid backpressure paths into the IDU |
| Role of simulation | Oracle for grading generated compilers | **Witness** for every derived timing fact, including a tightness check at `d-1` |
| Role of the agent | Writes the compiler and is graded by oracles | Navigates the netlist, forms hypotheses (which FSM is the MXU sequencer, when `mdst` is written), writes queries and micro-kernels, and reconciles conflicts. It is graded by simulation witnesses |
| Determinism | TargetGen is deterministic; the agent is only in Phase 1 | The extractor core is deterministic C++ over CIRCT. The agent only proposes, and every accepted fact carries a machine-checked witness |
| Output artifact | Dialects, lowering passes, encoders, FileCheck tests | A machine description (`atlas.rtl.yaml`) consumed by `atlas-opt`, plus an RTG `.dot` and a discrepancy report |
| Ceiling | A correct, then fast, compiler from high-level models | A provably safe and tight schedule for hand-written or generated `.S` |

The two are **complementary**. Merlin's Phase 2 optimizer and its
`command_stream_reorder_emitter` both need exactly this timing contract. This project
could become the RTL-evidence producer for Merlin's `schedule_contract.yaml`, which would
move its `evidence.kind` from `target_execution_unit` (Python) to an RTL-derived kind.

---

## 3. What must be derived

These are the facts `atlas-opt` needs. Each row is keyed to the PLAN.md item it replaces
or verifies.

| # | Fact | Where it lives in RTL (hypothesis) | Replaces / verifies |
|---|---|---|---|
| F1 | Mnemonic → unit, operand roles (reads/writes, pair span, implicit state) | Decode table (`AtlasDecode` in `out.scala`: `msrc1`, `msrc2`, `mdst`, `acc_read`, `acc_write`, `mxu_0_valid`, …) | PLAN §4.1 footprint; D2, D3, D8 |
| F2 | Issue → **operand-read** cycle window, per source operand | RF read-port enable timing relative to issue | PLAN §1.3-1. PLAN assumes reads happen at completion, which makes WAR cost `L`. If the RTL reads at issue, WAR drops to ~1. This is the largest potential win |
| F3 | Issue → **result-commit** cycle, per destination | RF / acc / VMEM write-enable timing | PLAN §1.2 latency table (RAW distance) |
| F4 | Unit occupancy and initiation interval | Sequencer FSM / counter busy duration; `ready` back to IDU | PLAN §1.3-3 ("not pipelined in the model") |
| F5 | Interlocks the hardware actually enforces | ready/valid or stall paths from units into the IDU `issue` condition | Which edges are correctness edges and which are only performance edges. Also D4 ("blocking" transfers) |
| F6 | Bank/port structure: SRAM banks, read/write port counts, arbitration | `seq.firmem` / `*_ext` SRAM instances, their addressing, and muxes into them | PLAN §1.3-2 and open question 2: is RAR on the same `m` register really a conflict? |
| F7 | XLU as a separate unit or shared with the VPU | Instance hierarchy and issue-valid signals (`xlu_valid` exists in the decode table) | D1 |
| F8 | DMA timing: queue depth, per-channel vs. shared engine, latency formula in bytes | DMA engine FSMs and TileLink/off-chip interface | PLAN §1.3-4, D7, and the DMA row of §1.2 |
| F9 | `delay N` semantics: exact issue-to-issue distance | IDU countdown register | PLAN §1.3-7, §5.4 |
| F10 | Control flow: delay-slot count and redirect timing | `PcControl` / branch unit | PLAN §1.3-6 |
| F11 | Capacities and address maps (VMEM size, MRF size) | Parameter constants, SRAM depths | Merlin descriptor's VMEM 1.5 vs 1 MiB conflict |

F2, F4, F5, and F6 are where an RTL-derived model is most likely to beat the Python
model. They are also the facts Merlin does not extract.

---

## 4. Architecture

```
 Atlas Chisel (chipyard_atlas, AtlasRocketConfig, generator=atlas)
   │  sbt elaborate → .fir (with @[...scala] source locators)
   ▼
 firtool (names and aggregates preserved; also emit Verilog + arcilator model)
   │  → atlas_hw.mlir   (hw / comb / seq, locations intact)
   ▼
┌──────────────────── deterministic extractor (C++, CIRCT) ───────────────────┐
│ E1 census       instance tree, regs, mems/SRAM banks, ports, ready/valid    │
│ E2 decode       opcode BitPat → control-signal vector (icmp fan-out)       │
│ E3 netgraph     bit-level → signal-level graph; seq depth on every edge     │
│ E4 path query   "from RF read port P to unit U": depth, enables, muxes     │
│ E5 fsm          recover FSMs/counters driving busy/valid; bounds (BMC)     │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │ tool API (CLI + JSON)
┌──────────────────────────────▼───────────── agent layer ────────────────────┐
│ Navigator   maps spec concepts → modules/signals (uses scala locators)     │
│ Extractor   proposes a fact + the queries that justify it                  │
│ Witness     writes micro-kernels; runs arcilator/verilator; checks d, d-1  │
│ Reconciler  diffs vs npu_model, spec, Merlin contract; files discrepancies │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               ▼
 Fact ledger (every fact: value, status, provenance, witness)
   │  project RTG → IIG
   ▼
 atlas.rtl.yaml  ──►  atlas-opt (--machine=rtl)        + rtg.dot + discrepancies.md
```

### 4.1 Why deterministic tools plus an agent, rather than either one alone

- **Pure static analysis is not enough.** Mapping "the cycle `vmatmul.mxu1` commits
  `acc0`" onto the netlist requires knowing which register is the MXU1 sequencer's
  state, which decoded bit starts it, and which write-enable is `acc0`. Names survive
  only partially after lowering. Counter bounds are often parameters threaded through
  several modules, and some latencies are data-dependent (DMA bytes). This is
  navigation and hypothesis work, which suits an agent.
- **A pure agent is not trustworthy.** An LLM reading Chisel will confidently report 35
  cycles because the spec says 35. So the rule is: **the agent proposes, tools and
  simulation dispose.** A fact is `VERIFIED` only when a machine-checked witness passes:
  a CIRCT query result plus a simulation at the claimed distance, and, for distances, a
  detectable failure or monitor fire at `d-1` to prove tightness.
- This mirrors Merlin's stance ("derive, never hardcode"; fail-closed `UNKNOWN`, which is
  "NOT a pass"), applied to a different fact class.

### 4.2 Toolchain choices

- **Extractor core: C++ against CIRCT**, as `circt-opt` analysis passes plus a small
  query CLI (`atlas-rtl-query`). This matches `atlas-opt`'s C++20/CMake stack so the
  two can share `isa.def` / the machine-description loader. Python bindings are an
  acceptable prototyping path.
- **Agent orchestration:** the Claude Agent SDK (or Claude Code with an `AGENT.md`) and
  a fixed tool set: `rtl-query`, `rtl-grep-scala`, `assemble` (reuse `npu_model`'s
  assembler), `sim-arc`, `sim-verilator`, `bmc`, and `ledger-write`. No free shell during
  graded runs.
- **Simulation:** arcilator first (fast, already used by Merlin as the Atlas oracle),
  verilator for anything arcilator can't model, and `circt-bmc` for bounded questions
  like "can `mxu1.busy` fall before cycle 35 after issue?".
- **Monitors:** SystemVerilog-free hazard monitors, attached at the harness level or as
  inserted `seq` probes, that flag RF-bank double access, unit issue while busy, and
  reads of in-flight destinations. These turn silent corruption at `d-1` into a
  detectable signal. Wrong-value detection alone misses cases where stale and new data
  happen to be equal.

---

## 5. Fact ledger and output format

Each fact carries its evidence. The statuses are `VERIFIED` (a witness passed),
`DERIVED` (static evidence only), `UNKNOWN` (fail-closed, and `atlas-opt` must fall back
to conservative values), and `CONFLICT` (sources disagree; see the report).

```yaml
# atlas.rtl.yaml (sketch)
schema_version: 1
family: circt_timing            # complements Merlin's circt_static
inputs:
  fir_sha256: …
  toolchain: {firtool: 1.x, circt-opt: …, arcilator: …}
units:
  MXU1: {occupancy: 35, pipelined: false, interlocked_by_hw: false,
         status: VERIFIED, provenance: [atlas/mxu/ipt/InnerProductTreesTop.scala:…],
         witness: witnesses/mxu1_occupancy.json}
ops:
  VMATMUL_MXU1:
    unit: MXU1
    reads:  [{res: MReg, operand: vs1, span: 1, window: [1, 3]},   # cycles after issue
             {res: WSlot, operand: vs2, window: [1, 35]}]
    writes: [{res: Acc, operand: vd, commit: 35}]
distances:          # projected IIG; atlas-opt reads these directly
  - {a: VMATMUL_MXU1, b: VMATPOP_BF16_ACC_MXU1, res: Acc, kind: RAW, d: 35,
     status: VERIFIED, tight: true}
discrepancies:
  - {id: D1, fact: XLU_separate_unit, rtl: true, npu_model: false, spec: true}
```

**Projection rule (RTG → IIG).** For ops `A` then `B` sharing resource `R`:
- RAW: `d = commit_A(R) − read_start_B(R) + 1`
- WAR: `d = read_end_A(R) − commit_B(R) + 1`
- WAW: `d = commit_A(R) − commit_B(R) + 1`
- Bank or port: derived from port counts and access windows
- Unit: `d = II(unit)`

All results are clamped at ≥ 1. The exact `+1` and `delay` encoding offsets come from F9.
Every projected distance is itself witnessed (§6, S6). The formula is a hypothesis
generator, not a proof.

---

## 6. Stages and deliverables

| Stage | Work | Exit criterion |
|---|---|---|
| **S0 Elaborate** | Reproducible build of the Atlas generator to FIRRTL, then `atlas_hw.mlir`, Verilog, and the arcilator model, with names and source locators kept. Pin toolchain digests | `atlas_hw.mlir` checked into artifacts by hash. Arcilator runs `MatmulProgram` and matches `npu_model`'s golden output |
| **S1 Census (E1)** | Instance tree, registers, memories and banks, ready/valid bundles, all back-annotated to Scala lines | `census.json`. Every SRAM classified (MRF / VMEM / weight / acc / IMEM) or explicitly `UNKNOWN`. Closes F11 |
| **S2 Decode (E2)** | Recover the opcode → control-vector table from the netlist and diff it against `out.scala` and `isa_definition.py` (reusing Merlin's `check_isa_matches_rtl` idea) | F1 for all mnemonics. Coverage is reported first, and uncovered mnemonics are listed |
| **S3 RTG (E3, E4)** | Signal-level graph; sequential depth on RF-port → unit → RF-port paths; `rtg.dot` coloured by unit | The agent can answer "what drives `mdst` write-enable for MXU1" with a query trace |
| **S4 Timing facts (E5 + agent)** | F2–F10: FSM/counter recovery, BMC bounds, and fitting data-dependent formulas (DMA) on training sizes, then checking held-out sizes | Every F-row is `VERIFIED` or has a written reason for `UNKNOWN` |
| **S5 Project** | RTG → IIG; emit `atlas.rtl.yaml`; add `--machine=rtl` to `atlas-opt`'s loader | `atlas-opt` consumes it with no code changes beyond the loader |
| **S6 Witness sweep** | For every distance `d`: micro-kernel at `d` passes, and at `d-1` a monitor fires or the value is wrong. Then run all ~90 `npu_model` kernels, original and `atlas-opt -O3 --machine=rtl`, on arcilator | 100% distance coverage. Golden outputs match on RTL. Cycle report of RTL vs `npu_model` vs `atlas-opt` prediction |
| **S7 Stretch: dynamic graph** | Instrument the RTL sim to log per-instruction issue, read, and commit events. Build the observed dependency graph per kernel and diff it against `atlas-opt`'s static graph | Missing edges (unsafe) = 0. Extra edges (conservative) are counted and reported |

Deliverables: `RTL_GRAPH_PLAN.md` (this file), `atlas.rtl.yaml`, `rtg.dot`,
`discrepancies.md` (resolving PLAN.md D1–D8 and open questions 1–2), the witness corpus,
and an agent run log per fact.

---

## 7. Evaluation (is the agentic part worth it?)

Borrowing Merlin's experimental discipline (`agentic_experiment_integrity.md`, X1–X8):

1. **Blind derivation.** During extraction the agent must not see `npu_model` latency
   tables, the spec's timing tables, or Merlin's `schedule_contract.yaml`. These are
   masked by the sandbox. The reconciler sees them only afterwards. Otherwise
   "derived = model" is circular.
2. **Arms:** (a) deterministic extractor only, (b) agent + tools, (c) agent + tools +
   the spec, (d) a human-transcribed table (today's PLAN.md). Metrics: fraction of F-rows
   `VERIFIED`, number of distances that are **tight**, number of unsafe distances (must
   be 0), agent cost/time, and number of real RTL-vs-model discrepancies found.
3. **Payoff metric:** `atlas-opt` cycles on RTL sim with `--machine=rtl` vs
   `--machine=model`, with correctness required. If F2 (read-at-issue) turns out true
   anywhere, this should show up as a measurable win.
4. **Injected-fault control** (like Merlin's X5): perturb one RTL latency, for example
   add a pipeline register to the MXU1 commit path, and check that the pipeline
   re-derives the new value and `atlas-opt` output changes accordingly. Also check the
   inverse: the model-derived description now produces a kernel that fails on RTL.
5. **Portability probe:** re-run on a second configuration (for example, a different VPU
   lane count) with no human edits.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| firtool optimizations (dedup, inlining, CSE, name dropping) erase the anchors the agent navigates by | Keep names, aggregates, and locators. Run at a low optimization level for extraction, and use a separate optimized build only for simulation speed |
| The full RTL tree is external (Merlin ships only a curated subset: MXU, scalar ISA, params); DMA, LSU, VPU lanes, and the MRF are missing from it | Blocking prerequisite: access to the full `atlas-npu` tree (§9, Q1) |
| DMA and off-chip timing go through Chipyard/TileLink and a serial link, so they aren't deterministic from core RTL alone | Derive the core-side DMA FSM exactly and model the link as a declared parameter with measured bounds. Keep DMA distances `DERIVED` and never `VERIFIED` against real silicon |
| The hardware interlocks some hazards, so some "distances" are only performance hints | F5 marks each edge `correctness` or `performance`. `atlas-opt` can then relax performance edges while keeping them in its cost model |
| Silent corruption at `d-1` looks correct when data happens to match | Hazard monitors (§4.2) plus randomized operand data in witnesses |
| The agent fabricates or anchors on spec numbers | Blind protocol, `VERIFIED` requires a witness, and the ledger rejects facts without provenance |
| Arcilator and verilator disagree, or arcilator lacks support for some constructs | Cross-run a sample of witnesses on both. Fall back to verilator |
| Model drift over time | CI job: re-extract on each RTL bump and diff `atlas.rtl.yaml`, analogous to `check_isa_sync.py` |

---

## 9. Open questions

1. **RTL access:** can I get the full buildable Atlas tree (`atlas-npu/src/main/scala/atlas`
   and the `chipyard_atlas` checkout, `AtlasRocketConfig`), or only Merlin's curated
   subset? S3 and later need the full tree.
2. **Source of truth:** once RTL facts exist, should `atlas-opt` default to
   `--machine=rtl`, with `npu_model` becoming the thing that gets corrected?
3. **Merlin integration:** should this output be contributed upstream as a
   `circt_timing` fact family that feeds Merlin's Atlas `schedule_contract.yaml`, or stay
   standalone in this repo?
4. **Agent runtime:** Claude Agent SDK harness vs. Claude Code + `AGENT.md` +
   restricted tools? What compute and time budget per extraction run?
5. **Scope of "graph":** is the RTG → IIG machine description the goal, or do you also
   want S7 (the dynamic per-kernel graph from RTL traces) in v1?
