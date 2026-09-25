# Automatic DMA-wait insertion

This documents the automatic DMA-wait insertion changes against merged `main` at `4cb2dc173e4730e75943b7362db41c7d01976bdb`. The compiler now inserts missing `dma.wait.chN` instructions in its default pipeline. RTL, NPU-model sources, and existing kernels are unchanged. Source links below are relative to this repository and refer to the working files.

## High-level changes

1. Infer DMA waits before dependent accesses, channel reuse, marked completion signals, halts, and program exits.
2. Carry pending DMA commands through branches and loops, including exits reached through labels or conditional fallthrough.
3. Use the same completion-conflict rules for wait insertion and scheduling, so the scheduler preserves the required ordering.
4. Preserve explicit waits and the original rejection checks when automatic insertion is disabled.
5. Validate repaired programs against explicit-wait references and compare both correctness and per-kernel cycles with the merged baseline.

## 1. Insert waits where they are needed

`insert-dma-waits` runs after `strip-artifacts` and before `fill-delay-slots` and `schedule`. It records which DMA commands may still be pending on each channel. An existing matching wait clears that channel. Otherwise, the pass inserts a wait before the first conflicting access, reuse of that channel, an `atlas.release` signal, or a program exit. A fixed `delay` does not clear pending DMA, even if it would be long enough at one modeled latency.

For example, the pass turns this input:

```asm
addi x7, x0, 32
dma.load.ch0 x0, x0, x7
addi x8, x0, 9
lw x2, 0(x0)
```

into this order before scheduling:

```asm
addi x7, x0, 32
dma.load.ch0 x0, x0, x7
addi x8, x0, 9
dma.wait.ch0
lw x2, 0(x0)
```

The independent `addi` can overlap the transfer. The existing scheduler then supplies fixed delays and chooses issue times. This repair also applies to programs without `atlas.release`; the annotation is still needed to identify a CSR write as a completion signal.

The dependency checks cover overlapping VMEM accesses and writes to scalar registers that a pending DMA command will read at completion. The latter includes address, length, and configuration operands and follows the pinned model's operand lifetime. Known disjoint VMEM ranges can overlap; an unknown address may alias any range. Changes to the shared DMA base between queued DMA commands retain the existing FIFO ordering rule.

| File | Specific change |
| --- | --- |
| [insert_dma_waits.cpp](src/passes/insert_dma_waits.cpp#L86-L104) | Compute which pending channels must finish before the next instruction, including same-channel reuse and completion boundaries. |
| [insert_dma_waits.cpp](src/passes/insert_dma_waits.cpp#L150-L251) | Compute footprints from scalar value analysis, plan missing waits, emit them in channel order, and report the number inserted. Existing waits remain intact. |
| [depgraph.cpp](src/core/depgraph.cpp#L33-L48), [depgraph.h](src/core/depgraph.h#L31-L32) | Share completion-conflict detection between insertion and scheduling. Queued DMA commands with overlapping VMEM ranges are ordered even when both read, matching the timing checker's existing restriction. |

## 2. Preserve control flow and useful overlap

Each channel carries a set of possible DMA command sites. At a control-flow join, the pass unions the predecessor sets, so a wait is required if any reaching path needs it. Loop backedges participate in the same analysis. Planned waits only accumulate; after each planning round, the pass recomputes pending commands with those waits clearing their channels. This avoids inferring safety from only the first loop iteration or one branch outcome.

Waits needed by an instruction retained in a branch delay slot are placed before its branch; the slot and branch condition retain their order. The default stripping pass can move DMA launch instructions out of delay slots. A custom pipeline that retains such launches must include `strip-artifacts` before insertion. Existing wait instructions in delay slots remain supported.

An end label can be a branch target without already having a basic block. The pass creates an exit block when that path needs a wait, also covering the final conditional branch's fallthrough. This avoids draining every loop backedge merely because another path exits. Waits for ordinary unlabeled falloff are folded into the last block. When no wait is needed, insertion leaves the original blocks unchanged.

| File | Specific change |
| --- | --- |
| [insert_dma_waits.cpp](src/passes/insert_dma_waits.cpp#L60-L84) | Propagate reaching DMA commands through the CFG until the pending state stabilizes. |
| [insert_dma_waits.cpp](src/passes/insert_dma_waits.cpp#L107-L145) | Validate reachable successors and account for end-label targets and conditional fallthrough exits. |
| [insert_dma_waits.cpp](src/passes/insert_dma_waits.cpp#L183-L249) | Plan waits until stable, omit newly planned waits proved idle, preserve explicit waits, and retain only necessary exit-block changes. |

## 3. Integrate with compiler validation

The default pipeline repairs missing waits before running the existing static release checks. Those checks still run after subsequent passes, so annotated signals cannot lose required synchronization during scheduling. Invalid annotations, unsupported relocation, malformed reachable CFG edges, and unsupported delay-slot configurations are rejected before earlier passes modify the program. Automatic insertion requires the scheduling pass.

To retain the previous manual-wait behavior, select the old pipeline explicitly:

```sh
build/atlas-opt kernel.S --passes strip-artifacts,fill-delay-slots,schedule -o kernel.opt.S
```

For annotated programs, this still rejects missing waits and unsafe channel reuse. `--check` remains a timing inspection of the supplied input; it never inserts waits and does not enforce the optimizer's all-path explicit-wait requirement.

| File | Specific change |
| --- | --- |
| [registry.cpp](src/passes/registry.cpp#L92-L183), [pass.h](src/passes/pass.h#L27-L31) | Register the new pass, validate its prerequisites, and defer pending-channel rejection until missing waits have been inserted. |
| [README.md](README.md#L42-L53) | Document automatic insertion, idle-entry assumptions, custom pipelines, and the distinction from `--check`. |

## 4. Regression coverage

Missing-wait inputs are not a valid equivalence oracle: their asynchronous execution may already use incomplete data. The new model tests instead compare optimized repairs against separate programs containing correct explicit waits. They initialize nonzero data, compare scalar registers, VMEM, DRAM, selected matrix data, configuration state, and halt status, and test normal and 100× DMA latency. Completion tests additionally observe data at the requested CSR completion value before later engines tick.

The C++ tests inspect wait placement, preserved overlap, existing-wait retention, control-flow exits, rejection without mutation, and repeated insertion. Timing checks use DMA scales 0.1×, 1×, 10×, and 100×. Existing publication tests that deliberately require rejection now omit insertion explicitly; this preserves coverage of the manual-wait contract.

| File | Specific change |
| --- | --- |
| [dma_wait_insertion_tests.cpp](tests/dma_wait_insertion_tests.cpp#L53-L177) | Add direct-pass and scheduled-output checks for memory dependencies, operands, channels, boundaries, branches, loops, and preflight failures. |
| [test_dma_wait_insertion.py](tests/test_dma_wait_insertion.py#L65-L298) | Add 44 model cases with explicit-wait references, varied DMA latency, nonzero expected data, and reoptimization checks. |
| [publication_dma_tests.cpp](tests/publication_dma_tests.cpp#L9-L55) | Keep prior C++ rejection tests on the explicit manual-wait pipeline. |
| [test_publication.py](tests/test_publication.py#L35-L45), [test_publication_cfg.py](tests/test_publication_cfg.py#L79-L107), [test_publication_channel_reuse.py](tests/test_publication_channel_reuse.py#L40-L91) | Preserve Python rejection coverage when insertion is disabled. |
| [CMakeLists.txt](CMakeLists.txt#L40-L42) | Register the new C++ test executable with CTest. |

## 5. A/B validation

The baseline compiler is built from the exact merged `main` commit `4cb2dc1`. Both sides use identical unmodified model sources from `bf0f0183a2230fb71d614d7f9e2777dca0c42716` in isolated temporary snapshots. The existing Python environment supplies Python 3.14.7, PyTorch 2.11.0, NumPy 2.4.4, and pytest 9.0.3. No model working-tree edits are used.

| Check | Baseline `4cb2dc1` | Updated compiler |
| --- | ---: | ---: |
| Original full suite | 142 passed, 1 known baseline failure | See unchanged-test comparison below |
| Same current 187-test suite | 142 passed, 45 failed | **186 passed, 1 known baseline failure** |
| New automatic-wait model cases within that suite | 0 passed, 44 failed | **44 passed, 0 failed** |
| CTest | 3/3 passed | **4/4 passed** |
| Original valid kernel corpus | 79 equivalent kernels | **79 equivalent kernels** |
| DMA-using kernels with every wait replaced by `nop` before compilation | 0/78 passed | **78/78 passed** |

The known failure on both sides is the original, unoptimized `SmolVLARmsNormProgram` missing its golden output. The baseline’s other 44 failures in the current suite are the new repair cases. Running the completely unchanged old suite against the new default pipeline produces 133 passes and 10 failures: the known RMSNorm failure plus nine tests whose old expectation was rejection of missing waits. The corresponding tests in the current suite still verify that rejection with insertion disabled and all pass. No original kernel changes its checked data.

For the removed-wait experiment, a wrapper substitutes `nop` for each actual `dma.wait.chN` instruction before invoking the compiler, preserving labels and branch-slot positions. The harness compares the compiled result against the original kernel with its explicit waits. All 78 DMA-using kernels are repaired; `AddiProgram`, which has no DMA waits, passes both versions. Including the registry test and known RMSNorm failure, pytest reports 2 passed / 79 failed for the baseline and 80 passed / 1 failed for the updated compiler.

### Performance on unchanged valid inputs

| Workload | Original kernel | Baseline optimized | Updated optimized | Change from baseline optimizer |
| --- | ---: | ---: | ---: | ---: |
| Total, 79 comparable kernels | 593,460 | 540,775 | 540,784 | +9 cycles, approximately 0.0017% |
| `ParameterizedMatmul64x32x96Program` | 11,128 | 11,051 | 11,060 | +9 cycles, approximately 0.0814% |
| Fused attention Q32/K64 | 18,867 | 14,320 | 14,320 | Unchanged |

**78 of 79 kernels keep exactly the same cycle counts.** The matmul adds two static waits on channel 1: before outer-loop reuse and at final exit. Its prefetch runs only when `x21 < x11`, and its inner loop repeats under that same condition. The existing inner-loop wait therefore consumes every actual prefetch. The new analysis unions control-flow paths without remembering the relationship between those two branch conditions, so it conservatively considers DMA pending on an infeasible exit path. The first new wait also constrains nearby scheduling. Eliminating this small cost safely would require more precise path analysis or wait placement; this change does not claim zero performance regression.

### Additional control-flow checks

A deterministic generator exercised 3,000 programs with joins, loops, multiple DMA command sites, scalar operand changes, release signals, and exits. Both first and second optimized outputs passed timing checks at 0.1×, 1×, 10×, and 100× DMA latency: **24,000 timing checks passed**. Running insertion again after stripping scheduling artifacts added no new waits in any program: **3,000 insertion-idempotence checks passed**.

Forty-two generated programs changed unrelated instruction order during a second full optimization. The retained seed-24 example produces exactly the same reordered output with the baseline compiler, establishing that example as existing scheduler behavior. Repeated wait insertion is stable; universal byte-identical output from the whole scheduler is not claimed. These generated checks use the C++ timing simulator, separately from the 44 Python model tests.

### Retained results and reproduction

Source snapshots, exact commands, logs, assembly outputs, and CSV cycle tables are under `/tmp/atlas-dma-wait-validation/`. The baseline snapshot contains compiler `4cb2dc1`; `candidate-final` contains the tested working sources. Both have the same exported model pin. `results/unchanged-input-cycle-comparison.csv` records all 79 kernel comparisons, and `artifacts/fuzz/REPORT.md` records the generated-program checks and their reproduction commands. These files are local temporary artifacts, not repository changes.

The full suite was run from `candidate-final` with each compiler executable, holding the tests and model fixed:

```sh
cd /tmp/atlas-dma-wait-validation/candidate-final
/bwrcq/C/reednicolas/ee194-sp26-chipyard/generators/sp26-atlas-acc/npu-model/.venv/bin/python -m pytest tests \
  --atlas-opt=/tmp/atlas-dma-wait-validation/baseline/build/atlas-opt \
  --artifacts-dir=/tmp/atlas-dma-wait-validation/artifacts/current-suite-baseline-compiler
/bwrcq/C/reednicolas/ee194-sp26-chipyard/generators/sp26-atlas-acc/npu-model/.venv/bin/python -m pytest tests \
  --atlas-opt=/tmp/atlas-dma-wait-validation/candidate-final/build/atlas-opt \
  --artifacts-dir=/tmp/atlas-dma-wait-validation/artifacts/current-suite-candidate
ctest --test-dir build --output-on-failure
```

The baseline command intentionally fails the 44 new repair cases; both full-suite commands include the known RMSNorm failure. The removed-wait experiment uses the same Python and `tests/test_equivalence.py`, with `--atlas-opt="python3 /tmp/atlas-dma-wait-validation/strip_dma_waits.py /path/to/atlas-opt"`. To build the working repository in a fresh directory:

```sh
cmake -S . -B /tmp/atlas-dma-waits-build -G 'Unix Makefiles'
cmake --build /tmp/atlas-dma-waits-build -j 4
ctest --test-dir /tmp/atlas-dma-waits-build --output-on-failure
```

Kernel-model tests require the pinned `third_party/npu_model` package; these runs use the isolated snapshots rather than changing the current model checkout.

## Scope

The compiler still assumes no earlier work is active at program entry. It preserves DMA instruction order, channel numbers, and memory layout. Unknown addresses can cause conservative waits, and inferred waits at a join may execute on a path whose channel is already idle. This work does not add loop unrolling, cross-block engine scheduling, or automatic recognition of completion CSRs.

The model's queued `dma.config.chN` and completion-time operand reads remain part of the compiler contract; the RTL handles configuration locally. Automatic waits do not provide host acknowledgment, buffer ownership, or proof that Atlas has left an IMEM slot. Model falloff remains distinct from an RTL instruction-memory boundary. Validation uses the NPU model and compiler timing checks, not RTL simulation or hardware measurements.
