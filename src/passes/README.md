# Writing an optimization pass

A pass is one function that rewrites the program's basic blocks in place:

```cpp
void myPass(Code& code, PassContext& ctx);
```

Passes run in the order listed in `registry.cpp`. `schedule` always runs last: it
reorders each block, picks every instruction's issue cycle, and chooses the
`delay`s. **So a pass never has to think about timing.** It only has to keep the
program's meaning when the instructions run one at a time, in order.

## What you work with

`src/core` has everything a pass needs:

| Header | Use it for |
|---|---|
| `core/asm.h` | `Instr` (opcode, `rd/rs1/rs2`, `imm`, `target`), `makeInstr`, `formatInstr` |
| `core/blocks.h` | `Code` → `blocks` → `Block` (`labels`, `body`, `terminator`, delay `slot`, `succs`); `blockInstructions(block)` |
| `core/values.h` | `blockEntryValues(code)`: scalar register values known at the start of each block |
| `core/machine.h` | `footprintOf(instr, regs)`: every register, tensor row, accumulator, and VMEM line an instruction reads or writes, and when; `dependence(a, fa, b, fb)`: how many cycles `b` must wait for `a`, and why |
| `core/depgraph.h` | `buildGraph(instrs, entryValues, dmaOperandRegisters(...))`: the dependency graph of a block. `g.out[i]` lists the edges leaving instruction `i`; each edge has a `distance`, a `kind` (RAW, WAR, WAW, rule, order) and a `reason` |
| `core/simulator.h` | `simulate(program)`: cycle count (same as npu_model's) and any broken timing rules |

## Rules every pass follows

1. **Keep the meaning** of each block when its instructions run one at a time, in
   order. If you move an instruction, check `buildGraph`: an instruction may only
   move later than every instruction it depends on, and earlier than everything
   that depends on it.
2. **Keep the block structure** (labels, terminators, successors) unless the pass
   is about control flow. If it is, update `succs` too.
3. **The delay slot runs on both paths**, after the branch: when the branch is
   taken and when it isn't.
4. **Only DRAM is live when the program ends.** Registers, VMEM, weight slots, and
   accumulators may be changed or dropped if no later `dma.store` needs them.
5. **Don't rely on anything in [OPEN_QUESTIONS.md](../../OPEN_QUESTIONS.md).** Use
   the conservative choice listed there.
6. **Report what you did** with one line in `ctx.log`.

## Adding a pass

1. Create `src/passes/my_pass.cpp`. CMake picks up new files automatically.

   ```cpp
   // my-pass: one line saying what it improves.
   #include "core/depgraph.h"
   #include "passes/pass.h"

   void myPass(Code& code, PassContext& ctx) {
       std::vector<RegValues> entry = blockEntryValues(code);
       int changed = 0;
       for (size_t bi = 0; bi < code.blocks.size(); bi++) {
           Block& b = code.blocks[bi];
           DepGraph g = buildGraph(blockInstructions(b), entry[bi]);
           // ... inspect g, edit b.body ...
       }
       ctx.log.push_back("my-pass: changed " + std::to_string(changed) + " instructions");
   }
   ```

2. Declare it at the bottom of `pass.h`: `void myPass(Code& code, PassContext& ctx);`
3. Add it to the list in `registry.cpp`, somewhere before `schedule`:
   `{"my-pass", "what it does", myPass},`
4. Add a test to `tests/tests.cpp`. Build a small program with `parseAsm`, run
   `optimize(program, {"strip-artifacts", "my-pass", "schedule"})`, and check the
   result: its instructions, and `simulate(result)` (fewer cycles, no violations).
5. Check it on every kernel:

   ```sh
   cmake --build build && build/atlas-tests                 # unit tests + all kernels in our simulator
   ~/Projects/npu_model/.venv/bin/python -m pytest          # equivalence on npu_model (rtl-match)
   build/atlas-opt kernel.S --viz kernel.html               # see what the pass changed
   ```

   While developing, `build/atlas-opt kernel.S --passes strip-artifacts,my-pass,schedule`
   runs only the passes you name.
