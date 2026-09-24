// fill-delay-slots: gives each empty branch delay slot useful work.
#include "core/depgraph.h"
#include "passes/pass.h"

void fillDelaySlots(Code& code, PassContext& ctx) {
    std::vector<RegValues> entry = blockEntryValues(code);
    uint32_t dmaRegs = dmaOperandRegisters(flatten(code).instrs);
    int filled = 0;
    for (size_t bi = 0; bi < code.blocks.size(); bi++) {
        Block& b = code.blocks[bi];
        if (!hasDelaySlot(b) || b.slot) continue;
        std::vector<Instr> nodes = b.body;
        nodes.push_back(*b.terminator);
        DepGraph g = buildGraph(nodes, entry[bi], dmaRegs);
        // The slot runs after the branch on both paths, so take the latest plain scalar
        // instruction that nothing after it depends on (not even the branch).
        for (int i = (int)b.body.size() - 1; i >= 0; i--) {
            const Instr& in = b.body[i];
            if (in.op->opClass != OpClass::Alu || in.rd == 0 || !g.out[i].empty()) continue;
            b.slot = in;
            b.body.erase(b.body.begin() + i);
            filled++;
            break;
        }
    }
    ctx.log.push_back("fill-delay-slots: filled " + std::to_string(filled) + " branch delay slots");
}
