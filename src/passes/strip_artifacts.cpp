// strip-artifacts: removes the old schedule so `schedule` can build a new one.
#include "core/machine.h"
#include "passes/pass.h"

static bool isArtifact(const Instr& in) {
    return (in.op->opClass == OpClass::Delay && !in.keep) || isNop(in);
}

// Preserve scalar RAW/WAR/WAW order, including jump link registers.
static bool scalarRegistersConflict(const Instr& a, const Instr& b) {
    Footprint fa = footprintOf(a, unknownRegs()), fb = footprintOf(b, unknownRegs());
    for (const Access& x : fa.accesses)
        for (const Access& y : fb.accesses)
            if (x.res == Res::XReg && y.res == Res::XReg && x.first == y.first && (x.write || y.write)) return true;
    return false;
}

void stripArtifacts(Code& code, PassContext& ctx) {
    int removed = 0;
    for (Block& b : code.blocks) {
        std::vector<Instr> kept;
        for (const Instr& in : b.body) {
            if (isArtifact(in)) removed++;
            else kept.push_back(in);
        }
        b.body = kept;
        if (!b.slot) continue;
        if (isArtifact(*b.slot)) {
            b.slot.reset();
            removed++;
        } else if (b.slot->op->opClass != OpClass::Delay && b.slot->op->opClass != OpClass::Halt && !scalarRegistersConflict(*b.slot, *b.terminator)) {
            // The slot runs on both paths, like the block body, so it can run before the branch.
            b.body.push_back(*b.slot);
            b.slot.reset();
        }
    }
    ctx.log.push_back("strip-artifacts: removed " + std::to_string(removed) + " delays and no-op fillers");
}
