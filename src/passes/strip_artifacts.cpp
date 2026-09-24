// strip-artifacts: removes the old schedule so `schedule` can build a new one.
#include "core/machine.h"
#include "passes/pass.h"

static bool isArtifact(const Instr& in) {
    return (in.op->opClass == OpClass::Delay && !in.keep) || isNop(in);
}

// True if `a` writes a scalar register that `b` reads.
static bool writesRegisterReadBy(const Instr& a, const Instr& b) {
    Footprint fa = footprintOf(a, unknownRegs()), fb = footprintOf(b, unknownRegs());
    for (const Access& w : fa.accesses)
        for (const Access& r : fb.accesses)
            if (w.write && !r.write && w.res == Res::XReg && r.res == Res::XReg && w.first == r.first) return true;
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
        } else if (!writesRegisterReadBy(*b.slot, *b.terminator)) {
            // The slot runs on both paths, like the block body, so it can run before the branch.
            b.body.push_back(*b.slot);
            b.slot.reset();
        }
    }
    ctx.log.push_back("strip-artifacts: removed " + std::to_string(removed) + " delays and no-op fillers");
}
