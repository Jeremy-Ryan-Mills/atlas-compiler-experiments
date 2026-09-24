// unroll-loops: fully unrolls loops whose trip count is known, so `schedule` sees
// one straight-line block and can overlap one iteration's transfers with another's
// compute (the blocks no longer drain at every back-edge).
#include "core/values.h"
#include "passes/pass.h"

static const int kMaxTrips = 64;
static const int kMaxUnrolledInstrs = 2048;  // keeps IMEM use and scheduling time small

// True if a block other than `self` branches or jumps to one of `self`'s labels.
static bool enteredFromElsewhere(const Code& code, size_t self) {
    for (size_t i = 0; i < code.blocks.size(); i++) {
        const Block& b = code.blocks[i];
        if (i == self || !b.terminator || b.terminator->target.empty()) continue;
        for (const std::string& l : code.blocks[self].labels)
            if (b.terminator->target == l) return true;
    }
    return false;
}

// How many times loop block `b` runs when entered with `regs` (0 if unknown or too many).
static int tripCount(const Block& b, RegValues regs) {
    for (int trips = 1; trips <= kMaxTrips; trips++) {
        for (const Instr& in : b.body) applyScalar(in, regs);
        std::optional<bool> taken = branchTaken(*b.terminator, regs);  // reads registers before the slot runs
        if (!taken) return 0;
        if (b.slot) applyScalar(*b.slot, regs);
        if (!*taken) return trips;
    }
    return 0;
}

// Unrolls the first loop it can. A loop is a block whose branch jumps back to its own
// label, entered only by falling into it from the previous block (or at program start).
static bool unrollOne(Code& code, int& trips) {
    std::vector<RegValues> entry = blockEntryValues(code);
    for (size_t bi = 0; bi < code.blocks.size(); bi++) {
        Block& b = code.blocks[bi];
        if (!b.terminator || b.terminator->op->opClass != OpClass::Branch) continue;
        bool selfLoop = false;
        for (const std::string& l : b.labels) selfLoop |= b.terminator->target == l;
        if (!selfLoop || enteredFromElsewhere(code, bi)) continue;

        RegValues regs = zeroRegs();  // program start
        if (bi > 0) {
            const Block& prev = code.blocks[bi - 1];
            if (prev.terminator && prev.terminator->op->opClass != OpClass::Branch) continue;  // doesn't fall through
            regs = entry[bi - 1];
            for (const Instr& in : blockInstructions(prev)) applyScalar(in, regs);
        }
        trips = tripCount(b, regs);
        int size = (int)b.body.size() + (b.slot ? 1 : 0);
        if (trips == 0 || trips * size > kMaxUnrolledInstrs) continue;

        // Each iteration runs the body, the branch, then the delay slot (on both paths).
        std::vector<Instr> body;
        for (int t = 0; t < trips; t++) {
            body.insert(body.end(), b.body.begin(), b.body.end());
            if (b.slot) body.push_back(*b.slot);
        }
        b.body = body;
        b.labels.clear();  // only this loop's own branch used them
        b.terminator.reset();
        b.slot.reset();

        // Rebuilding merges the unrolled block with its neighbours into one straight-line
        // block and recomputes successors. Flattening turns empty delay slots into nops,
        // so empty them again.
        code = buildBlocks(flatten(code));
        for (Block& nb : code.blocks)
            if (nb.slot && isNop(*nb.slot)) nb.slot.reset();
        return true;
    }
    return false;
}

void unrollLoops(Code& code, PassContext& ctx) {
    int loops = 0, iterations = 0, trips = 0;
    // Unrolling an inner loop can turn its outer loop into a single block, so repeat.
    while (unrollOne(code, trips)) loops++, iterations += trips;
    ctx.log.push_back("unroll-loops: unrolled " + std::to_string(loops) + " loops (" + std::to_string(iterations) +
                      " iterations in total)");
}
