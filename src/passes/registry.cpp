#include <deque>
#include <stdexcept>

#include "passes/pass.h"

namespace {

// Only a matching wait clears a channel's may-pending bit.
unsigned dmaAfter(const Instr& in, unsigned pending) {
    if (in.op->engine != Engine::Dma) return pending;
    unsigned channel = 1u << in.op->channel;
    return in.op->opClass == OpClass::DmaWait ? pending & ~channel : pending | channel;
}

template <typename Fn>
void visitInstructions(const Block& block, Fn visit) {
    for (const Instr& in : block.body) visit(in);
    if (block.terminator) visit(*block.terminator);
    if (block.slot) visit(*block.slot);
}

void validateReleaseDma(const Code& code, bool checkPending = true) {
    for (const Block& block : code.blocks) {
        visitInstructions(block, [](const Instr& in) {
            if (in.release && in.op->opClass != OpClass::Csr)
                throw std::runtime_error("line " + std::to_string(in.line) + ": atlas.release is only supported on CSR instructions");
        });
        if (block.slot && block.slot->release)
            throw std::runtime_error("line " + std::to_string(block.slot->line) +
                                     ": atlas.release in a delay slot is not supported by the optimizer");
    }
    if (!checkPending || code.blocks.empty()) return;

    // Entry starts idle but must retain pending DMA from backedges.
    std::vector<unsigned> entry(code.blocks.size(), 0);
    std::vector<bool> reached(code.blocks.size(), false), queued(code.blocks.size(), false);
    std::deque<int> work{0};
    reached[0] = queued[0] = true;
    while (!work.empty()) {
        int index = work.front();
        work.pop_front();
        queued[index] = false;
        const Block& block = code.blocks[index];
        if (block.unknownSuccs)
            throw std::runtime_error("atlas.release requires known control-flow successors");
        unsigned pending = entry[index];
        visitInstructions(block, [&](const Instr& in) { pending = dmaAfter(in, pending); });
        for (int successor : block.succs) {
            if (successor < 0 || successor >= (int)code.blocks.size())
                throw std::runtime_error("atlas.release encountered an invalid control-flow successor");
            unsigned joined = entry[successor] | pending;
            if (!reached[successor] || joined != entry[successor]) {
                reached[successor] = true;
                entry[successor] = joined;
                if (!queued[successor]) work.push_back(successor), queued[successor] = true;
            }
        }
    }

    // Wait for convergence to include all joins and backedges.
    for (size_t index = 0; index < code.blocks.size(); index++) {
        if (!reached[index]) continue;
        unsigned pending = entry[index];
        visitInstructions(code.blocks[index], [&](const Instr& in) {
            // A channel flag cannot count overlapping commands; require idle reuse.
            if (in.op->engine == Engine::Dma && in.op->opClass != OpClass::DmaWait &&
                (pending & (1u << in.op->channel))) {
                std::string channel = "ch" + std::to_string(in.op->channel);
                throw std::runtime_error("line " + std::to_string(in.line) +
                                         ": atlas.release cannot prove DMA channel " + channel +
                                         " idle before " + in.op->name + "; add dma.wait." + channel +
                                         " before channel reuse");
            }
            if (in.release && pending != 0) {
                std::string channels;
                for (int channel = 0; channel < 8; channel++)
                    if (pending & (1u << channel)) {
                        if (!channels.empty()) channels += ", ";
                        channels += "ch" + std::to_string(channel);
                    }
                throw std::runtime_error("line " + std::to_string(in.line) +
                                         ": atlas.release may publish with pending DMA on " + channels +
                                         "; add matching dma.wait instructions on every reaching path");
            }
            pending = dmaAfter(in, pending);
        });
    }
}

}  // namespace

const std::vector<Pass>& allPasses() {
    // Add new passes here. `schedule` must stay last: it picks the issue cycles
    // and delays for whatever the earlier passes produced.
    static const std::vector<Pass> passes = {
        {"strip-artifacts", "remove the old schedule: delays and no-op fillers", stripArtifacts},
        {"insert-dma-waits", "insert DMA waits before dependent accesses, channel reuse, and completion", insertDmaWaits},
        {"fill-delay-slots", "move an independent scalar instruction into each empty branch delay slot", fillDelaySlots},
        {"schedule", "list-schedule every block and choose the delays", schedule},
    };
    return passes;
}

void runPasses(Code& code, const std::vector<std::string>& names, PassContext& ctx) {
    bool stripsArtifacts = names.empty(), insertsDmaWaits = names.empty(), schedules = names.empty(), hasRelease = false;
    int firstReleaseLine = 0;
    for (const std::string& name : names) {
        stripsArtifacts |= name == "strip-artifacts";
        insertsDmaWaits |= name == "insert-dma-waits";
        schedules |= name == "schedule";
    }
    for (const Block& block : code.blocks)
        visitInstructions(block, [&](const Instr& in) {
            if (in.release && !hasRelease) firstReleaseLine = in.line;
            hasRelease |= in.release;
        });
    // Reject unrelocatable addresses before any pass mutates the program.
    auto validate = [](const Instr& in) {
        std::string reason;
        if (in.op->name == "auipc")
            reason = "auipc is not supported by the optimizer (PC-relative values cannot be relocated)";
        else if (in.op->name == "jalr")
            reason = "jalr is not supported by the optimizer (indirect targets cannot be relocated)";
        else if (in.op->name == "jal" && in.rd != 0)
            reason = "jal with a nonzero link register is not supported by the optimizer (link values cannot be relocated)";
        if (!reason.empty()) throw std::runtime_error("line " + std::to_string(in.line) + ": " + reason);
    };
    for (const Block& b : code.blocks) {
        for (const Instr& in : b.body) validate(in);
        if (b.terminator) validate(*b.terminator);
        if (b.slot) {
            validate(*b.slot);
            OpClass slotClass = b.slot->op->opClass;
            // Only stripped delays are safe in branch slots.
            bool retainedDelay = slotClass == OpClass::Delay && (b.slot->keep || !stripsArtifacts);
            if (retainedDelay || slotClass == OpClass::Halt)
                throw std::runtime_error("line " + std::to_string(b.slot->line) + ": " + b.slot->op->name +
                                         " in a delay slot is not supported by the optimizer");
            if (insertsDmaWaits && !stripsArtifacts && b.slot->op->engine == Engine::Dma && slotClass != OpClass::DmaWait)
                throw std::runtime_error("line " + std::to_string(b.slot->line) +
                                         ": insert-dma-waits requires strip-artifacts to move DMA commands out of delay slots");
        }
    }
    if (insertsDmaWaits && !code.blocks.empty()) {
        std::deque<int> work{0};
        std::vector<bool> reached(code.blocks.size(), false);
        reached[0] = true;
        while (!work.empty()) {
            const Block& block = code.blocks[work.front()];
            work.pop_front();
            if (block.unknownSuccs)
                throw std::runtime_error("insert-dma-waits requires known control-flow successors");
            for (int successor : block.succs) {
                if (successor < 0 || successor >= (int)code.blocks.size())
                    throw std::runtime_error("insert-dma-waits encountered an invalid control-flow successor");
                if (!reached[successor]) work.push_back(successor), reached[successor] = true;
            }
        }
    }
    for (const std::string& name : names) {
        bool known = false;
        for (const Pass& p : allPasses()) known |= name == p.name;
        if (!known) throw std::runtime_error("unknown pass '" + name + "'");
    }
    if (insertsDmaWaits && !schedules)
        throw std::runtime_error("insert-dma-waits requires the schedule pass");
    if (hasRelease) {
        if (!schedules)
            throw std::runtime_error("line " + std::to_string(firstReleaseLine) +
                                     ": atlas.release requires the schedule pass");
        validateReleaseDma(code, !insertsDmaWaits);
    }
    bool waitsReady = !insertsDmaWaits;
    for (const Pass& p : allPasses()) {
        bool selected = names.empty();
        for (const std::string& name : names) selected |= name == p.name;
        if (selected) {
            p.run(code, ctx);
            waitsReady |= std::string(p.name) == "insert-dma-waits";
            if (hasRelease) validateReleaseDma(code, waitsReady);
        }
    }
}
