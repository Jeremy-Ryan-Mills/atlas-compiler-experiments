#include <stdexcept>

#include "passes/pass.h"

const std::vector<Pass>& allPasses() {
    // Add new passes here. `schedule` must stay last: it picks the issue cycles
    // and delays for whatever the earlier passes produced.
    static const std::vector<Pass> passes = {
        {"strip-artifacts", "remove the old schedule: delays and no-op fillers", stripArtifacts},
        {"fill-delay-slots", "move an independent scalar instruction into each empty branch delay slot", fillDelaySlots},
        {"schedule", "list-schedule every block and choose the delays", schedule},
    };
    return passes;
}

void runPasses(Code& code, const std::vector<std::string>& names, PassContext& ctx) {
    bool stripsArtifacts = names.empty();
    for (const std::string& name : names) stripsArtifacts |= name == "strip-artifacts";
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
        }
    }
    for (const std::string& name : names) {
        bool known = false;
        for (const Pass& p : allPasses()) known |= name == p.name;
        if (!known) throw std::runtime_error("unknown pass '" + name + "'");
    }
    for (const Pass& p : allPasses()) {
        bool selected = names.empty();
        for (const std::string& name : names) selected |= name == p.name;
        if (selected) p.run(code, ctx);
    }
}
