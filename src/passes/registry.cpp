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
    for (const Block& b : code.blocks)
        for (const Instr& in : b.body)
            if (in.op->name == "auipc")  // its result is its own address, which every pass changes
                throw std::runtime_error("line " + std::to_string(in.line) + ": auipc is not supported by the optimizer");
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
