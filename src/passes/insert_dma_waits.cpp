#include <algorithm>
#include <array>
#include <deque>
#include <set>
#include <stdexcept>

#include "core/depgraph.h"
#include "passes/pass.h"

namespace {

using Pending = std::array<std::set<int>, 8>;

struct Step {
    Instr in;
    Footprint footprint;
    size_t boundary;
    int command = -1;
};

struct Plan {
    std::vector<Step> steps;
    std::vector<unsigned> waits;
};

unsigned activeChannels(const Pending& pending) {
    unsigned mask = 0;
    for (int channel = 0; channel < 8; channel++)
        if (!pending[channel].empty()) mask |= 1u << channel;
    return mask;
}

void clearChannels(Pending& pending, unsigned mask) {
    for (int channel = 0; channel < 8; channel++)
        if (mask & (1u << channel)) pending[channel].clear();
}

void advance(Pending& pending, const Step& step) {
    const OpInfo& op = *step.in.op;
    if (op.engine != Engine::Dma) return;
    if (op.opClass == OpClass::DmaWait) pending[op.channel].clear();
    else pending[op.channel].insert(step.command);
}

bool joinInto(Pending& into, const Pending& from) {
    bool changed = false;
    for (int channel = 0; channel < 8; channel++) {
        size_t before = into[channel].size();
        into[channel].insert(from[channel].begin(), from[channel].end());
        changed |= into[channel].size() != before;
    }
    return changed;
}

struct Flow {
    std::vector<Pending> entry;
    std::vector<bool> reached;
};

Flow pendingAtEntry(const Code& code, const std::vector<Plan>& plans) {
    Flow flow{std::vector<Pending>(code.blocks.size()), std::vector<bool>(code.blocks.size(), false)};
    std::deque<int> work{0};
    std::vector<bool> queued(code.blocks.size(), false);
    flow.reached[0] = queued[0] = true;
    while (!work.empty()) {
        int index = work.front();
        work.pop_front();
        queued[index] = false;
        Pending pending = flow.entry[index];
        for (const Step& step : plans[index].steps) {
            clearChannels(pending, plans[index].waits[step.boundary]);
            advance(pending, step);
        }
        clearChannels(pending, plans[index].waits.back());
        for (int successor : code.blocks[index].succs) {
            bool changed = joinInto(flow.entry[successor], pending);
            if (!flow.reached[successor] || changed) {
                flow.reached[successor] = true;
                if (!queued[successor]) work.push_back(successor), queued[successor] = true;
            }
        }
    }
    return flow;
}

unsigned requiredWaits(const Pending& pending, const Step& step, const std::vector<Footprint>& commands) {
    const OpInfo& op = *step.in.op;
    if (step.in.release || op.opClass == OpClass::Halt) return activeChannels(pending);
    unsigned waits = 0;
    for (int channel = 0; channel < 8; channel++) {
        if (pending[channel].empty()) continue;
        if (op.engine == Engine::Dma && op.opClass != OpClass::DmaWait && op.channel == channel) {
            waits |= 1u << channel;
            continue;
        }
        for (int command : pending[channel]) {
            EdgeKind kind;
            if (conflictsAtCompletion(commands[command], step.footprint, kind)) {
                waits |= 1u << channel;
                break;
            }
        }
    }
    return waits;
}

void validateFlow(const Code& code) {
    std::vector<bool> reached(code.blocks.size(), false);
    std::deque<int> work;
    if (!code.blocks.empty()) work.push_back(0), reached[0] = true;
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
    for (const Block& block : code.blocks) {
        if (block.slot && block.slot->op->engine == Engine::Dma && block.slot->op->opClass != OpClass::DmaWait)
            throw std::runtime_error("line " + std::to_string(block.slot->line) +
                                     ": insert-dma-waits requires strip-artifacts to move DMA commands out of delay slots");
    }
}

// Give end-label targets and the last branch's fallthrough an actual CFG exit.
void addExit(Code& code) {
    int exit = (int)code.blocks.size();
    for (int index = 0; index < exit; index++) {
        Block& block = code.blocks[index];
        if (block.terminator && isControlFlow(*block.terminator->op)) {
            const std::string& target = block.terminator->target;
            if (std::find(code.endLabels.begin(), code.endLabels.end(), target) != code.endLabels.end())
                block.succs.push_back(exit);
        }
        if (index + 1 == exit && (!block.terminator || block.terminator->op->opClass == OpClass::Branch))
            block.succs.push_back(exit);
    }
    Block end;
    end.labels = code.endLabels;
    code.endLabels.clear();
    code.blocks.push_back(std::move(end));
}

}  // namespace

void insertDmaWaits(Code& code, PassContext& ctx) {
    validateFlow(code);
    if (code.blocks.empty()) {
        ctx.log.push_back("insert-dma-waits: inserted 0 waits");
        return;
    }
    Code candidate = code;
    addExit(candidate);
    size_t exit = candidate.blocks.size() - 1;
    std::vector<RegValues> entries = blockEntryValues(candidate);
    std::vector<Plan> plans(candidate.blocks.size());
    std::vector<Footprint> commands;
    for (size_t index = 0; index < candidate.blocks.size(); index++) {
        const Block& block = candidate.blocks[index];
        Plan& plan = plans[index];
        plan.waits.resize(block.body.size() + 1, 0);
        RegValues regs = entries[index];
        auto addStep = [&](const Instr& in, size_t boundary) {
            Step step{in, footprintOf(in, regs), boundary};
            if (in.op->engine == Engine::Dma && in.op->opClass != OpClass::DmaWait) {
                step.command = (int)commands.size();
                commands.push_back(step.footprint);
            }
            plan.steps.push_back(std::move(step));
            applyScalar(in, regs);
        };
        for (size_t i = 0; i < block.body.size(); i++) addStep(block.body[i], i);
        if (block.terminator) addStep(*block.terminator, block.body.size());
        if (block.slot) addStep(*block.slot, block.body.size());
    }

    // Each round adds waits, then recomputes reachability with those channel kills.
    // Command-site sets retain all possible transfers at joins and loop backedges.
    Flow flow;
    bool changed;
    do {
        flow = pendingAtEntry(candidate, plans);
        changed = false;
        for (size_t index = 0; index < plans.size(); index++) {
            if (!flow.reached[index]) continue;
            Pending pending = flow.entry[index];
            Plan& plan = plans[index];
            for (const Step& step : plan.steps) {
                unsigned& waits = plan.waits[step.boundary];
                clearChannels(pending, waits);
                unsigned needed = requiredWaits(pending, step, commands);
                changed |= (needed & ~waits) != 0;
                waits |= needed;
                clearChannels(pending, needed);
                advance(pending, step);
            }
            if (index == exit) {
                unsigned needed = activeChannels(pending);
                changed |= (needed & ~plan.waits.back()) != 0;
                plan.waits.back() |= needed;
            }
        }
    } while (changed);

    int inserted = 0;
    for (size_t index = 0; index < candidate.blocks.size(); index++) {
        if (!flow.reached[index]) continue;
        Block& block = candidate.blocks[index];
        Pending pending = flow.entry[index];
        std::vector<Instr> body;
        size_t current = 0;
        auto emitWaits = [&](size_t boundary) {
            unsigned needed = plans[index].waits[boundary] & activeChannels(pending);
            for (int channel = 0; channel < 8; channel++) {
                if (!(needed & (1u << channel))) continue;
                body.push_back(makeInstr("dma.wait.ch" + std::to_string(channel)));
                inserted++;
            }
            clearChannels(pending, needed);
        };
        for (const Step& step : plans[index].steps) {
            if (current <= step.boundary) {
                emitWaits(step.boundary);
                current = step.boundary + 1;
            }
            if (step.boundary < block.body.size()) body.push_back(step.in);
            advance(pending, step);
        }
        if (current <= block.body.size()) emitWaits(block.body.size());
        block.body = std::move(body);
        block.scheduled = false;
        block.issue.clear();
    }

    Block& final = candidate.blocks.back();
    Block& previous = candidate.blocks[exit - 1];
    bool merge = final.labels.empty() && !previous.terminator;
    if (final.body.empty() || merge) {
        if (merge) previous.body.insert(previous.body.end(), final.body.begin(), final.body.end());
        candidate.endLabels = final.labels;
        candidate.blocks.pop_back();
        for (Block& block : candidate.blocks)
            std::erase(block.succs, (int)exit);
    }
    if (inserted != 0) code = std::move(candidate);
    ctx.log.push_back("insert-dma-waits: inserted " + std::to_string(inserted) + " waits");
}
