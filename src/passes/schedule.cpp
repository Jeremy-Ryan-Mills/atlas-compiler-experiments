// schedule: list-schedules every block on a cycle-by-cycle reservation table and
// records the issue cycles; flatten() later turns idle cycles into `delay`s.
#include <algorithm>
#include <climits>
#include <numeric>
#include <stdexcept>

#include "core/depgraph.h"
#include "core/reservations.h"
#include "passes/pass.h"

static std::runtime_error scheduleError(const Instr& in, const std::string& why) {
    return std::runtime_error("line " + std::to_string(in.line) + " (" + formatInstr(in) + "): " + why);
}

// Reorders and times one block. The block may assume an idle machine on entry and
// drains (everything it started finishes) before its successors begin.
static void scheduleBlock(Block& block, const RegValues& entry, bool robustDma, bool lastBlock, bool fallthroughHalt, uint32_t dmaRegs) {
    std::vector<Instr> nodes = blockInstructions(block);
    int nb = (int)block.body.size(), n = (int)nodes.size();
    int term = block.terminator ? nb : -1;
    int slot = hasDelaySlot(block) ? nb + 1 : -1;

    DepGraph g = buildGraph(nodes, entry, dmaRegs);
    for (int i = 0; i < n; i++) {
        if (!g.footprints[i].error.empty()) throw scheduleError(nodes[i], g.footprints[i].error);
        std::string alone = ReservationTable().conflict(nodes[i], g.footprints[i], 0);
        if (!alone.empty()) throw scheduleError(nodes[i], "can never issue: " + alone);
    }
    if (slot >= 0 && g.footprints[slot].doneAge > 0)
        throw scheduleError(nodes[slot], "only single-cycle scalar instructions are supported in a delay slot");

    std::vector<int> height = criticalHeights(g);
    std::vector<int> earliest(n, 0), waitingPreds(n, 0), issue(n, -1);
    for (const Edge& e : g.edges) waitingPreds[e.to]++;

    // Expected DMA timing (npu_model's estimate), used only to decide when to issue a
    // dma.wait. Correctness never depends on it: after a wait the schedule assumes nothing.
    int channelRelease[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int dmaQueueEnd = 0;
    auto release = [&](int i) { return channelRelease[nodes[i].op->channel]; };

    ReservationTable table;
    int cycle = 0, placed = 0, nextFree = 0, lastPlaced = 0, lastBody = -1;
    while (placed < nb) {
        // Among ready instructions that fit this cycle, take the one on the longest path.
        int best = -1, bestWait = -1;
        bool otherWork = false;
        for (int i = 0; i < nb; i++) {
            if (issue[i] >= 0 || waitingPreds[i] > 0) continue;
            bool isWait = nodes[i].op->opClass == OpClass::DmaWait;
            if (!isWait) otherWork = true;
            if (earliest[i] > cycle) continue;
            if (isWait) {
                if (bestWait < 0 || release(i) < release(bestWait)) bestWait = i;
                continue;
            }
            if (best >= 0 && height[i] <= height[best]) continue;  // ties keep program order
            if (!table.conflict(nodes[i], g.footprints[i], cycle).empty()) continue;
            best = i;
        }
        // A dma.wait stalls the frontend until its transfer is done, so other work goes first.
        if (best < 0 && bestWait >= 0 && (!otherWork || cycle >= release(bestWait))) {
            // Idle cycles before a wait overlap the transfer; idle cycles after it do not.
            // So issue the wait just before its most critical waiting instruction can go.
            int firstUse = INT_MAX, critical = -1;
            for (int e : g.out[bestWait]) {
                int s = g.edges[e].to;
                if (s >= nb || waitingPreds[s] != 1) continue;
                int c = std::max(cycle + 1, earliest[s]);
                while (!table.conflict(nodes[s], g.footprints[s], c).empty()) c++;
                if (critical < 0 || height[s] > height[critical] || (height[s] == height[critical] && c < firstUse))
                    critical = s, firstUse = c;
            }
            if (firstUse == INT_MAX || firstUse - 1 <= cycle) best = bestWait;
        }
        if (best < 0) {
            cycle++;
            if (cycle - lastPlaced > 100000) throw std::runtime_error("scheduler made no progress (internal error)");
            continue;
        }
        issue[best] = cycle;
        table.reserve(nodes[best], g.footprints[best], cycle);
        if (robustDma && nodes[best].op->opClass == OpClass::DmaWait) table.extendForWait(cycle);
        if (g.footprints[best].dmaCycles > 0) {  // transfers run one at a time, in issue order
            int latency = g.footprints[best].dmaCycles;
            dmaQueueEnd = std::max(cycle + latency - 1, dmaQueueEnd + latency);
            channelRelease[nodes[best].op->channel] = dmaQueueEnd + 2;
        }
        for (int e : g.out[best]) {
            const Edge& ed = g.edges[e];
            earliest[ed.to] = std::max(earliest[ed.to], cycle + ed.distance);
            waitingPreds[ed.to]--;
        }
        nextFree = cycle + naturalGap(nodes[best]);
        lastPlaced = cycle;
        lastBody = best;
        cycle = nextFree;
        placed++;
    }

    // Everything started in this block must finish before the next block starts.
    int drain = 0;
    for (int i = 0; i < nb; i++) drain = std::max(drain, issue[i] + g.footprints[i].doneAge + 1);
    // Kept delays need an extra guard cycle before halt, even across blocks.
    bool keptDelay = lastBody >= 0 && nodes[lastBody].op->opClass == OpClass::Delay && nodes[lastBody].keep;

    if (term >= 0) {
        bool halt = nodes[term].op->opClass == OpClass::Halt;
        // With a delay slot, the next block starts two cycles after the branch.
        int t = std::max({nextFree, earliest[term], halt ? drain : drain - 2});
        if (halt && keptDelay) t = std::max(t, nextFree + 1);
        if (slot >= 0) t = std::max(t, earliest[slot] - 1);
        while (!table.conflict(nodes[term], g.footprints[term], t).empty() ||
               (slot >= 0 && !table.conflict(nodes[slot], g.footprints[slot], t + 1).empty()))
            t++;
        block.terminatorCycle = t;
        block.endCycle = slot >= 0 ? t + 2 : t + 1;
    } else {
        block.endCycle = lastBlock ? nextFree : std::max(nextFree, drain);
        if (fallthroughHalt && keptDelay) block.endCycle = std::max(block.endCycle, nextFree + 1);
    }

    std::vector<int> order(nb);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return issue[a] < issue[b]; });
    block.body.clear();
    block.issue.clear();
    for (int i : order) {
        block.body.push_back(nodes[i]);
        block.issue.push_back(issue[i]);
    }
    block.scheduled = true;
}

void schedule(Code& code, PassContext& ctx) {
    std::vector<RegValues> entry = blockEntryValues(code);
    uint32_t dmaRegs = dmaOperandRegisters(flatten(code).instrs);
    for (size_t bi = 0; bi < code.blocks.size(); bi++) {
        try {
            bool lastBlock = bi + 1 == code.blocks.size();
            size_t next = bi + 1;
            // Stripping can leave empty labeled blocks.
            while (next < code.blocks.size() && code.blocks[next].body.empty() && !code.blocks[next].terminator) next++;
            bool fallthroughHalt = next < code.blocks.size() && code.blocks[next].body.empty() &&
                                   code.blocks[next].terminator && code.blocks[next].terminator->op->opClass == OpClass::Halt;
            scheduleBlock(code.blocks[bi], entry[bi], ctx.robustDma, lastBlock, fallthroughHalt, dmaRegs);
        } catch (const std::runtime_error& e) {
            throw std::runtime_error("block " + std::to_string(bi) + ": " + e.what());
        }
    }
    ctx.log.push_back(std::string("schedule: list-scheduled ") + std::to_string(code.blocks.size()) + " blocks (" +
                      (ctx.robustDma ? "robust" : "npu_model") + " DMA timing)");
}
