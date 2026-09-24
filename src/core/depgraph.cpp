#include "core/depgraph.h"

#include <algorithm>
#include <map>

namespace {

// Adds an edge, or raises the distance of an existing edge between the same nodes.
struct EdgeSet {
    DepGraph& g;
    std::map<std::pair<int, int>, int> index;

    void add(int from, int to, int distance, EdgeKind kind, const std::string& reason) {
        auto it = index.find({from, to});
        if (it == index.end()) {
            index[{from, to}] = (int)g.edges.size();
            g.edges.push_back({from, to, distance, kind, reason});
        } else if (distance > g.edges[it->second].distance) {
            g.edges[it->second] = {from, to, distance, kind, reason};
        }
    }
};

bool overlaps(const Access& x, const Access& y) {
    if (x.res != y.res) return false;
    if (x.anywhere || y.anywhere) return true;
    return x.first < y.first + y.count && y.first < x.first + x.count;
}

// Does instruction `f` conflict with what DMA instruction `dma` does at completion?
bool conflictsAtCompletion(const Footprint& dma, const Footprint& f, EdgeKind& kind) {
    for (const Access& x : dma.accesses) {
        if (!x.atCompletion) continue;
        for (const Access& y : f.accesses) {
            if (!overlaps(x, y) || (!x.write && !y.write)) continue;
            if (x.res == Res::DmaBase && y.atCompletion) continue;  // the DMA queue keeps these in order
            kind = x.write && y.write ? EdgeKind::WAW : x.write ? EdgeKind::RAW : EdgeKind::WAR;
            return true;
        }
    }
    return false;
}

}  // namespace

uint32_t dmaOperandRegisters(const std::vector<Instr>& instrs) {
    uint32_t mask = 0;
    for (const Instr& in : instrs) {
        if (in.op->engine != Engine::Dma || in.op->opClass == OpClass::DmaWait) continue;
        if (in.op->opClass != OpClass::DmaConfig) mask |= 1u << in.rd | 1u << in.rs2;
        mask |= 1u << in.rs1;
    }
    return mask & ~1u;
}

DepGraph buildGraph(const std::vector<Instr>& instrs, const RegValues& entry, uint32_t dmaRegs) {
    DepGraph g;
    g.nodes = instrs;
    int n = (int)instrs.size();
    RegValues regs = entry;
    for (const Instr& in : instrs) {
        g.footprints.push_back(footprintOf(in, regs));
        applyScalar(in, regs);
    }

    EdgeSet edges{g, {}};
    for (int b = 0; b < n; b++)
        for (int a = 0; a < b; a++) {
            Dependence d = dependence(g.nodes[a], g.footprints[a], g.nodes[b], g.footprints[b]);
            if (d.distance > 0) edges.add(a, b, d.distance, d.kind, d.reason);
        }

    // A DMA reads its registers and moves data when it completes, which is only known
    // to have happened once the matching dma.wait issues. Later conflicting accesses
    // therefore wait for that dma.wait.
    for (int d = 0; d < n; d++) {
        const OpInfo& op = *g.nodes[d].op;
        if (op.engine != Engine::Dma || op.opClass == OpClass::DmaWait) continue;
        int wait = -1;
        for (int k = d + 1; k < n && wait < 0; k++)
            if (g.nodes[k].op->opClass == OpClass::DmaWait && g.nodes[k].op->channel == op.channel) wait = k;
        for (int k = d + 1; k < n; k++) {
            EdgeKind kind;
            if (k == wait || !conflictsAtCompletion(g.footprints[d], g.footprints[k], kind)) continue;
            if (wait >= 0 && wait < k)
                edges.add(wait, k, 1, kind, std::string(edgeKindName(kind)) + " with " + op.name + " (done once the wait issues)");
            else
                edges.add(d, k, 1, EdgeKind::Order, op.name + " may still be in flight (no dma.wait in between)");
        }
    }

    // A dma.wait for a transfer started in an earlier block guards data this block can't
    // see, so later VMEM accesses, DMA commands and writes to DMA operand registers stay behind it.
    for (int w = 0; w < n; w++) {
        const OpInfo& op = *g.nodes[w].op;
        if (op.opClass != OpClass::DmaWait) continue;
        bool local = false;
        for (int d = 0; d < w; d++)
            if (g.nodes[d].op->engine == Engine::Dma && g.nodes[d].op->channel == op.channel) local = true;
        if (local) continue;
        for (int k = w + 1; k < n; k++) {
            bool guarded = g.nodes[k].op->engine == Engine::Dma && g.nodes[k].op->opClass != OpClass::DmaWait;
            for (const Access& a : g.footprints[k].accesses) {
                if (a.res == Res::Vmem) guarded = true;
                if (a.res == Res::XReg && a.write && (dmaRegs >> a.first & 1)) guarded = true;
            }
            if (guarded) edges.add(w, k, 1, EdgeKind::Order, op.name + " guards a transfer started in an earlier block");
        }
    }

    g.in.assign(n, {});
    g.out.assign(n, {});
    for (int e = 0; e < (int)g.edges.size(); e++) {
        g.out[g.edges[e].from].push_back(e);
        g.in[g.edges[e].to].push_back(e);
    }
    return g;
}

std::vector<int> criticalHeights(const DepGraph& g) {
    int n = (int)g.nodes.size();
    std::vector<int> height(n, 0);
    for (int i = n - 1; i >= 0; i--) {
        height[i] = g.footprints[i].doneAge + 1;
        for (int e : g.out[i]) {
            const Edge& ed = g.edges[e];
            int d = ed.distance;
            // A dma.wait holds the frontend until the transfer completes, so count the transfer time.
            const Instr& to = g.nodes[ed.to];
            if (to.op->opClass == OpClass::DmaWait && to.op->channel == g.nodes[i].op->channel)
                d = std::max(d, g.footprints[i].dmaCycles + 2);
            height[i] = std::max(height[i], d + height[ed.to]);
        }
    }
    return height;
}
