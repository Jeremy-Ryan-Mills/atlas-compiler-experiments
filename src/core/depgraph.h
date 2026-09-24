#pragma once

#include <string>
#include <vector>

#include "core/machine.h"

// b must issue at least `distance` cycles after a (from = a, to = b).
struct Edge {
    int from, to;
    int distance;
    EdgeKind kind;
    std::string reason;
};

// Dependency graph of a straight-line list of instructions (one basic block).
// Nodes are the instructions in program order, so every edge points forward.
struct DepGraph {
    std::vector<Instr> nodes;
    std::vector<Footprint> footprints;
    std::vector<Edge> edges;
    std::vector<std::vector<int>> in, out;  // edge indices entering / leaving each node
};

// `dmaRegs` is a bit mask of the x registers that DMA commands anywhere in the
// program read (they read them when the transfer completes).
DepGraph buildGraph(const std::vector<Instr>& instrs, const RegValues& entry, uint32_t dmaRegs = 0xFFFFFFFE);

uint32_t dmaOperandRegisters(const std::vector<Instr>& instrs);

// Longest path (in cycles) from each node until everything after it has finished.
// The scheduler issues the instructions with the largest height first.
std::vector<int> criticalHeights(const DepGraph& g);
