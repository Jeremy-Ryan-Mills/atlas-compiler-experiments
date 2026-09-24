#pragma once

#include <string>
#include <vector>

#include "core/blocks.h"
#include "core/depgraph.h"
#include "core/simulator.h"

// One dependency graph together with the cycle each node issues at.
struct GraphView {
    DepGraph graph;
    std::vector<int> cycles;       // issue cycle of each node, relative to the block start
    std::vector<bool> redundant;   // edges implied by longer paths
    int length = 0;                // cycles until the next block can start
};

struct BlockView {
    std::string name;
    GraphView before, after;
    int lowerBound = 0;            // critical path length of the dependency graph
};

struct ProgramView {
    std::string source;
    std::vector<BlockView> blocks;
    SimResult before, after;
    std::vector<std::string> log;
};

// Builds the before/after graphs of every block. `optimized` must have the same
// blocks as `original` (the passes keep the block structure).
ProgramView buildProgramView(const std::string& source, const AsmProgram& original, const Code& optimized,
                             const SimResult& before, const SimResult& after, const std::vector<std::string>& log);

// Self-contained HTML page (no network access needed) showing each block's graph
// before and after optimization, laid out on a cycle timeline or by dependency depth.
std::string renderHtml(const ProgramView& view);
