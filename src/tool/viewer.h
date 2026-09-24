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
};

// Builds the before/after graphs of every optimized block. Each is paired with the
// original blocks its instructions came from (several, if a pass merged blocks).
ProgramView buildProgramView(const std::string& source, const AsmProgram& original, const Code& optimized,
                             const SimResult& before, const SimResult& after);

// Self-contained HTML page showing each block's dependency graph before and after
// optimization, with every instruction placed at the cycle it issues.
std::string renderHtml(const ProgramView& view);
