#pragma once

#include <string>
#include <vector>

#include "core/blocks.h"

// Settings and results shared by the passes of one run.
struct PassContext {
    bool robustDma = true;         // never assume when a dma.wait releases (OPEN_QUESTIONS.md #1)
    std::vector<std::string> log;  // each pass adds a line describing what it did
};

// A pass rewrites the blocks of a program in place. See src/passes/README.md.
struct Pass {
    const char* name;         // used with --passes on the command line
    const char* description;
    void (*run)(Code& code, PassContext& ctx);
};

// Every pass, in the order they run.
const std::vector<Pass>& allPasses();

// Runs the named passes in registry order (every pass if `names` is empty).
void runPasses(Code& code, const std::vector<std::string>& names, PassContext& ctx);

// The passes (one .cpp file each).
void stripArtifacts(Code& code, PassContext& ctx);
void fillDelaySlots(Code& code, PassContext& ctx);
void schedule(Code& code, PassContext& ctx);
