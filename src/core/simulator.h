#pragma once

#include <string>
#include <vector>

#include "core/asm.h"

struct SimOptions {
    long long maxCycles = 50000000;
    double dmaLatencyScale = 1.0;  // > 1 stretches every DMA transfer, to test robustness
    int maxViolations = 20;
};

struct SimResult {
    long long cycles = 0;       // matches npu_model's cycle count for the same program
    long long issued = 0;       // issued instructions, including delays but not halt
    long long delays = 0;       // dynamic `delay` instructions
    std::vector<std::string> violations;          // broken dependences or hardware rules
    std::string stopReason;                       // empty when the program ran to its end
};

// RTL-match timing simulation with scalar execution and in-flight hazard checks.
// Halt reports unfinished work; natural falloff drains units and delays.
SimResult simulate(const AsmProgram& prog, const SimOptions& opt = {});
