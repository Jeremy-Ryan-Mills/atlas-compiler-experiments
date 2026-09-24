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
    long long issued = 0;       // dynamic instructions issued (including delays)
    long long delays = 0;       // dynamic `delay` instructions
    long long dmaBusy = 0;      // cycles the DMA engine spends transferring (a lower bound on `cycles`)
    std::vector<std::string> violations;          // broken dependences or hardware rules
    std::string stopReason;                       // empty when the program ran to its end
};

// Runs the program on a timing model of the rtl-match core: scalar code is executed
// for real (so loops and addresses are exact), and every issued instruction is
// checked against everything still in flight.
SimResult simulate(const AsmProgram& prog, const SimOptions& opt = {});
