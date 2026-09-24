#pragma once

#include <optional>
#include <string>
#include <vector>

#include "core/asm.h"

// A basic block. A taken branch or jump runs exactly one more instruction (its
// delay slot) before the target, and that slot also runs when the branch is not
// taken. So the slot is kept with its branch instead of with the next block.
struct Block {
    std::vector<std::string> labels;
    std::vector<Instr> body;
    std::optional<Instr> terminator;  // branch, jump, or ecall/ebreak ending the block
    std::optional<Instr> slot;        // delay-slot instruction of a branch/jump (empty = nop)
    std::vector<int> succs;           // indices of successor blocks
    bool unknownSuccs = false;        // jalr: successors not known statically

    // Filled in by the scheduler (all cycles relative to the block start).
    bool scheduled = false;
    std::vector<int> issue;           // issue cycle of body[i]; body is kept in issue order
    int terminatorCycle = -1;
    int endCycle = 0;                 // earliest cycle the next block may issue its first instruction
};

struct Code {
    std::vector<Block> blocks;
    std::vector<std::string> endLabels;  // labels placed after the last instruction
};

// Splits a flat program into basic blocks and computes successors.
Code buildBlocks(const AsmProgram& prog);

// Turns blocks back into a flat program. Scheduled blocks get `delay`
// instructions for their idle cycles; others are emitted as they are.
AsmProgram flatten(const Code& code);

bool hasDelaySlot(const Block& b);  // block ends with a branch or jump
int naturalGap(const Instr& in);    // cycles until the next issue when nothing stalls (1, or N+1 for delay N)

// The block's instructions in program order: body, terminator, then the delay
// slot (a nop if the slot is empty).
std::vector<Instr> blockInstructions(const Block& b);

// Issue cycle of each instruction when run as written (a dma.wait releases at once).
std::vector<int> asWrittenCycles(const std::vector<Instr>& instrs);
