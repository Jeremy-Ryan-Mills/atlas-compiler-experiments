#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "core/blocks.h"

// Known values of the scalar registers at one program point (nullopt = unknown).
using RegValues = std::array<std::optional<uint32_t>, 32>;

RegValues unknownRegs();  // x0 = 0, everything else unknown
RegValues zeroRegs();     // all registers 0 (the model's reset state)

// Result of a scalar ALU instruction, or nullopt if an operand is unknown.
std::optional<uint32_t> aluResult(const Instr& in, const RegValues& regs);

// Updates `regs` with the scalar register written by `in` (if any).
void applyScalar(const Instr& in, RegValues& regs);

// Register values known at the start of each block, by forward constant
// propagation over the control-flow graph. The program starts with all registers 0.
std::vector<RegValues> blockEntryValues(const Code& code);
