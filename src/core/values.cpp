#include "core/values.h"

#include <deque>

RegValues unknownRegs() {
    RegValues r;
    r[0] = 0;
    return r;
}

RegValues zeroRegs() {
    RegValues r;
    for (auto& v : r) v = 0;
    return r;
}

// Semantics follow npu_model's isa_definition.py (RV32, results masked to 32 bits).
std::optional<uint32_t> aluResult(const Instr& in, const RegValues& regs) {
    const std::string& n = in.op->name;
    if (n == "lui") return (uint32_t)((in.imm & 0xFFFFF) << 12);
    if (!regs[in.rs1]) return std::nullopt;
    uint32_t a = *regs[in.rs1];
    int32_t sa = (int32_t)a;
    if (hasOperand(*in.op, "i")) {
        uint32_t imm = (uint32_t)signExtend(in.imm, 12);
        int shamt = (int)(in.imm & 0x1F);
        if (n == "addi") return a + imm;
        if (n == "slti") return sa < (int32_t)imm ? 1u : 0u;
        if (n == "sltiu") return a < imm ? 1u : 0u;
        if (n == "xori") return a ^ imm;
        if (n == "ori") return a | imm;
        if (n == "andi") return a & imm;
        if (n == "slli") return a << shamt;
        if (n == "srli") return a >> shamt;
        if (n == "srai") return (uint32_t)(sa >> shamt);
        return std::nullopt;
    }
    if (!regs[in.rs2]) return std::nullopt;
    uint32_t b = *regs[in.rs2];
    int32_t sb = (int32_t)b;
    if (n == "add") return a + b;
    if (n == "sub") return a - b;
    if (n == "sll") return a << (b & 0x1F);
    if (n == "slt") return sa < sb ? 1u : 0u;
    if (n == "sltu") return a < b ? 1u : 0u;
    if (n == "xor") return a ^ b;
    if (n == "srl") return a >> (b & 0x1F);
    if (n == "sra") return (uint32_t)(sa >> (b & 0x1F));
    if (n == "or") return a | b;
    if (n == "and") return a & b;
    return std::nullopt;
}

void applyScalar(const Instr& in, RegValues& regs) {
    OpClass c = in.op->opClass;
    bool writesX = c == OpClass::Alu || c == OpClass::Csr || c == OpClass::Jump || c == OpClass::ScalarLoad;
    if (!writesX || in.rd == 0) return;
    regs[in.rd] = c == OpClass::Alu ? aluResult(in, regs) : std::nullopt;
}

static bool mergeInto(RegValues& into, const RegValues& from) {
    bool changed = false;
    for (int r = 1; r < 32; r++)
        if (into[r] && (!from[r] || *from[r] != *into[r])) into[r].reset(), changed = true;
    return changed;
}

std::vector<RegValues> blockEntryValues(const Code& code) {
    size_t n = code.blocks.size();
    std::vector<RegValues> entry(n);
    std::vector<bool> reached(n, false);
    if (n == 0) return entry;
    entry[0] = zeroRegs();
    reached[0] = true;
    bool anyUnknownJump = false;

    std::deque<int> work = {0};
    while (!work.empty()) {
        int bi = work.front();
        work.pop_front();
        const Block& b = code.blocks[bi];
        RegValues regs = entry[bi];
        for (const Instr& in : b.body) applyScalar(in, regs);
        if (b.terminator) applyScalar(*b.terminator, regs);
        if (b.slot) applyScalar(*b.slot, regs);
        anyUnknownJump |= b.unknownSuccs;
        for (int s : b.succs) {
            if (!reached[s]) {
                entry[s] = regs, reached[s] = true, work.push_back(s);
            } else if (mergeInto(entry[s], regs)) {
                work.push_back(s);
            }
        }
    }
    // Blocks that are unreachable or reached through jalr get no assumptions.
    for (size_t i = 0; i < n; i++)
        if (!reached[i] || (anyUnknownJump && i > 0)) entry[i] = unknownRegs();
    return entry;
}
