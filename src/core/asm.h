#pragma once

#include <stdexcept>
#include <string>
#include <vector>

// Execution engines of the Atlas core (npu_model rtl-match).
enum class Engine { Scalar, Lsu, Mxu0, Mxu1, Vpu, Xlu, Dma };

// Groups of instructions that share the same timing behavior.
enum class OpClass {
    Alu, Csr, Branch, Jump, Delay, Halt, Fence,
    ScalarLoad, ScalarStore, ScaleLoad, ScaleImm,
    VLoad, VStore,
    WeightPush, AccPushFp8, AccPushBf16, PopFp8, PopBf16, MatMul, MatMulAcc,
    VpuElementwise, VpuPack, VpuUnpack, VpuRowReduce, VpuColReduce, VpuLoadImmPair, VpuLoadImmSingle,
    Transpose,
    DmaLoad, DmaStore, DmaConfig, DmaWait,
};

// One opcode. `operands` lists the assembly operands in order, one token each:
// a register kind (x, e, m, w or a for acc) followed by the field it goes in
// (d = rd, 1 = rs1, 2 = rs2), or i (immediate), t (branch target), @ (imm(x rs1)).
// For example "xd @" is `lw x1, 8(x2)` and "ad m1 w2" is `vmatmul.mxu0 acc0, m0, w0`.
struct OpInfo {
    std::string name;
    std::string operands;
    OpClass opClass;
    Engine engine;
    int mxu = -1;           // 0 or 1 for MXU instructions
    int channel = -1;       // DMA channel for dma.* instructions
    bool twoInput = false;  // VPU binary ops read two register pairs
};

const OpInfo* findOp(const std::string& name);  // nullptr if not an Atlas instruction
const char* engineName(Engine e);
bool hasOperand(const OpInfo& op, const std::string& token);  // e.g. hasOperand(op, "x1")
bool isControlFlow(const OpInfo& op);                         // branches and jumps (one delay slot)

// One assembly instruction. rd/rs1/rs2 hold x, e, m, w or acc register numbers
// depending on the opcode's operand list.
struct Instr {
    const OpInfo* op = nullptr;
    int rd = 0, rs1 = 0, rs2 = 0;
    long long imm = 0;
    std::string immText;      // immediate as written in the source, reused when printing
    std::string target;       // branch / jal label
    std::string comment;      // trailing comment, without the '#'
    std::vector<std::string> leadingComments;  // full-line comments right above it
    bool keep = false;        // "delay N # keep" is never removed by the optimizer
    int line = 0;             // source line, 0 if created by the optimizer
};

// A flat program, as read from (or written to) a .S file.
struct AsmProgram {
    std::vector<Instr> instrs;
    // labels[i] holds the labels placed right before instruction i (size = instrs.size() + 1).
    std::vector<std::vector<std::string>> labels;
};

struct ParseError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

AsmProgram parseAsm(const std::string& text, const std::string& fileName = "<input>");
AsmProgram readAsmFile(const std::string& path);
std::string formatInstr(const Instr& in);
std::string printAsm(const AsmProgram& prog);

Instr makeInstr(const std::string& name, int rd = 0, int rs1 = 0, int rs2 = 0, long long imm = 0);
Instr makeNop();
Instr makeDelay(int cycles);
bool isNop(const Instr& in);  // scalar ALU op writing x0: it has no effect
long long signExtend(long long value, int bits);
