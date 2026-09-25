#include "core/asm.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <sstream>

// ---------------------------------------------------------------- opcode table

static std::vector<OpInfo> buildTable() {
    std::vector<OpInfo> t;
    auto add = [&](const std::string& name, const char* operands, OpClass c, Engine e) {
        t.push_back(OpInfo{name, operands, c, e});
    };
    for (const char* n : {"add", "sub", "sll", "slt", "sltu", "xor", "srl", "sra", "or", "and"})
        add(n, "xd x1 x2", OpClass::Alu, Engine::Scalar);
    for (const char* n : {"addi", "slti", "sltiu", "xori", "ori", "andi", "slli", "srli", "srai"})
        add(n, "xd x1 i", OpClass::Alu, Engine::Scalar);
    add("lui", "xd i", OpClass::Alu, Engine::Scalar);
    add("auipc", "xd i", OpClass::Alu, Engine::Scalar);
    for (const char* n : {"csrrw", "csrrs", "csrrc", "csrrwi", "csrrsi", "csrrci"})
        add(n, "xd x1 i", OpClass::Csr, Engine::Scalar);
    for (const char* n : {"beq", "bne", "blt", "bge", "bltu", "bgeu"})
        add(n, "x1 x2 t", OpClass::Branch, Engine::Scalar);
    add("jal", "xd t", OpClass::Jump, Engine::Scalar);
    add("jalr", "xd x1 i", OpClass::Jump, Engine::Scalar);
    add("delay", "i", OpClass::Delay, Engine::Scalar);
    add("ecall", "", OpClass::Halt, Engine::Scalar);
    add("ebreak", "", OpClass::Halt, Engine::Scalar);
    add("fence", "", OpClass::Fence, Engine::Scalar);
    add("seli", "ed i", OpClass::ScaleImm, Engine::Scalar);

    for (const char* n : {"lb", "lh", "lw", "lbu", "lhu"}) add(n, "xd @", OpClass::ScalarLoad, Engine::Lsu);
    for (const char* n : {"sb", "sh", "sw"}) add(n, "x2 @", OpClass::ScalarStore, Engine::Lsu);
    add("seld", "ed @", OpClass::ScaleLoad, Engine::Lsu);
    add("vload", "md @", OpClass::VLoad, Engine::Lsu);
    add("vstore", "md @", OpClass::VStore, Engine::Lsu);

    for (int m = 0; m < 2; m++) {
        Engine e = m == 0 ? Engine::Mxu0 : Engine::Mxu1;
        std::string s = ".mxu" + std::to_string(m);
        add("vmatpush.weight" + s, "wd m1", OpClass::WeightPush, e);
        add("vmatpush.acc.fp8" + s, "ad m1", OpClass::AccPushFp8, e);
        add("vmatpush.acc.bf16" + s, "ad m1", OpClass::AccPushBf16, e);
        add("vmatpop.fp8.acc" + s, "md a2 e1", OpClass::PopFp8, e);
        add("vmatpop.bf16.acc" + s, "md a2", OpClass::PopBf16, e);
        add("vmatmul" + s, "ad m1 w2", OpClass::MatMul, e);
        add("vmatmul.acc" + s, "ad m1 w2", OpClass::MatMulAcc, e);
        for (int i = (int)t.size() - 7; i < (int)t.size(); i++) t[i].mxu = m;
    }

    for (const char* n : {"vadd.bf16", "vsub.bf16", "vmul.bf16", "vminimum.bf16", "vmaximum.bf16"}) {
        add(n, "md m1 m2", OpClass::VpuElementwise, Engine::Vpu);
        t.back().twoInput = true;
    }
    for (const char* n : {"vmov", "vrecip.bf16", "vexp.bf16", "vexp2.bf16", "vrelu.bf16", "vsin.bf16",
                          "vcos.bf16", "vtanh.bf16", "vlog2.bf16", "vsqrt.bf16", "vsquare.bf16", "vcube.bf16"})
        add(n, "md m1", OpClass::VpuElementwise, Engine::Vpu);
    for (const char* n : {"vredsum.row.bf16", "vredmin.row.bf16", "vredmax.row.bf16"})
        add(n, "md m1", OpClass::VpuRowReduce, Engine::Vpu);
    for (const char* n : {"vredsum.bf16", "vredmin.bf16", "vredmax.bf16"})
        add(n, "md m1", OpClass::VpuColReduce, Engine::Vpu);
    add("vpack.bf16.fp8", "md m2 e1", OpClass::VpuPack, Engine::Vpu);
    add("vunpack.fp8.bf16", "md m2 e1", OpClass::VpuUnpack, Engine::Vpu);
    for (const char* n : {"vli.all", "vli.row"}) add(n, "md i", OpClass::VpuLoadImmPair, Engine::Vpu);
    for (const char* n : {"vli.col", "vli.one"}) add(n, "md i", OpClass::VpuLoadImmSingle, Engine::Vpu);
    add("vtrpose.xlu", "md m1", OpClass::Transpose, Engine::Xlu);

    for (int ch = 0; ch < 8; ch++) {
        std::string s = ".ch" + std::to_string(ch);
        add("dma.load" + s, "xd x1 x2", OpClass::DmaLoad, Engine::Dma);
        add("dma.store" + s, "xd x1 x2", OpClass::DmaStore, Engine::Dma);
        add("dma.config" + s, "x1", OpClass::DmaConfig, Engine::Dma);
        add("dma.wait" + s, "", OpClass::DmaWait, Engine::Dma);
        for (int i = (int)t.size() - 4; i < (int)t.size(); i++) t[i].channel = ch;
    }
    return t;
}

const OpInfo* findOp(const std::string& name) {
    static const std::vector<OpInfo> table = buildTable();
    static std::map<std::string, const OpInfo*> byName;
    if (byName.empty())
        for (const OpInfo& op : table) byName[op.name] = &op;
    auto it = byName.find(name);
    return it == byName.end() ? nullptr : it->second;
}

bool isControlFlow(const OpInfo& op) { return op.opClass == OpClass::Branch || op.opClass == OpClass::Jump; }

// ---------------------------------------------------------------- helpers

long long signExtend(long long value, int bits) {
    value &= (1LL << bits) - 1;
    if (value & (1LL << (bits - 1))) value -= 1LL << bits;
    return value;
}

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    return s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}

static std::string lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

static std::vector<std::string> tokenize(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',' || std::isspace((unsigned char)c)) {
            if (!cur.empty()) out.push_back(cur), cur.clear();
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

bool hasOperand(const OpInfo& op, const std::string& token) {
    std::vector<std::string> spec = tokenize(op.operands);
    return std::find(spec.begin(), spec.end(), token) != spec.end();
}

static bool isLabelName(const std::string& s) {
    if (s.empty() || !(std::isalpha((unsigned char)s[0]) || s[0] == '_')) return false;
    for (char c : s)
        if (!(std::isalnum((unsigned char)c) || c == '_' || c == '.')) return false;
    return true;
}

// Parses a decimal, 0x hex or 0b binary integer, with an optional sign.
static bool parseInt(const std::string& text, long long& value) {
    std::string s = lower(text);
    size_t i = 0;
    bool negative = false;
    if (i < s.size() && (s[i] == '-' || s[i] == '+')) negative = s[i++] == '-';
    int base = 10;
    if (s.compare(i, 2, "0x") == 0) base = 16, i += 2;
    else if (s.compare(i, 2, "0b") == 0) base = 2, i += 2;
    if (i >= s.size()) return false;
    long long v = 0;
    for (; i < s.size(); i++) {
        char c = s[i];
        int d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : 99;
        if (d >= base) return false;
        v = v * base + d;
    }
    value = negative ? -v : v;
    return true;
}

static long long parseImm(const std::string& tok, std::string* text = nullptr) {
    long long v;
    if (!parseInt(tok, v)) throw ParseError("expected integer, got '" + tok + "'");
    if (text) *text = tok;
    return v;
}

static int parseXReg(const std::string& tok) {
    long long n;
    if (tok.size() > 1 && tok[0] == 'x' && parseInt(tok.substr(1), n) && n >= 0 && n < 32) return (int)n;
    throw ParseError("expected scalar register, got '" + tok + "'");
}

// Register kinds in operand lists: prefix in assembly and number of registers.
static std::string regPrefix(char kind) { return kind == 'a' ? "acc" : std::string(1, kind); }
static int regCount(char kind) { return kind == 'x' || kind == 'e' ? 32 : kind == 'm' ? 64 : 2; }

static int parseReg(const std::string& tok, char kind) {
    if (kind == 'x') return parseXReg(tok);
    std::string t = lower(tok), prefix = regPrefix(kind);
    long long n;
    if (t.compare(0, prefix.size(), prefix) == 0 && parseInt(t.substr(prefix.size()), n) && n >= 0 && n < regCount(kind))
        return (int)n;
    throw ParseError("expected " + prefix + "0.." + prefix + std::to_string(regCount(kind) - 1) + ", got '" + tok + "'");
}

static int& field(Instr& in, char f) { return f == 'd' ? in.rd : f == '1' ? in.rs1 : in.rs2; }
static int field(const Instr& in, char f) { return f == 'd' ? in.rd : f == '1' ? in.rs1 : in.rs2; }

// ---------------------------------------------------------------- instructions

Instr makeInstr(const std::string& name, int rd, int rs1, int rs2, long long imm) {
    Instr in;
    in.op = findOp(name);
    if (!in.op) throw ParseError("unknown instruction '" + name + "'");
    in.rd = rd, in.rs1 = rs1, in.rs2 = rs2, in.imm = imm;
    return in;
}

Instr makeNop() { return makeInstr("addi"); }
Instr makeDelay(int cycles) { return makeInstr("delay", 0, 0, 0, cycles); }
bool isNop(const Instr& in) { return in.op->opClass == OpClass::Alu && in.rd == 0; }

// Same expansion as npu_model's converter.expand_li, so both tools agree on addresses.
static std::vector<Instr> expandLi(int rd, long long value) {
    long long v = value & 0xFFFFFFFFLL;
    if (v >= 0x80000000LL) v -= 0x100000000LL;
    if (v >= -2048 && v <= 2047) return {makeInstr("addi", rd, 0, 0, v & 0xFFF)};
    long long lo = signExtend(v & 0xFFF, 12);
    long long hi = ((v - lo) >> 12) & 0xFFFFF;
    if (lo == 0) return {makeInstr("lui", rd, 0, 0, hi)};
    return {makeInstr("lui", rd, 0, 0, hi), makeInstr("addi", rd, rd, 0, lo & 0xFFF)};
}

// Match an exact, case-sensitive, whitespace-delimited token.
static bool hasReleaseToken(const std::string& comment) {
    std::istringstream words(comment);
    for (std::string word; words >> word;)
        if (word == "atlas.release") return true;
    return false;
}

// Fills the operands of `in` from the tokens after the mnemonic. A branch target
// written as a number (a word offset) is reported through `numericTarget`.
static void parseOperands(Instr& in, const std::vector<std::string>& toks, bool& numericTarget) {
    std::vector<std::string> spec = tokenize(in.op->operands);
    numericTarget = false;
    if (toks.size() != spec.size() + 1)
        throw ParseError("'" + toks[0] + "' expects " + std::to_string(spec.size()) + " operands, got " +
                         std::to_string(toks.size() - 1));
    for (size_t i = 0; i < spec.size(); i++) {
        const std::string& tok = toks[i + 1];
        char kind = spec[i][0];
        if (kind == 'i') {
            in.imm = parseImm(tok, &in.immText);
        } else if (kind == 't') {
            if (isLabelName(tok)) in.target = tok;
            else in.imm = parseImm(tok), numericTarget = true;
        } else if (kind == '@') {  // imm(xN)
            size_t open = tok.find('('), close = tok.find(')');
            if (open == std::string::npos || close == std::string::npos || close < open)
                throw ParseError("expected imm(xN), got '" + tok + "'");
            in.imm = parseImm(open == 0 ? "0" : tok.substr(0, open), &in.immText);
            in.rs1 = parseXReg(tok.substr(open + 1, close - open - 1));
        } else {
            field(in, spec[i][1]) = parseReg(tok, kind);
        }
    }
}

AsmProgram parseAsm(const std::string& text, const std::string& fileName) {
    AsmProgram prog;
    prog.labels.emplace_back();
    std::vector<std::pair<int, long long>> numericTargets;  // (instruction index, word offset)
    std::istringstream stream(text);
    std::string raw;
    int lineNo = 0;
    while (std::getline(stream, raw)) {
        lineNo++;
        size_t hash = raw.find('#');
        std::string line = trim(raw.substr(0, hash));
        std::string comment = hash == std::string::npos ? "" : trim(raw.substr(hash + 1));
        try {
            size_t colon = line.find(':');  // "name:" defines a label; "name: instr" is also accepted
            if (colon != std::string::npos && isLabelName(trim(line.substr(0, colon)))) {
                prog.labels.back().push_back(trim(line.substr(0, colon)));
                line = trim(line.substr(colon + 1));
            }
            bool release = hasReleaseToken(comment);
            if (line.empty()) {
                if (release) throw ParseError("atlas.release must annotate a CSR instruction");
                continue;
            }
            std::vector<std::string> toks = tokenize(line);
            std::string name = lower(toks[0]);
            std::vector<Instr> produced;
            if (name == "nop" && toks.size() == 1) {
                produced.push_back(makeNop());
            } else if (name == "li" && toks.size() == 3) {
                produced = expandLi(parseXReg(toks[1]), parseImm(toks[2]));
            } else {
                Instr in;
                in.op = findOp(name);
                if (!in.op) throw ParseError("unknown instruction '" + toks[0] + "'");
                bool numeric;
                parseOperands(in, toks, numeric);
                if (numeric) numericTargets.push_back({(int)prog.instrs.size(), in.imm});
                produced.push_back(in);
            }
            for (size_t i = 0; i < produced.size(); i++) {
                Instr& in = produced[i];
                in.line = lineNo;
                if (i == 0) in.comment = comment;
                in.keep = in.op->opClass == OpClass::Delay && comment.find("keep") != std::string::npos;
                in.release = release;
                if (in.release && in.op->opClass != OpClass::Csr)
                    throw ParseError("atlas.release is only valid on a CSR instruction");
                prog.instrs.push_back(in);
                prog.labels.emplace_back();
            }
        } catch (const ParseError& e) {
            throw ParseError(fileName + ":" + std::to_string(lineNo) + ": " + e.what());
        }
    }

    std::map<std::string, int> defined;
    for (size_t i = 0; i < prog.labels.size(); i++)
        for (const std::string& l : prog.labels[i]) {
            if (defined.count(l)) throw ParseError(fileName + ": label '" + l + "' defined twice");
            defined[l] = (int)i;
        }
    // Numeric branch offsets count instruction words; give each target a label so it survives reordering.
    for (auto [index, offset] : numericTargets) {
        long long dest = index + offset;
        if (dest < 0 || dest > (long long)prog.instrs.size())
            throw ParseError(fileName + ":" + std::to_string(prog.instrs[index].line) + ": branch target out of range");
        std::string name = "L_" + std::to_string(dest);
        while (defined.count(name) && defined[name] != dest) name += "_";
        if (!defined.count(name)) prog.labels[dest].push_back(name), defined[name] = (int)dest;
        prog.instrs[index].target = name;
        prog.instrs[index].imm = 0;
    }
    for (const Instr& in : prog.instrs)
        if (!in.target.empty() && !defined.count(in.target))
            throw ParseError(fileName + ":" + std::to_string(in.line) + ": undefined label '" + in.target + "'");
    return prog;
}

AsmProgram readAsmFile(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw ParseError("cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return parseAsm(ss.str(), path);
}

std::string formatInstr(const Instr& in) {
    std::string imm = in.immText.empty() ? std::to_string(in.imm) : in.immText;
    std::string s = in.op->name;
    std::vector<std::string> spec = tokenize(in.op->operands);
    for (size_t i = 0; i < spec.size(); i++) {
        s += i == 0 ? " " : ", ";
        char kind = spec[i][0];
        if (kind == 'i') s += imm;
        else if (kind == 't') s += in.target.empty() ? imm : in.target;
        else if (kind == '@') s += imm + "(x" + std::to_string(in.rs1) + ")";
        else s += regPrefix(kind) + std::to_string(field(in, spec[i][1]));
    }
    return s;
}

std::string printAsm(const AsmProgram& prog) {
    std::string out;
    for (size_t i = 0; i <= prog.instrs.size(); i++) {
        if (i < prog.labels.size())
            for (const std::string& l : prog.labels[i]) out += l + ":\n";
        if (i == prog.instrs.size()) break;
        const Instr& in = prog.instrs[i];
        if (in.release && in.op->opClass != OpClass::Csr)
            throw ParseError("atlas.release is only valid on a CSR instruction");
        bool token = hasReleaseToken(in.comment);
        if (token && !in.release)
            throw ParseError("atlas.release comment disagrees with the instruction's release flag");
        std::string comment = in.comment;
        if (in.release && !token) comment += (comment.empty() ? "" : " ") + std::string("atlas.release");
        out += formatInstr(in);
        if (!comment.empty()) out += "   # " + comment;
        out += "\n";
    }
    return out;
}
