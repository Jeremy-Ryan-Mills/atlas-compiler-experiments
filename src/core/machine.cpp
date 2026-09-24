#include "core/machine.h"

#include <algorithm>
#include <climits>

int dmaTransferCycles(long long bytes) {
    long long offchip = (bytes + 8 + 3) / 4 * 2;  // 32-bit link, 2 cycles per beat, 2 command words
    long long vmem = (bytes + 63) / 64;
    return (int)std::max(1LL, std::max(offchip, vmem));
}

int unitCapacity(Unit u, int index) {
    if (u == Unit::MxuCompute) return index == 0 ? 3 : 2;
    return 1;
}

const char* unitName(Unit u) {
    switch (u) {
        case Unit::ScalarLoad: return "scalar load path";
        case Unit::ScalarWriteback: return "scalar write port";
        case Unit::VloadPath: return "VLOAD path";
        case Unit::VstorePath: return "VSTORE path";
        case Unit::VmemBank: return "VMEM bank";
        case Unit::Xlu: return "XLU";
        case Unit::MxuPort: return "MXU port";
        case Unit::MxuCompute: return "MXU in-flight matmuls";
        case Unit::MxuAccRead: return "accumulator read";
        case Unit::MxuAccWrite: return "accumulator write";
        case Unit::MxuWeightStream: return "weight push stream";
        case Unit::MxuAccStream: return "accumulator push stream";
    }
    return "?";
}

const char* edgeKindName(EdgeKind k) {
    switch (k) {
        case EdgeKind::RAW: return "RAW";
        case EdgeKind::WAR: return "WAR";
        case EdgeKind::WAW: return "WAW";
        case EdgeKind::Rule: return "rule";
        case EdgeKind::Order: return "order";
    }
    return "?";
}

bool isBarrier(const Instr& in) {
    OpClass c = in.op->opClass;
    return c == OpClass::Delay || c == OpClass::Csr || c == OpClass::Fence;
}

bool vpuUsesBothSlots(const OpInfo& op) {
    return op.twoInput || op.opClass == OpClass::VpuRowReduce;
}

// Lane-logic groups from npu_model's vpu.py: members of a group cannot overlap.
static int vpuLogicGroup(const std::string& name) {
    static const std::vector<std::vector<std::string>> groups = {
        {"vadd.bf16", "vsub.bf16", "vredsum.row.bf16"},
        {"vexp.bf16", "vexp2.bf16"},
        {"vsin.bf16", "vcos.bf16"},
        {"vsquare.bf16", "vcube.bf16"},
        {"vmaximum.bf16", "vredmax.bf16"},
        {"vminimum.bf16", "vredmin.bf16"},
        {"vli.all", "vli.row", "vli.col", "vli.one"},
    };
    for (size_t g = 0; g < groups.size(); g++)
        for (const std::string& n : groups[g])
            if (n == name) return (int)g;
    return -1;
}

bool vpuCanOverlap(const OpInfo& a, const OpInfo& b) {
    if (vpuUsesBothSlots(a) || vpuUsesBothSlots(b)) return false;
    if (a.name == b.name) return false;
    int ga = vpuLogicGroup(a.name);
    return ga < 0 || ga != vpuLogicGroup(b.name);
}

namespace {

// Small helper that collects the accesses and holds of one instruction.
struct Builder {
    Footprint f;

    void x(int reg, bool write, int age, bool atCompletion = false) {
        if (reg == 0) return;
        Access a{Res::XReg, write, reg, 1, age, 1};
        a.atCompletion = atCompletion;
        f.accesses.push_back(a);
    }
    void e(int reg, bool write, int age) { f.accesses.push_back({Res::EReg, write, reg, 1, age, 1}); }
    void mrows(int reg, bool write, int age, int step = 1) {
        if (reg < 0 || reg > 63) {
            f.error = "register pair runs past m63";
            return;
        }
        f.accesses.push_back({Res::MReg, write, reg * 32, 32, age, step});
    }
    void mxuRows(Res res, int mxu, int index, bool write, int age) {
        f.accesses.push_back({res, write, (mxu * 2 + index) * 32, 32, age, 1});
    }
    void hold(Unit u, int index, int from, int to, int alt = -1) { f.holds.push_back({u, index, from, to, alt}); }
    void needEven(int reg, const char* what) {
        if (reg & 1) f.error = std::string(what) + " must name an even register (m" + std::to_string(reg) + ")";
    }

    // VMEM lines [byteAddr, byteAddr + bytes), line i touched at age + i * step.
    void vmem(std::optional<long long> byteAddr, std::optional<long long> bytes, bool write, int age, int step,
              bool atCompletion = false) {
        if (bytes && *bytes <= 0) return;  // a zero-length transfer touches nothing
        Access a{Res::Vmem, write, 0, 1, age, step};
        a.atCompletion = atCompletion;
        if (!byteAddr || !bytes) {
            a.anywhere = true;
            a.count = bytes ? (int)((*bytes + kLineBytes - 1) / kLineBytes) : 1;
        } else {
            long long firstLine = *byteAddr / kLineBytes;
            long long lastLine = (*byteAddr + *bytes - 1) / kLineBytes;
            a.first = (int)firstLine;
            a.count = (int)(lastLine - firstLine + 1);
            if (*byteAddr + *bytes > kVmemBytes) f.error = "VMEM access past the end of VMEM";
        }
        f.accesses.push_back(a);
    }
    void bankHold(std::optional<long long> byteAddr, int from, int to) {
        if (byteAddr) {
            hold(Unit::VmemBank, (int)(*byteAddr / kVmemBankBytes), from, to);
        } else {
            for (int b = 0; b < kVmemBanks; b++) hold(Unit::VmemBank, b, from, to);  // unknown bank
        }
    }
};

std::optional<long long> reg(const RegValues& regs, int r) {
    if (!regs[r]) return std::nullopt;
    return (long long)*regs[r];
}

// Byte address used by scalar loads/stores (ScalarCore keeps only the VMEM address bits).
std::optional<long long> scalarAddress(const Instr& in, const RegValues& regs) {
    auto base = reg(regs, in.rs1);
    if (!base) return std::nullopt;
    return (*base + signExtend(in.imm, 12)) & 0x1FFFFF;
}

// vload/vstore operands are word addresses; the imm adds 32 words per unit and the
// LSU drops the three low word bits to get a 32-byte line.
std::optional<long long> vectorAddress(const Instr& in, const RegValues& regs) {
    auto base = reg(regs, in.rs1);
    if (!base) return std::nullopt;
    long long alu = *base + signExtend(in.imm, 12) * 32;
    return ((alu >> 3) & 0xFFFF) * kLineBytes;
}

void scalarRegs(Builder& b, const Instr& in) {
    const OpInfo& op = *in.op;
    bool immediateCsr = op.opClass == OpClass::Csr && op.name.back() == 'i';  // csrr*i: rs1 holds an immediate
    if (hasOperand(op, "x1") && !immediateCsr) b.x(in.rs1, false, 0);
    if (hasOperand(op, "x2")) b.x(in.rs2, false, 0);
    if (hasOperand(op, "xd")) {
        b.x(in.rd, true, 0);
        if (in.rd != 0) b.hold(Unit::ScalarWriteback, 0, 0, 0);
    }
}

}  // namespace

Footprint footprintOf(const Instr& in, const RegValues& regs) {
    Builder b;
    const OpInfo& op = *in.op;
    int mxu = op.mxu;
    int cf = mxu == 0 ? 63 : 3;  // age of the first accumulator row written by a matmul

    switch (op.opClass) {
        case OpClass::Alu:
        case OpClass::Csr:
        case OpClass::Branch:
        case OpClass::Jump:
            scalarRegs(b, in);
            break;
        case OpClass::Delay:
        case OpClass::Halt:
        case OpClass::Fence:
        case OpClass::DmaWait:
            break;
        case OpClass::ScaleImm:
            b.e(in.rd, true, 0);
            b.hold(Unit::ScalarWriteback, 0, 0, 0);
            break;

        case OpClass::ScalarLoad:
        case OpClass::ScaleLoad: {
            auto addr = scalarAddress(in, regs);
            b.x(in.rs1, false, 0);
            b.vmem(addr ? std::optional<long long>(*addr & ~3LL) : std::nullopt, 4, false, 1, 1);
            if (op.opClass == OpClass::ScaleLoad) b.e(in.rd, true, 3);
            else b.x(in.rd, true, 3);
            b.hold(Unit::ScalarLoad, 0, 0, 2);
            b.hold(Unit::ScalarWriteback, 0, 3, 3);
            b.bankHold(addr, 1, 1);
            break;
        }
        case OpClass::ScalarStore: {
            auto addr = scalarAddress(in, regs);
            int size = in.op->name == "sb" ? 1 : in.op->name == "sh" ? 2 : 4;
            b.x(in.rs1, false, 0);
            b.x(in.rs2, false, 0);
            b.vmem(addr ? std::optional<long long>(*addr & ~(long long)(size - 1)) : std::nullopt, size, true, 1, 1);
            b.bankHold(addr, 1, 1);
            break;
        }
        case OpClass::VLoad:
        case OpClass::VStore: {
            bool load = op.opClass == OpClass::VLoad;
            auto addr = vectorAddress(in, regs);
            if (addr && (*addr % 1024 != 0 || *addr / kVmemBankBytes != (*addr + 1023) / kVmemBankBytes))
                b.f.error = "vload/vstore address must be 1 KiB aligned inside one VMEM bank";
            b.x(in.rs1, false, 0);
            if (load) {
                b.vmem(addr, 1024, false, 1, 1);  // VMEM row r read at age 1+r
                b.mrows(in.rd, true, 3);          // register row r written at age 3+r
                b.f.mregWrites = {in.rd};
                b.f.writeRelease = 34;
                b.f.writeDuringRead = true;
                b.hold(Unit::VloadPath, 0, 0, 34);
                b.bankHold(addr, 1, 32);
            } else {
                b.mrows(in.rd, false, 1);
                b.vmem(addr, 1024, true, 3, 1);
                b.f.mregReads = {in.rd};
                b.f.readRelease = b.f.writeRelease = 34;
                b.hold(Unit::VstorePath, 0, 0, 34);
                b.bankHold(addr, 3, 34);
            }
            break;
        }

        case OpClass::WeightPush:
            b.mrows(in.rs1, false, 0);
            b.mxuRows(Res::Weight, mxu, in.rd, true, 1);
            b.f.mregReads = {in.rs1};
            b.f.readRelease = b.f.writeRelease = 32;
            b.hold(Unit::MxuPort, mxu * 4 + 1, 0, 31, mxu * 4 + 0);
            b.hold(Unit::MxuWeightStream, mxu, 1, 32);
            break;
        case OpClass::AccPushFp8:
        case OpClass::AccPushBf16: {
            bool pair = op.opClass == OpClass::AccPushBf16;
            b.mrows(in.rs1, false, 0);
            if (pair) b.mrows(in.rs1 + 1, false, 0);
            b.mxuRows(Res::Acc, mxu, in.rd, true, 1);
            b.f.mregReads = pair ? std::vector<int>{in.rs1, in.rs1 + 1} : std::vector<int>{in.rs1};
            b.f.readRelease = b.f.writeRelease = 32;
            if (pair) {
                b.hold(Unit::MxuPort, mxu * 4 + 0, 0, 31);
                b.hold(Unit::MxuPort, mxu * 4 + 1, 0, 31);
            } else {
                b.hold(Unit::MxuPort, mxu * 4 + 1, 0, 31, mxu * 4 + 0);
            }
            b.hold(Unit::MxuAccStream, mxu, 1, 32);
            b.hold(Unit::MxuAccWrite, mxu * 2 + in.rd, 1, 32);
            break;
        }
        case OpClass::PopFp8:
        case OpClass::PopBf16: {
            bool pair = op.opClass == OpClass::PopBf16;
            b.mxuRows(Res::Acc, mxu, in.rs2, false, 0);
            if (!pair) b.e(in.rs1, false, 0);
            b.mrows(in.rd, true, 1);
            if (pair) b.mrows(in.rd + 1, true, 1);
            b.f.mregWrites = pair ? std::vector<int>{in.rd, in.rd + 1} : std::vector<int>{in.rd};
            b.f.readRelease = b.f.writeRelease = 32;
            b.hold(Unit::MxuPort, mxu * 4 + 2, 0, 31);
            if (pair) b.hold(Unit::MxuPort, mxu * 4 + 3, 0, 31);
            b.hold(Unit::MxuAccRead, mxu * 2 + in.rs2, 0, 31);
            break;
        }
        case OpClass::MatMul:
        case OpClass::MatMulAcc:
            b.mrows(in.rs1, false, 0);
            b.mxuRows(Res::Weight, mxu, in.rs2, false, 1);
            if (op.opClass == OpClass::MatMulAcc) b.mxuRows(Res::Acc, mxu, in.rd, false, 0);
            b.mxuRows(Res::Acc, mxu, in.rd, true, cf);
            b.f.mregReads = {in.rs1};
            b.f.readRelease = b.f.writeRelease = 32;
            b.hold(Unit::MxuPort, mxu * 4 + 0, 0, 31);
            b.hold(Unit::MxuCompute, mxu, 0, cf + 31);
            b.hold(Unit::MxuAccRead, mxu * 2 + in.rd, 0, 31);
            b.hold(Unit::MxuAccWrite, mxu * 2 + in.rd, cf, cf + 31);
            break;

        case OpClass::VpuElementwise: {
            // A BF16 pair streams its low register in ages 0..31 and its high register in 32..63.
            std::vector<int> sources = {in.rs1};
            if (op.twoInput) sources.push_back(in.rs2);
            for (int s : sources) {
                b.needEven(s, "BF16 source");
                b.mrows(s, false, 0);
                b.mrows(s + 1, false, 32);
                b.f.mregReads.push_back(s);
                b.f.mregReads.push_back(s + 1);
            }
            b.needEven(in.rd, "BF16 destination");
            b.mrows(in.rd, true, 2);
            b.mrows(in.rd + 1, true, 34);
            b.f.mregWrites = {in.rd, in.rd + 1};
            b.f.readRelease = 63, b.f.writeRelease = 65;
            break;
        }
        case OpClass::VpuPack:
            b.needEven(in.rs2, "BF16 source");
            b.mrows(in.rs2, false, 0);
            b.mrows(in.rs2 + 1, false, 32);
            b.e(in.rs1, false, 0);
            b.mrows(in.rd, true, 3, 2);  // one packed FP8 row every other cycle
            b.f.mregReads = {in.rs2, in.rs2 + 1};
            b.f.mregWrites = {in.rd};
            b.f.readRelease = 63, b.f.writeRelease = 65;
            break;
        case OpClass::VpuUnpack:
            b.needEven(in.rd, "BF16 destination");
            b.mrows(in.rs2, false, 0);
            b.e(in.rs1, false, 0);
            b.mrows(in.rd, true, 3);
            b.mrows(in.rd + 1, true, 35);
            b.f.mregReads = {in.rs2};
            b.f.mregWrites = {in.rd, in.rd + 1};
            b.f.readRelease = 31, b.f.writeRelease = 66;
            break;
        case OpClass::VpuRowReduce: {
            int lag = op.name == "vredsum.row.bf16" ? 7 : 2;
            b.needEven(in.rs1, "BF16 source");
            b.needEven(in.rd, "BF16 destination");
            b.mrows(in.rs1, false, 0);
            b.mrows(in.rs1 + 1, false, 0);
            b.mrows(in.rd, true, lag);
            b.mrows(in.rd + 1, true, lag);
            b.f.mregReads = {in.rs1, in.rs1 + 1};
            b.f.mregWrites = {in.rd, in.rd + 1};
            b.f.readRelease = 31, b.f.writeRelease = 31 + lag;
            break;
        }
        case OpClass::VpuColReduce:
            b.needEven(in.rs1, "BF16 source");
            b.needEven(in.rd, "BF16 destination");
            for (int pass = 0; pass < 2; pass++) {  // the source pair is streamed twice
                b.mrows(in.rs1, false, pass * 64);
                b.mrows(in.rs1 + 1, false, pass * 64 + 32);
            }
            b.mrows(in.rd, true, 66);
            b.mrows(in.rd + 1, true, 98);
            b.f.mregReads = {in.rs1, in.rs1 + 1};
            b.f.mregWrites = {in.rd, in.rd + 1};
            b.f.readRelease = 127, b.f.writeRelease = 129;
            break;
        case OpClass::VpuLoadImmPair:
            b.needEven(in.rd, "BF16 destination");
            b.mrows(in.rd, true, 1);
            b.mrows(in.rd + 1, true, 33);
            b.f.mregWrites = {in.rd, in.rd + 1};
            b.f.writeRelease = 64;
            break;
        case OpClass::VpuLoadImmSingle:
            b.mrows(in.rd, true, 1);
            b.f.mregWrites = {in.rd};
            b.f.writeRelease = 32;
            break;
        case OpClass::Transpose:
            b.mrows(in.rs1, false, 1);
            b.mrows(in.rd, true, 34);
            b.f.mregReads = {in.rs1};
            b.f.mregWrites = {in.rd};
            b.f.readRelease = 33, b.f.writeRelease = 65;
            b.hold(Unit::Xlu, 0, 0, 65);
            break;

        case OpClass::DmaLoad:
        case OpClass::DmaStore: {
            // The model reads the DMA's registers and moves the data when the transfer completes.
            bool load = op.opClass == OpClass::DmaLoad;
            int vmemReg = load ? in.rd : in.rs1;
            auto addr = reg(regs, vmemReg);
            auto bytes = reg(regs, in.rs2);
            b.x(in.rd, false, 0, true);
            b.x(in.rs1, false, 0, true);
            b.x(in.rs2, false, 0, true);
            Access base{Res::DmaBase, false, 0, 1, 0, 1};
            base.atCompletion = true;
            b.f.accesses.push_back(base);
            b.vmem(addr, bytes, load, 0, 0, true);
            if (addr && (*addr % 32 != 0)) b.f.error = "DMA VMEM address must be 32-byte aligned";
            b.f.dmaCycles = dmaTransferCycles(bytes ? *bytes : 1024);  // unknown size: guess one tile
            break;
        }
        case OpClass::DmaConfig: {
            b.x(in.rs1, false, 0, true);
            Access base{Res::DmaBase, true, 0, 1, 0, 1};
            base.atCompletion = true;
            b.f.accesses.push_back(base);
            b.f.dmaCycles = dmaTransferCycles(0);  // npu_model sizes it from x0: 4 cycles
            break;
        }
    }

    Footprint& f = b.f;
    if (op.engine == Engine::Vpu) f.vpuLive = f.writeRelease;  // a VPU slot frees on its last write cycle
    f.doneAge = std::max(f.readRelease, f.writeRelease);
    for (const Access& a : f.accesses)
        if (!a.atCompletion) f.doneAge = std::max(f.doneAge, a.lastAge());
    for (const Hold& h : f.holds) f.doneAge = std::max(f.doneAge, h.to);
    return f;
}

static std::string elementName(Res res, int element) {
    switch (res) {
        case Res::XReg: return "x" + std::to_string(element);
        case Res::EReg: return "e" + std::to_string(element);
        case Res::MReg: return "m" + std::to_string(element / 32);
        case Res::Acc: return "mxu" + std::to_string(element / 64) + ".acc" + std::to_string(element / 32 % 2);
        case Res::Weight: return "mxu" + std::to_string(element / 64) + ".w" + std::to_string(element / 32 % 2);
        case Res::Vmem: {
            char buf[32];
            snprintf(buf, sizeof buf, "VMEM 0x%x", element * kLineBytes);
            return buf;
        }
        case Res::DmaBase: return "dma.base";
    }
    return "?";
}

// Distance needed so every element y touches is ordered after x touches it.
// Returns INT_MIN when the two accesses cannot touch the same element.
static int dataDistance(const Access& x, const Access& y, int& element) {
    // y's DMA accesses happen at completion, which can be right after issue.
    auto ageY = [&](long long e) { return y.atCompletion ? 0LL : y.age + (e - y.first) * y.step; };
    auto ageX = [&](long long e) { return x.age + (e - x.first) * x.step; };
    if (x.anywhere || y.anywhere) {
        element = x.anywhere ? y.first : x.first;
        int minY = y.atCompletion ? 0 : y.age;
        return x.lastAge() - minY + 1;
    }
    long long lo = std::max(x.first, y.first);
    long long hi = std::min(x.first + x.count - 1, y.first + y.count - 1);
    if (lo > hi) return INT_MIN;
    element = (int)lo;
    return (int)std::max(ageX(lo) - ageY(lo), ageX(hi) - ageY(hi)) + 1;
}

static bool contains(const std::vector<int>& v, int x) { return std::find(v.begin(), v.end(), x) != v.end(); }

Dependence dependence(const Instr& a, const Footprint& fa, const Instr& b, const Footprint& fb) {
    Dependence best;
    auto consider = [&](int d, EdgeKind kind, const std::string& why) {
        if (d > best.distance) best = {d, kind, why};
    };
    const OpInfo& A = *a.op;
    const OpInfo& B = *b.op;

    if (isBarrier(a)) consider(A.opClass == OpClass::Delay ? 1 + (int)(a.imm & 0xFFF) : 1, EdgeKind::Order, A.name + " is a barrier");
    if (isBarrier(b)) consider(1, EdgeKind::Order, B.name + " is a barrier");

    // DMA commands leave the queue in issue order; waits stay ordered with their channel.
    if (A.engine == Engine::Dma && B.engine == Engine::Dma) {
        bool aWait = A.opClass == OpClass::DmaWait, bWait = B.opClass == OpClass::DmaWait;
        if (!aWait && !bWait) consider(1, EdgeKind::Order, "DMA queue order");
        else if (A.channel == B.channel) consider(1, EdgeKind::Order, "DMA channel " + std::to_string(A.channel));
    }

    // Row/element timing of every shared piece of storage.
    for (const Access& x : fa.accesses) {
        if (x.atCompletion) continue;  // handled through the matching dma.wait
        for (const Access& y : fb.accesses) {
            if (x.res != y.res || (!x.write && !y.write)) continue;
            int element = 0;
            int d = dataDistance(x, y, element);
            if (d == INT_MIN) continue;
            EdgeKind kind = x.write && y.write ? EdgeKind::WAW : x.write ? EdgeKind::RAW : EdgeKind::WAR;
            consider(std::max(d, 1), kind, std::string(edgeKindName(kind)) + " on " + elementName(x.res, element));
        }
    }

    // ScalarCore's logical MREG reservations are held until their release age.
    for (int r : fa.mregWrites) {
        if (contains(fb.mregReads, r))
            consider(fa.writeRelease + 1, EdgeKind::RAW, "m" + std::to_string(r) + " reserved for writing until age " + std::to_string(fa.writeRelease));
        if (contains(fb.mregWrites, r))
            consider(fa.writeRelease + 1, EdgeKind::WAW, "m" + std::to_string(r) + " reserved for writing until age " + std::to_string(fa.writeRelease));
    }
    if (!fb.writeDuringRead)
        for (int r : fa.mregReads)
            if (contains(fb.mregWrites, r))
                consider(fa.readRelease + 1, EdgeKind::WAR, "m" + std::to_string(r) + " reserved for reading until age " + std::to_string(fa.readRelease));

    // MXU sequencer rules (npu_model mxu.py).
    if (A.mxu >= 0 && A.mxu == B.mxu) {
        int m = A.mxu;
        std::string mx = "MXU" + std::to_string(m) + ": ";
        auto isCompute = [](const OpInfo& o) { return o.opClass == OpClass::MatMul || o.opClass == OpClass::MatMulAcc; };
        auto isAccPush = [](const OpInfo& o) { return o.opClass == OpClass::AccPushFp8 || o.opClass == OpClass::AccPushBf16; };
        auto accOf = [&](const Instr& in) {
            const OpInfo& o = *in.op;
            if (isCompute(o) || isAccPush(o)) return in.rd;
            if (o.opClass == OpClass::PopBf16 || o.opClass == OpClass::PopFp8) return in.rs2;
            return -1;
        };
        auto slotOf = [&](const Instr& in) {
            if (isCompute(*in.op)) return in.rs2;
            if (in.op->opClass == OpClass::WeightPush) return in.rd;
            return -1;
        };
        bool bWeight = B.opClass == OpClass::WeightPush;
        if (isCompute(A) && !bWeight && accOf(a) == accOf(b))
            consider(m == 0 ? 64 : 4, EdgeKind::Rule, mx + "wait until row 0 of the matmul on acc" + std::to_string(accOf(a)) + " is written");
        if (m == 0 && isCompute(A) && bWeight && slotOf(a) == slotOf(b))
            consider(63, EdgeKind::Rule, mx + "weight slot still feeding the systolic array");
        if (m == 1) {
            bool aWeight = A.opClass == OpClass::WeightPush;
            if (aWeight && (bWeight || isCompute(B)) && slotOf(a) == slotOf(b))
                consider(32, EdgeKind::Rule, mx + "weight push still active on w" + std::to_string(slotOf(a)));
            if (isCompute(A) && bWeight && slotOf(a) == slotOf(b))
                consider(32, EdgeKind::Rule, mx + "matmul still reading w" + std::to_string(slotOf(a)));
            if (isAccPush(A) && (isCompute(B) || B.opClass == OpClass::AccPushFp8) && accOf(a) == accOf(b))
                consider(33, EdgeKind::Rule, mx + "accumulator push still active on acc" + std::to_string(accOf(a)));
        }
    }
    return best;
}
