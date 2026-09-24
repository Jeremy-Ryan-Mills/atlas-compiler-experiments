#pragma once

#include <string>
#include <vector>

#include "core/asm.h"
#include "core/values.h"

// Timing model of the Atlas core, following npu_model's rtl-match branch.
// "Age" is the number of cycles since an instruction issued (age 0 = issue cycle).

const int kVmemBytes = 1536 * 1024;
const int kVmemBankBytes = 256 * 1024;
const int kVmemBanks = kVmemBytes / kVmemBankBytes;
const int kLineBytes = 32;

// Storage an instruction reads or writes.
enum class Res { XReg, EReg, MReg, Acc, Weight, Vmem, DmaBase };

// Elements [first, first + count) of one kind of storage. Element i is touched at
// age + i * step. Elements are: register numbers for XReg/EReg, reg*32+row for MReg,
// (mxu*2+index)*32+row for Acc/Weight, and 32-byte line numbers for Vmem.
struct Access {
    Res res;
    bool write;
    int first, count;
    int age, step;
    bool anywhere = false;      // address unknown: may touch any element
    bool atCompletion = false;  // DMA: happens when the transfer finishes (time unknown)
    int lastAge() const { return age + (count - 1) * step; }
};

// Hardware structures that only one (or a few) instructions may use per cycle.
enum class Unit {
    ScalarLoad,       // scalar load command/response path
    ScalarWriteback,  // scalar register write port shared by S1 and load responses
    VloadPath, VstorePath,
    VmemBank,         // index = VMEM bank (0..5), one LSU access per cycle
    Xlu,
    MxuPort,          // index = mxu*4 + port (0,1 read ports, 2,3 write ports)
    MxuCompute,       // index = mxu, in-flight matmul tracker (3 on MXU0, 2 on MXU1)
    MxuAccRead,       // index = mxu*2 + acc
    MxuAccWrite,      // index = mxu*2 + acc
    MxuWeightStream,  // index = mxu, one weight push writing per cycle
    MxuAccStream,     // index = mxu, one accumulator push writing per cycle
};

// A unit held from age `from` to age `to` (inclusive). If `alt` >= 0 the
// instruction may use unit index `alt` instead when `index` is busy.
struct Hold {
    Unit unit;
    int index;
    int from, to;
    int alt = -1;
};

// Everything the scheduler needs to know about one instruction.
struct Footprint {
    std::vector<Access> accesses;
    std::vector<Hold> holds;
    std::vector<int> mregReads, mregWrites;  // logical MREG reservations taken at issue
    int readRelease = 0, writeRelease = 0;   // ages at which those reservations are dropped
    bool writeDuringRead = false;            // vload may write registers others are still reading
    int vpuLive = 0;                          // VPU: occupies a VPU slot for ages 0 .. vpuLive-1
    int doneAge = 0;                          // last age at which the instruction uses any resource
    int dmaCycles = 0;                        // DMA commands: expected transfer time (npu_model's estimate)
    std::string error;                        // set when the operands are illegal
};

Footprint footprintOf(const Instr& in, const RegValues& regs);

enum class EdgeKind { RAW, WAR, WAW, Rule, Order };
const char* edgeKindName(EdgeKind k);

// Why b must wait for a, and for how many cycles.
struct Dependence {
    int distance = 0;  // b must issue at least this many cycles after a; 0 = independent
    EdgeKind kind = EdgeKind::Order;
    std::string reason;
};

// Minimum issue distance between a and b, where a comes first in program order.
// DMA accesses that happen at completion are handled by the graph builder instead.
Dependence dependence(const Instr& a, const Footprint& fa, const Instr& b, const Footprint& fb);

// Frontend barriers: nothing may move across them.
bool isBarrier(const Instr& in);

// VPU issue rule: can `b` issue while `a` still occupies a VPU slot?
bool vpuCanOverlap(const OpInfo& a, const OpInfo& b);
bool vpuUsesBothSlots(const OpInfo& op);

int unitCapacity(Unit u, int index);
const char* unitName(Unit u);

// Cycles the model's DMA engine needs for one transfer of `bytes` bytes.
int dmaTransferCycles(long long bytes);
