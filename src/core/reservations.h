#pragma once

#include <map>
#include <string>
#include <vector>

#include "core/machine.h"

// Cycle-by-cycle record of the hardware that issued instructions occupy: engine
// paths and ports (Hold), the 32 physical MREG banks (one read and one write port
// each; mN and m(N+32) share a bank), and the two VPU issue slots.
class ReservationTable {
public:
    // Returns "" if `in` can issue at `cycle`, otherwise the reason it cannot.
    std::string conflict(const Instr& in, const Footprint& f, int cycle) const;
    void reserve(const Instr& in, const Footprint& f, int cycle);

    // DMA stalls require reservations from the wait through each resource's last use.
    // Port rows become unknown.
    void extendForWait(int cycle);

    // Drops bookkeeping for cycles before `cycle` (used by long simulations).
    void forgetBefore(int cycle);

private:
    struct PortUse {
        int reg = -1, row = -1;  // -1: unknown (reserved by extendForWait)
        bool shareable = false;  // VPU reads of the same register row share one port
    };
    struct UnitWindow { int key, from, to; };
    struct PortWindow { int key, from, to; };

    std::map<int, std::map<int, int>> units_;       // unit key -> cycle -> users
    std::map<int, std::map<int, PortUse>> ports_;   // bank*2 + isWrite -> cycle -> user
    std::map<int, std::vector<const OpInfo*>> vpu_; // cycle -> VPU instructions holding a slot
    std::vector<UnitWindow> unitWindows_;
    std::vector<PortWindow> portWindows_;

    struct PortRequest { int key, cycle, reg, row; bool shareable; };
    std::vector<PortRequest> portRequests(const Instr& in, const Footprint& f, int cycle) const;
    bool unitFree(Unit u, int index, int from, int to) const;
    int chooseIndex(const Hold& h, int cycle) const;  // index to use, or -1 if busy
};
