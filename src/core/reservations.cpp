#include "core/reservations.h"

static int unitKey(Unit u, int index) { return (int)u * 64 + index; }

bool ReservationTable::unitFree(Unit u, int index, int from, int to) const {
    auto it = units_.find(unitKey(u, index));
    if (it == units_.end()) return true;
    int cap = unitCapacity(u, index);
    for (auto c = it->second.lower_bound(from); c != it->second.end() && c->first <= to; ++c)
        if (c->second >= cap) return false;
    return true;
}

int ReservationTable::chooseIndex(const Hold& h, int cycle) const {
    if (unitFree(h.unit, h.index, cycle + h.from, cycle + h.to)) return h.index;
    if (h.alt >= 0 && unitFree(h.unit, h.alt, cycle + h.from, cycle + h.to)) return h.alt;
    return -1;
}

std::vector<ReservationTable::PortRequest> ReservationTable::portRequests(const Instr& in, const Footprint& f,
                                                                          int cycle) const {
    std::vector<PortRequest> out;
    for (const Access& a : f.accesses) {
        if (a.res != Res::MReg) continue;
        for (int i = 0; i < a.count; i++) {
            int element = a.first + i;
            int reg = element / 32, row = element % 32;
            bool shareable = !a.write && in.op->engine == Engine::Vpu;
            out.push_back({(reg % 32) * 2 + (a.write ? 1 : 0), cycle + a.age + i * a.step, reg, row, shareable});
        }
    }
    return out;
}

std::string ReservationTable::conflict(const Instr& in, const Footprint& f, int cycle) const {
    for (const Hold& h : f.holds)
        if (chooseIndex(h, cycle) < 0)
            return std::string(unitName(h.unit)) + " " + std::to_string(h.index) + " busy";

    if (f.vpuLive > 0) {
        auto it = vpu_.find(cycle);
        if (it != vpu_.end()) {
            if (it->second.size() >= 2) return "both VPU slots busy";
            for (const OpInfo* other : it->second)
                if (!vpuCanOverlap(*other, *in.op)) return "VPU busy with " + other->name;
        }
    }

    std::vector<PortRequest> requests = portRequests(in, f, cycle);
    for (size_t i = 0; i < requests.size(); i++) {
        const PortRequest& p = requests[i];
        int bank = p.key / 2;
        auto shares = [&](const PortUse& u) {
            return u.shareable && p.shareable && u.reg == p.reg && u.row == p.row;
        };
        auto same = ports_.find(p.key);
        if (same != ports_.end()) {
            auto u = same->second.find(p.cycle);
            if (u != same->second.end() && !shares(u->second))
                return "MREG bank " + std::to_string(bank) + " port busy (m" + std::to_string(p.reg) + ")";
        }
        auto other = ports_.find(p.key ^ 1);
        if (other != ports_.end()) {
            auto u = other->second.find(p.cycle);
            if (u != other->second.end() && u->second.reg == p.reg && u->second.row == p.row)
                return "same-row read/write on m" + std::to_string(p.reg);
        }
        for (size_t j = 0; j < i; j++) {  // the instruction's own port uses must also fit together
            const PortRequest& q = requests[j];
            if (q.cycle != p.cycle) continue;
            if (q.key == p.key && !(q.shareable && p.shareable && q.reg == p.reg && q.row == p.row))
                return "instruction needs MREG bank " + std::to_string(bank) + " twice in one cycle";
            if ((q.key ^ 1) == p.key && q.reg == p.reg && q.row == p.row)
                return "instruction reads and writes the same MREG row in one cycle";
        }
    }
    return "";
}

void ReservationTable::reserve(const Instr& in, const Footprint& f, int cycle) {
    for (const Hold& h : f.holds) {
        int index = chooseIndex(h, cycle);
        if (index < 0) index = h.index;  // caller ignored conflict(); record it anyway
        int key = unitKey(h.unit, index);
        for (int c = cycle + h.from; c <= cycle + h.to; c++) units_[key][c]++;
        unitWindows_.push_back({key, cycle + h.from, cycle + h.to});
    }
    for (int age = 0; age < f.vpuLive; age++) vpu_[cycle + age].push_back(in.op);

    std::map<int, PortWindow> windows;  // per port key: first and last cycle used
    for (const PortRequest& p : portRequests(in, f, cycle)) {
        ports_[p.key][p.cycle] = {p.reg, p.row, p.shareable};
        auto it = windows.find(p.key);
        if (it == windows.end()) windows[p.key] = {p.key, p.cycle, p.cycle};
        else it->second.from = std::min(it->second.from, p.cycle), it->second.to = std::max(it->second.to, p.cycle);
    }
    for (auto& [key, w] : windows) portWindows_.push_back(w);
}

void ReservationTable::extendForWait(int cycle) {
    for (UnitWindow& w : unitWindows_) {
        if (w.to < cycle || w.from <= cycle) continue;
        for (int c = cycle; c < w.from; c++) units_[w.key][c]++;
        w.from = cycle;
    }
    for (PortWindow& w : portWindows_) {
        if (w.to < cycle || w.from <= cycle) continue;
        for (int c = cycle; c < w.from; c++)
            if (!ports_[w.key].count(c)) ports_[w.key][c] = PortUse{};
        w.from = cycle;
    }
}

void ReservationTable::forgetBefore(int cycle) {
    for (auto& [key, byCycle] : units_) byCycle.erase(byCycle.begin(), byCycle.lower_bound(cycle));
    for (auto& [key, byCycle] : ports_) byCycle.erase(byCycle.begin(), byCycle.lower_bound(cycle));
    vpu_.erase(vpu_.begin(), vpu_.lower_bound(cycle));
    std::erase_if(unitWindows_, [&](const UnitWindow& w) { return w.to < cycle; });
    std::erase_if(portWindows_, [&](const PortWindow& w) { return w.to < cycle; });
}
