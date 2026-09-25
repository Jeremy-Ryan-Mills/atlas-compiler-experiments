#include "core/blocks.h"

#include <set>

bool hasDelaySlot(const Block& b) {
    return b.terminator && isControlFlow(*b.terminator->op);
}

int naturalGap(const Instr& in) {
    return in.op->opClass == OpClass::Delay ? 1 + (int)(in.imm & 0xFFF) : 1;
}

std::vector<Instr> blockInstructions(const Block& b) {
    std::vector<Instr> out = b.body;
    if (b.terminator) out.push_back(*b.terminator);
    if (hasDelaySlot(b)) out.push_back(b.slot ? *b.slot : makeNop());
    return out;
}

std::vector<int> asWrittenCycles(const std::vector<Instr>& instrs) {
    std::vector<int> cycles;
    int c = 0;
    for (const Instr& in : instrs) {
        cycles.push_back(c);
        c += naturalGap(in);
    }
    return cycles;
}

static int blockIndexOfLabel(const Code& code, const std::string& label) {
    for (size_t i = 0; i < code.blocks.size(); i++)
        for (const std::string& l : code.blocks[i].labels)
            if (l == label) return (int)i;
    return -1;
}

Code buildBlocks(const AsmProgram& prog) {
    int n = (int)prog.instrs.size();
    std::set<int> leaders = {0};
    for (int i = 0; i < n; i++) {
        const OpInfo& op = *prog.instrs[i].op;
        if (!prog.labels[i].empty()) leaders.insert(i);
        if (isControlFlow(op)) {
            if (i + 1 < n && !prog.labels[i + 1].empty())
                throw ParseError("line " + std::to_string(prog.instrs[i + 1].line) +
                                 ": a label on a delay-slot instruction is not supported");
            if (i + 1 < n && isControlFlow(*prog.instrs[i + 1].op))
                throw ParseError("line " + std::to_string(prog.instrs[i + 1].line) +
                                 ": branch or jump in a delay slot is illegal");
            leaders.insert(i + 2);
        }
        if (op.opClass == OpClass::Halt) leaders.insert(i + 1);
    }

    Code code;
    std::vector<int> starts(leaders.begin(), leaders.end());
    for (size_t k = 0; k < starts.size() && starts[k] < n; k++) {
        int begin = starts[k];
        int end = k + 1 < starts.size() ? std::min(starts[k + 1], n) : n;
        Block b;
        b.labels = prog.labels[begin];
        for (int i = begin; i < end; i++) {
            const Instr& in = prog.instrs[i];
            if (isControlFlow(*in.op)) {
                b.terminator = in;
                if (i + 1 < end) b.slot = prog.instrs[i + 1];
                break;
            }
            if (in.op->opClass == OpClass::Halt) {
                b.terminator = in;
                break;
            }
            b.body.push_back(in);
        }
        code.blocks.push_back(b);
    }
    code.endLabels = prog.labels[n];

    for (size_t i = 0; i < code.blocks.size(); i++) {
        Block& b = code.blocks[i];
        int next = i + 1 < code.blocks.size() ? (int)i + 1 : -1;
        auto addTarget = [&](const std::string& label) {
            int t = blockIndexOfLabel(code, label);
            if (t >= 0) b.succs.push_back(t);  // a label after the last instruction ends the program
        };
        if (!b.terminator) {
            if (next >= 0) b.succs.push_back(next);
        } else if (b.terminator->op->opClass == OpClass::Branch) {
            addTarget(b.terminator->target);
            if (next >= 0) b.succs.push_back(next);
        } else if (b.terminator->op->name == "jal") {
            addTarget(b.terminator->target);
        } else if (b.terminator->op->name == "jalr") {
            b.unknownSuccs = true;
        }
    }
    return code;
}

// Emits `delay` instructions covering `cycles` idle issue cycles.
static void emitIdle(std::vector<Instr>& out, int cycles) {
    while (cycles > 0) {
        int n = std::min(cycles, 4096);  // delay N covers N+1 cycles, and N is 12 bits
        out.push_back(makeDelay(n - 1));
        cycles -= n;
    }
}

AsmProgram flatten(const Code& code) {
    AsmProgram prog;
    std::vector<std::vector<std::string>> labelsAt;
    bool scheduled = false;
    for (size_t bi = 0; bi < code.blocks.size(); bi++) {
        const Block& b = code.blocks[bi];
        scheduled |= b.scheduled;
        size_t first = prog.instrs.size();
        labelsAt.resize(first + 1);
        for (const std::string& l : b.labels) labelsAt[first].push_back(l);

        if (!b.scheduled) {
            for (const Instr& in : b.body) prog.instrs.push_back(in);
            if (b.terminator) prog.instrs.push_back(*b.terminator);
            if (hasDelaySlot(b)) prog.instrs.push_back(b.slot ? *b.slot : makeNop());
            labelsAt.resize(prog.instrs.size() + 1);
            continue;
        }

        int nextFree = 0;  // earliest cycle the next instruction could issue without a delay
        auto place = [&](const Instr& in, int cycle) {
            emitIdle(prog.instrs, cycle - nextFree);
            prog.instrs.push_back(in);
            nextFree = cycle + naturalGap(in);
        };
        for (size_t i = 0; i < b.body.size(); i++) place(b.body[i], b.issue[i]);
        if (b.terminator) {
            place(*b.terminator, b.terminatorCycle);
            if (hasDelaySlot(b)) {
                Instr slot = b.slot ? *b.slot : makeNop();
                if (!b.slot) slot.comment = "delay slot";
                place(slot, b.terminatorCycle + 1);
            }
        } else if (bi + 1 < code.blocks.size()) {
            emitIdle(prog.instrs, b.endCycle - nextFree);  // let this block drain before the next one starts
        }
        labelsAt.resize(prog.instrs.size() + 1);
    }
    labelsAt.resize(prog.instrs.size() + 1);
    for (const std::string& l : code.endLabels) labelsAt.back().push_back(l);
    prog.labels = labelsAt;

    // Halt bypasses delay stalls; guard it with a NOP, preserving kept delays.
    // Join blocks first to cover labeled fallthrough halts.
    if (scheduled) {
        AsmProgram guarded;
        for (size_t i = 0; i < prog.instrs.size(); i++) {
            Instr in = prog.instrs[i];
            guarded.labels.push_back(prog.labels[i]);
            if (in.op->opClass == OpClass::Delay && i + 1 < prog.instrs.size() &&
                prog.instrs[i + 1].op->opClass == OpClass::Halt) {
                int idle = naturalGap(in);
                if (idle > 1 || in.keep) {
                    if (!in.keep) {
                        in.imm = idle - 2;
                        in.immText.clear();
                    }
                    guarded.instrs.push_back(in);
                    guarded.labels.emplace_back();
                }
                Instr guard = makeNop();
                guard.line = in.line;
                guard.comment = "halt guard";
                guarded.instrs.push_back(guard);
            } else {
                guarded.instrs.push_back(in);
            }
        }
        guarded.labels.push_back(prog.labels.back());
        return guarded;
    }
    return prog;
}
