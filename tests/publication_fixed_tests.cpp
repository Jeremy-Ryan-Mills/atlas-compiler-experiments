// Publication timing and annotation tests.
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "core/depgraph.h"
#include "core/simulator.h"
#include "passes/pass.h"

static int failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::printf("FAIL line %d: %s\n", __LINE__, #condition); failures++; } } while (0)

static bool throws(const std::function<void()>& fn) {
    try { fn(); } catch (const std::exception&) { return true; }
    return false;
}

static bool reports(const SimResult& result, const std::string& text) {
    for (const auto& violation : result.violations)
        if (violation.find(text) != std::string::npos) return true;
    return false;
}

static AsmProgram one(const Instr& in) { return {{in}, {{}, {}}}; }

static AsmProgram optimize(const std::string& text) {
    Code code = buildBlocks(parseAsm(text));
    PassContext ctx;
    runPasses(code, {}, ctx);
    return flatten(code);
}

static const std::string release = "csrrwi x0, x1, 0xC10 # atlas.release\n";

static void annotation_round_trip_and_validation() {
    const std::string marked = "csrrwi x0, x1, 0xC10 # before\tatlas.release after\n";
    AsmProgram parsed = parseAsm(marked);
    CHECK(parsed.instrs[0].release);
    CHECK(parsed.instrs[0].comment == "before\tatlas.release after");
    CHECK(parseAsm(printAsm(parsed)).instrs[0].release);
    CHECK(printAsm(parseAsm(printAsm(parsed))) == printAsm(parsed));
    for (const std::string token : {"atlas.releaseX", "xatlas.release", "atlas.release,", "ATLAS.RELEASE"}) {
        AsmProgram ordinary = parseAsm("csrrwi x0, x1, 0xC10 # " + token + "\n");
        CHECK(!ordinary.instrs[0].release);
        CHECK(!parseAsm(printAsm(ordinary)).instrs[0].release);
    }
    CHECK(!parseAsm("csrrwi x0, x1, 0xC10\n").instrs[0].release);
    CHECK(parseAsm("csrrwi x0, x2, 0xC11 # atlas.release\n").instrs[0].release);

    Instr generated = makeInstr("csrrwi", 0, 1, 0, 0xC10);
    generated.release = true;
    CHECK(parseAsm(printAsm(one(generated))).instrs[0].release);
    generated.comment = "handoff note";
    AsmProgram reparsed = parseAsm(printAsm(one(generated)));
    CHECK(reparsed.instrs[0].release);
    CHECK(reparsed.instrs[0].comment == "handoff note atlas.release");
    generated.comment = "handoff atlas.release note";
    CHECK(parseAsm(printAsm(one(generated))).instrs[0].comment == generated.comment);
    generated.release = false;
    CHECK(throws([&] { printAsm(one(generated)); }));

    for (const std::string instruction : {"vstore m0, 0(x0)", "delay 3", "fence", "nop", "li x1, 100000", ""})
        CHECK(throws([&] { parseAsm(instruction + " # atlas.release\n"); }));

    Instr invalid = makeInstr("vstore");
    invalid.release = true;
    CHECK(!footprintOf(invalid, zeroRegs()).error.empty());
    CHECK(throws([&] { printAsm(one(invalid)); }));
    SimResult checked = simulate(one(invalid));
    CHECK(reports(checked, "only valid on a CSR"));
    CHECK(checked.issued == 0);
    CHECK(!checked.stopReason.empty());
}

static void completion_distances_and_outgoing_order() {
    Instr publication = parseAsm(release).instrs[0];
    Footprint pub = footprintOf(publication, zeroRegs());
    for (const std::string text : {
             "addi x1, x0, 1", "sw x1, 0(x0)", "lw x2, 0(x0)", "seld e0, 0(x0)",
             "vload m0, 0(x0)", "vstore m0, 0(x0)", "vmov m2, m0", "vtrpose.xlu m2, m0",
             "vmatmul.mxu0 acc0, m0, w0", "vmatmul.mxu1 acc0, m0, w0"}) {
        Instr prior = parseAsm(text + "\n").instrs[0];
        Footprint f = footprintOf(prior, zeroRegs());
        CHECK(f.error.empty());
        CHECK(dependence(prior, f, publication, pub).distance >= f.doneAge + 1);
        CHECK(dependence(publication, pub, prior, f).distance >= 1);
        AsmProgram scheduled = optimize(text + "\n" + release);
        CHECK(simulate(scheduled).violations.empty());
        CHECK(printAsm(optimize(printAsm(scheduled))) == printAsm(scheduled));
    }
    AsmProgram plain = parseAsm("vstore m0, 0(x0)\ncsrrwi x0, x1, 0xC10\n");
    CHECK(dependence(plain.instrs[0], footprintOf(plain.instrs[0], zeroRegs()),
                     plain.instrs[1], footprintOf(plain.instrs[1], zeroRegs())).distance == 1);
    CHECK(simulate(plain).violations.empty());  // Unmarked behavior is unchanged.
    CHECK(reports(simulate(parseAsm("vstore m0, 0(x0)\n" + release)), "atlas.release publishes before"));
}

static void checker_rejects_same_tick_completion() {
    // VSTORE completes at issue+34, after same-cycle CSR execution.
    CHECK(reports(simulate(parseAsm("vstore m0, 0(x0)\ndelay 32\n" + release)), "atlas.release publishes before"));
    CHECK(simulate(parseAsm("vstore m0, 0(x0)\ndelay 33\n" + release)).violations.empty());
    CHECK(reports(simulate(parseAsm("lw x2, 0(x0)\ndelay 1\n" + release)), "atlas.release publishes before"));
    CHECK(simulate(parseAsm("lw x2, 0(x0)\ndelay 2\n" + release)).violations.empty());
    CHECK(reports(simulate(parseAsm("sw x1, 0(x0)\n" + release)), "atlas.release publishes before"));
    CHECK(simulate(parseAsm("sw x1, 0(x0)\nnop\n" + release)).violations.empty());

    // The checker uses modeled latency; preflight requires explicit DMA waits.
    CHECK(reports(simulate(parseAsm("dma.config.ch0 x0\ndelay 1\n" + release)), "atlas.release publishes before"));
    CHECK(simulate(parseAsm("dma.config.ch0 x0\ndelay 2\n" + release)).violations.empty());
    for (double scale : {1.0, 10.0, 100.0}) {
        SimOptions options;
        options.dmaLatencyScale = scale;
        CHECK(reports(simulate(parseAsm("dma.config.ch0 x0\n" + release), options), "atlas.release publishes before"));
        CHECK(simulate(parseAsm("dma.config.ch0 x0\ndma.wait.ch0\n" + release), options).violations.empty());
    }
}

static void checker_and_scheduler_cover_block_boundaries_and_slots() {
    for (const std::string prefix : {
             "vstore m0, 0(x0)\nexit:\n",
             "vstore m0, 0(x0)\njal x0, exit\nnop\nexit:\n",
             "vstore m0, 0(x0)\nbeq x0, x0, exit\nnop\nexit:\n"}) {
        AsmProgram out = optimize(prefix + release);
        CHECK(simulate(out).violations.empty());
        CHECK(printAsm(optimize(printAsm(out))) == printAsm(out));
    }
    // Reject release slots on both branch paths.
    for (const std::string branch : {"beq x0, x0, exit", "bne x0, x0, exit", "jal x0, exit"}) {
        AsmProgram in = parseAsm(branch + "\n" + release + "exit:\nnop\n");
        SimResult result = simulate(in);
        CHECK(reports(result, "atlas.release in a delay slot"));
        CHECK(!result.stopReason.empty());
        CHECK(result.issued == 0);
    }
}

int main() {
    const std::vector<std::pair<const char*, std::function<void()>>> tests = {
        {"annotation_round_trip_and_validation", annotation_round_trip_and_validation},
        {"completion_distances_and_outgoing_order", completion_distances_and_outgoing_order},
        {"checker_rejects_same_tick_completion", checker_rejects_same_tick_completion},
        {"checker_and_scheduler_cover_block_boundaries_and_slots", checker_and_scheduler_cover_block_boundaries_and_slots},
    };
    for (const auto& [name, test] : tests) {
        std::printf("%s\n", name);
        try { test(); } catch (const std::exception& error) {
            std::printf("FAIL unexpected exception: %s\n", error.what());
            failures++;
        }
    }
    std::printf("%d failures\n", failures);
    return failures != 0;
}
