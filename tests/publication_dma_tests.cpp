#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "passes/pass.h"

static int failures = 0, checks = 0;
static const std::vector<std::string> manualWaitPasses = {"strip-artifacts", "fill-delay-slots", "schedule"};
#define CHECK(condition) do { ++checks; if (!(condition)) { ++failures; std::printf("FAIL line %d: %s\n", __LINE__, #condition); } } while (0)

// Include unprintable metadata so rejection checks cover all mutable state.
static std::string snapshot(const Code& code) {
    std::ostringstream out;
    auto instruction = [&](const Instr& in) {
        out << in.op->name << ':' << in.rd << ':' << in.rs1 << ':' << in.rs2 << ':' << in.imm
            << ':' << in.immText << ':' << in.target << ':' << in.comment << ':' << in.keep
            << ':' << in.release << ':' << in.line << '\n';
    };
    for (const Block& block : code.blocks) {
        out << "block:" << block.scheduled << ':' << block.terminatorCycle << ':' << block.endCycle
            << ':' << block.unknownSuccs << '\n';
        for (const auto& label : block.labels) out << "label:" << label << '\n';
        for (int cycle : block.issue) out << "issue:" << cycle << '\n';
        for (int successor : block.succs) out << "successor:" << successor << '\n';
        for (const Instr& in : block.body) instruction(in);
        out << "terminator:" << (bool)block.terminator << '\n';
        if (block.terminator) instruction(*block.terminator);
        out << "slot:" << (bool)block.slot << '\n';
        if (block.slot) instruction(*block.slot);
    }
    for (const auto& label : code.endLabels) out << "end:" << label << '\n';
    return out.str();
}

static void rejectCode(Code code, const std::string& reason, const std::vector<std::string>& passes = manualWaitPasses) {
    std::string before = snapshot(code);
    PassContext ctx;
    ctx.log.push_back("existing log");
    bool rejected = false;
    try { runPasses(code, passes, ctx); }
    catch (const std::runtime_error& error) {
        rejected = true;
        CHECK(std::string(error.what()).find(reason) != std::string::npos);
    }
    CHECK(rejected);
    CHECK(snapshot(code) == before);
    CHECK(ctx.log == std::vector<std::string>{"existing log"});
}

static void reject(const std::string& source, const std::string& reason,
                   const std::vector<std::string>& passes = manualWaitPasses) {
    rejectCode(buildBlocks(parseAsm(source)), reason, passes);
}

static void acceptCode(Code code, const std::vector<std::string>& passes = {}) {
    PassContext ctx;
    try {
        runPasses(code, passes, ctx);
        CHECK(!ctx.log.empty());
    } catch (const std::exception& error) {
        std::printf("Unexpected rejection: %s\n", error.what());
        CHECK(false);
    }
}

static void accept(const std::string& source, const std::vector<std::string>& passes = {}) {
    acceptCode(buildBlocks(parseAsm(source)), passes);
}

int main() {
    const std::string release = "csrrwi x0, x1, 0xC10 # atlas.release\n";
    accept(release);  // idle entry
    accept("dma.config.ch0 x0\ndma.wait.ch0\n" + release);
    accept("dma.config.ch0 x0\ndma.wait.ch0\n" + release, {"schedule"});
    reject("delay 8\ndma.config.ch0 x0\n" + release, "pending DMA on ch0");
    reject("dma.load.ch2 x0, x0, x0\n" + release, "pending DMA on ch2");
    reject("dma.store.ch7 x0, x0, x0\n" + release, "pending DMA on ch7");
    reject("dma.config.ch2 x0\ndma.wait.ch1\n" + release, "pending DMA on ch2");
    reject("dma.config.ch0 x0\ndma.wait.ch0\ndma.config.ch0 x0\n" + release, "pending DMA on ch0");
    reject("dma.config.ch0 x0\ndma.config.ch7 x0\n" + release, "ch0, ch7");
    reject("dma.config.ch0 x0\ndelay 4095 # keep\n" + release, "pending DMA on ch0");

    // Wait before channel reuse, not just before release.
    reject("dma.config.ch0 x0\ndma.config.ch0 x0\ndma.wait.ch0\n" + release, "before channel reuse");
    reject("dma.config.ch0 x0\ndelay 100 # keep\ndma.config.ch0 x0\ndma.wait.ch0\n" + release,
           "before channel reuse");
    reject("dma.config.ch0 x0\ndma.wait.ch1\ndma.config.ch0 x0\ndma.wait.ch0\n" + release,
           "before channel reuse");
    reject("dma.load.ch0 x0, x0, x0\ndelay 100 # keep\ndma.store.ch0 x0, x0, x0\ndma.wait.ch0\n" + release,
           "before channel reuse");
    reject("beq x1, x0, idle\nnop\ndma.config.ch0 x0\njal x0, join\nnop\n"
           "idle:\naddi x2, x0, 0\njoin:\ndma.config.ch0 x0\ndma.wait.ch0\n" + release,
           "before channel reuse");
    reject("addi x10, x0, 0\naddi x11, x0, 2\nloop:\ndma.config.ch0 x0\ndelay 100 # keep\n"
           "addi x10, x10, 1\nblt x10, x11, loop\nnop\ndma.wait.ch0\n" + release,
           "before channel reuse");
    reject("dma.config.ch0 x0\njal x0, join\ndma.config.ch0 x0\njoin:\ndma.wait.ch0\n" + release,
           "before channel reuse");
    accept("dma.config.ch0 x0\ndma.wait.ch0\ndma.config.ch0 x0\ndma.wait.ch0\n" + release);
    accept("dma.config.ch0 x0\ndma.config.ch1 x0\ndma.wait.ch0\ndma.wait.ch1\n" + release);

    // Merge pending channels from either predecessor.
    std::string join = "beq x1, x0, idle\nnop\ndma.config.ch2 x0\njal x0, join\nnop\n"
                       "idle:\naddi x2, x0, 0\njoin:\n";
    reject(join + release, "pending DMA on ch2");
    accept(join + "dma.wait.ch2\n" + release);
    accept("dma.config.ch0 x0\nbeq x1, x0, left\nnop\ndma.wait.ch0\njal x0, join\nnop\n"
           "left:\ndma.wait.ch0\njoin:\n" + release);

    // Entry backedges can carry pending DMA.
    reject("entry:\n" + release + "dma.config.ch0 x0\njal x0, entry\nnop\n", "pending DMA on ch0");
    accept("entry:\ndma.wait.ch0\n" + release + "dma.config.ch0 x0\njal x0, entry\nnop\n");
    reject("jal x0, loop\nnop\nloop:\n" + release +
           "dma.config.ch1 x0\njal x0, loop\nnop\n", "pending DMA on ch1");

    // Unreachable commands and releases do not affect joins.
    accept("jal x0, live\nnop\ndead:\ndma.config.ch0 x0\nlive:\n" + release);
    accept("jal x0, done\nnop\ndead:\ndma.config.ch0 x0\n" + release + "done:\naddi x2, x0, 1\n");

    // Reject unknown or invalid reachable edges.
    Code unknown = buildBlocks(parseAsm(release));
    unknown.blocks[0].unknownSuccs = true;
    rejectCode(unknown, "known control-flow successors");
    Code invalid = buildBlocks(parseAsm(release));
    invalid.blocks[0].succs.push_back(99);
    rejectCode(invalid, "invalid control-flow successor");
    Code deadUnknown = buildBlocks(parseAsm("jal x0, live\nnop\ndead:\naddi x2, x0, 0\nlive:\n" + release));
    deadUnknown.blocks[1].unknownSuccs = true;
    acceptCode(deadUnknown);

    // Scan programmatic terminators in execution order.
    Code dmaTerminator = buildBlocks(parseAsm("dma.config.ch5 x0\npublish:\n" + release));
    dmaTerminator.blocks[0].terminator = dmaTerminator.blocks[0].body.back();
    dmaTerminator.blocks[0].body.pop_back();
    rejectCode(dmaTerminator, "pending DMA on ch5");

    // Slot effects apply to both branch paths.
    reject("jal x0, join\ndma.config.ch0 x0\njoin:\n" + release, "pending DMA on ch0");
    accept("dma.config.ch0 x0\njal x0, join\ndma.wait.ch0\njoin:\n" + release);
    accept("dma.config.ch0 x0\nbeq x1, x0, join\ndma.wait.ch0\naddi x2, x0, 0\njoin:\n" + release);
    reject("dma.config.ch0 x0\njal x0, join\ndma.wait.ch1\njoin:\n" + release, "pending DMA on ch0");
    accept(release + "jal x0, done\ndma.config.ch1 x0\ndone:\naddi x2, x0, 1\n");
    reject(release + "jal x0, done\ndma.config.ch1 x0\ndone:\n" + release, "pending DMA on ch1");

    // Reject before passes mutate code or logs.
    reject("delay 8\n" + release, "requires the schedule pass", {"strip-artifacts"});
    reject(release, "requires the schedule pass", {"fill-delay-slots"});
    reject("delay 8\njal x0, done\n" + release + "done:\nnop\n", "release in a delay slot");
    reject("beq x0, x1, done\n" + release + "done:\nnop\n", "release in a delay slot");
    reject("jal x0, done\nnop\ndead:\njal x0, done\n" + release + "done:\nnop\n", "release in a delay slot");
    Code malformed = buildBlocks(parseAsm("delay 8\naddi x2, x0, 1\n"));
    malformed.blocks[0].body.back().release = true;
    rejectCode(malformed, "only supported on CSR");

    Code direct = buildBlocks(parseAsm("delay 8\njal x0, done\n" + release + "done:\nnop\n"));
    std::string before = snapshot(direct);
    PassContext ctx;
    bool rejected = false;
    try { stripArtifacts(direct, ctx); }
    catch (const std::runtime_error& error) {
        rejected = true;
        CHECK(std::string(error.what()).find("release in a delay slot") != std::string::npos);
    }
    CHECK(rejected);
    CHECK(snapshot(direct) == before);
    CHECK(ctx.log.empty());

    // Unmarked code retains partial-pass support.
    accept("dma.config.ch0 x0\ncsrrwi x0, x1, 0xC10\n", {"strip-artifacts"});
    accept("dma.config.ch0 x0\ndelay 100 # keep\ndma.config.ch0 x0\ndma.wait.ch0\n"
           "csrrwi x0, x1, 0xC10\n", {"strip-artifacts"});
    std::printf("publication DMA preflight: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
