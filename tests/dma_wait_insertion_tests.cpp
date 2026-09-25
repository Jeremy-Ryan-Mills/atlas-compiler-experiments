#include <algorithm>
#include <cstdio>
#include <string>

#include "core/simulator.h"
#include "passes/pass.h"

static int failures = 0, checks = 0;
#define CHECK(condition) do { ++checks; if (!(condition)) { ++failures; std::printf("FAIL line %d: %s\n", __LINE__, #condition); } } while (0)

static AsmProgram insert(const std::string& source) {
    Code code = buildBlocks(parseAsm(source));
    PassContext ctx;
    insertDmaWaits(code, ctx);
    std::string once = printAsm(flatten(code));
    insertDmaWaits(code, ctx);
    CHECK(printAsm(flatten(code)) == once);
    return flatten(code);
}

static AsmProgram optimize(const std::string& source) {
    Code code = buildBlocks(parseAsm(source));
    PassContext ctx;
    runPasses(code, {}, ctx);
    return flatten(code);
}

static int waits(const AsmProgram& program, int channel = -1) {
    return std::count_if(program.instrs.begin(), program.instrs.end(), [&](const Instr& in) {
        return in.op->opClass == OpClass::DmaWait && (channel < 0 || in.op->channel == channel);
    });
}

static int position(const AsmProgram& program, const std::string& op, int start = 0) {
    for (int i = start; i < (int)program.instrs.size(); i++)
        if (program.instrs[i].op->name == op) return i;
    return -1;
}

static void safe(const std::string& source) {
    AsmProgram result = optimize(source);
    for (double scale : {0.1, 1.0, 10.0, 100.0}) {
        SimOptions options;
        options.dmaLatencyScale = scale;
        SimResult run = simulate(result, options);
        if (!run.violations.empty()) std::printf("%s\n", run.violations.front().c_str());
        CHECK(run.violations.empty());
        CHECK(run.stopReason.empty());
    }
    CHECK(printAsm(optimize(printAsm(result))) == printAsm(result));
}

static void dependencies_and_overlap() {
    const std::string setup = "addi x7, x0, 32\nlui x1, 1\n";
    AsmProgram consumer = insert(setup + "dma.load.ch0 x1, x0, x7\naddi x8, x0, 9\nlw x2, 0(x1)\n");
    CHECK(waits(consumer) == 1);
    CHECK(position(consumer, "dma.wait.ch0") + 1 == position(consumer, "lw"));
    CHECK(consumer.instrs[position(consumer, "dma.wait.ch0") - 1].rd == 8);
    safe(printAsm(consumer));

    for (const std::string overwrite : {"addi x1, x0, 0\n", "addi x7, x0, 0\n", "csrrwi x1, x0, 0xC10\n"}) {
        AsmProgram result = insert(setup + "dma.load.ch0 x1, x0, x7\n" + overwrite);
        CHECK(waits(result) == 1);
        CHECK(position(result, "dma.wait.ch0") == position(result, "dma.load.ch0") + 1);
        safe(printAsm(result));
    }

    AsmProgram disjoint = insert(setup + "dma.load.ch0 x1, x0, x7\nvstore m0, 0(x0)\ndma.wait.ch0\n");
    CHECK(waits(disjoint) == 1);
    CHECK(position(disjoint, "vstore") < position(disjoint, "dma.wait.ch0"));

    for (const std::string command : {"dma.load.ch0 x1, x0, x7\n", "dma.store.ch0 x0, x1, x7\n"}) {
        AsmProgram conflicting = insert(setup + command + "sw x0, 0(x1)\ndma.wait.ch0\n");
        CHECK(waits(conflicting) == 2);
        CHECK(position(conflicting, "dma.wait.ch0") < position(conflicting, "sw"));
        safe(printAsm(conflicting));
    }

    AsmProgram unknown = insert("lw x1, 0(x0)\naddi x7, x0, 32\ndma.load.ch0 x1, x0, x7\nvstore m0, 0(x0)\n");
    CHECK(position(unknown, "dma.wait.ch0") < position(unknown, "vstore"));
}

static void channels_and_boundaries() {
    const std::string setup = "addi x7, x0, 32\nlui x1, 1\n";
    AsmProgram reuse = insert(setup + "dma.load.ch0 x1, x0, x7\ndma.store.ch0 x0, x1, x7\necall\n");
    CHECK(waits(reuse, 0) == 2);
    CHECK(position(reuse, "dma.wait.ch0") < position(reuse, "dma.store.ch0"));
    safe(printAsm(reuse));

    AsmProgram independent = insert(setup + "dma.load.ch0 x1, x0, x7\ndma.load.ch1 x0, x0, x7\ndma.wait.ch0\ndma.wait.ch1\n");
    CHECK(waits(independent) == 2);
    CHECK(position(independent, "dma.load.ch1") < position(independent, "dma.wait.ch0"));

    AsmProgram conflict = insert(setup + "dma.load.ch0 x1, x0, x7\ndma.store.ch1 x0, x1, x7\n");
    CHECK(position(conflict, "dma.wait.ch0") < position(conflict, "dma.store.ch1"));
    safe(printAsm(conflict));

    const std::string stores = setup + "addi x8, x0, 1024\ndma.store.ch0 x0, x1, x7\ndma.store.ch1 x8, x1, x7\n";
    AsmProgram sharedSource = insert(stores);
    CHECK(position(sharedSource, "dma.wait.ch0") < position(sharedSource, "dma.store.ch1"));
    safe(stores);

    for (const std::string boundary : {"", "ecall\n", "ebreak\n", "csrrwi x0, x1, 0xC10 # atlas.release\n"}) {
        AsmProgram result = insert(setup + "dma.load.ch0 x1, x0, x7\n" + boundary);
        CHECK(waits(result, 0) == 1);
        safe(printAsm(result));
    }

    AsmProgram ordinaryCsr = insert(setup + "dma.load.ch0 x1, x0, x7\ncsrrwi x0, x1, 0xC10\ndma.wait.ch0\n");
    CHECK(position(ordinaryCsr, "csrrwi") < position(ordinaryCsr, "dma.wait.ch0"));
    CHECK(waits(ordinaryCsr) == 1);
    AsmProgram config = insert("dma.config.ch0 x1\naddi x1, x0, 0\n");
    CHECK(position(config, "dma.wait.ch0") < position(config, "addi"));
}

static void control_flow() {
    const std::string release = "csrrwi x0, x1, 0xC10 # atlas.release\n";
    for (int branch : {0, 1}) {
        std::string join = "addi x8, x0, " + std::to_string(branch) + "\nbeq x8, x0, idle\nnop\n"
                           "dma.config.ch2 x0\njal x0, join\nnop\nidle:\naddi x2, x0, 0\njoin:\n";
        CHECK(waits(insert(join + release), 2) == 1);
        safe(join + release);
        safe("addi x7, x0, 32\nlui x1, 1\ndma.load.ch0 x1, x0, x7\n"
             "addi x8, x0, " + std::to_string(branch) + "\nbeq x8, x0, done\nnop\ndma.wait.ch0\ndone:\n");
    }
    safe("addi x10, x0, 0\naddi x11, x0, 3\nloop:\ndma.config.ch0 x0\n"
         "addi x10, x10, 1\nblt x10, x11, loop\nnop\n");
    safe("addi x7, x0, 32\nlui x1, 1\ndma.load.ch0 x1, x0, x7\n"
         "beq x1, x0, done\naddi x1, x0, 0\ndone:\n");
    safe("jal x0, done\ndma.config.ch0 x0\ndone:\n");
    safe("dma.config.ch0 x0\njal x0, done\ndma.wait.ch0\ndone:\n");
    CHECK(waits(insert("jal x0, live\nnop\ndead:\ndma.config.ch0 x0\nlive:\n" + release)) == 0);
}

static void preserve_explicit_waits() {
    for (const std::string source : {
             "addi x7, x0, 32\nlui x1, 1\ndma.load.ch0 x1, x0, x7\ndma.wait.ch0\nlw x2, 0(x1)\n",
             "dma.config.ch0 x0\ndma.wait.ch0\ncsrrwi x0, x1, 0xC10 # atlas.release\n",
             "dma.config.ch0 x0\nbeq x1, x0, left\nnop\ndma.wait.ch0\njal x0, end\nnop\nleft:\ndma.wait.ch0\nend:\n"}) {
        CHECK(printAsm(insert(source)) == printAsm(flatten(buildBlocks(parseAsm(source)))));
        safe(source);
    }
}

static void reject(const std::string& source, const std::vector<std::string>& passes, const std::string& reason) {
    Code code = buildBlocks(parseAsm(source));
    std::string before = printAsm(flatten(code));
    PassContext ctx;
    ctx.log.push_back("existing log");
    bool rejected = false;
    try { runPasses(code, passes, ctx); }
    catch (const std::runtime_error& error) {
        rejected = true;
        CHECK(std::string(error.what()).find(reason) != std::string::npos);
    }
    CHECK(rejected);
    CHECK(printAsm(flatten(code)) == before);
    CHECK(ctx.log == std::vector<std::string>{"existing log"});
}

static void preflight() {
    reject("dma.config.ch0 x0\n", {"insert-dma-waits"}, "schedule");
    reject("delay 8\nauipc x1, 0\n", {}, "auipc");
    reject("jal x0, end\ndma.config.ch0 x0\nend:\n", {"insert-dma-waits", "schedule"}, "strip-artifacts");
    reject("dma.config.ch0 x0\ncsrrwi x0, x1, 0xC10 # atlas.release\n",
           {"strip-artifacts", "fill-delay-slots", "schedule"}, "pending DMA");
}

int main() {
    dependencies_and_overlap();
    channels_and_boundaries();
    control_flow();
    preserve_explicit_waits();
    preflight();
    std::printf("DMA wait insertion: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
