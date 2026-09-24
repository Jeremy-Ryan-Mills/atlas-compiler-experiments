#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "core/simulator.h"
#include "passes/pass.h"
#include "tool/viewer.h"

static void usage() {
    std::cerr << "usage: atlas-opt [options] input.S\n"
                 "  -o FILE            write the optimized program to FILE (default: stdout)\n"
                 "  --viz FILE.html    write the before/after dependency graph viewer\n"
                 "  --passes a,b,c     run only these passes (default: all; see --list-passes)\n"
                 "  --list-passes      list the passes in the order they run\n"
                 "  --dma-timing MODE  robust (default): valid for any DMA latency;\n"
                 "                     model: trust npu_model's DMA latency\n"
                 "  --check            only simulate the input and report problems\n"
                 "  -q                 print nothing unless something is wrong\n";
}

static bool writeFile(const std::string& path, const std::string& text) {
    std::ofstream f(path);
    if (!f) std::cerr << "atlas-opt: cannot write " << path << "\n";
    f << text;
    return (bool)f;
}

static void printProblems(const char* what, const SimResult& r) {
    for (const std::string& v : r.violations) std::cerr << "  " << what << ": " << v << "\n";
    if (!r.stopReason.empty()) std::cerr << "  " << what << ": simulation stopped: " << r.stopReason << "\n";
}

int main(int argc, char** argv) {
    std::string input, output, vizPath;
    std::vector<std::string> passNames;
    PassContext ctx;
    bool checkOnly = false, quiet = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        bool hasValue = i + 1 < argc;
        if (a == "-o" && hasValue) output = argv[++i];
        else if (a == "--viz" && hasValue) vizPath = argv[++i];
        else if (a == "--passes" && hasValue) {
            std::stringstream list(argv[++i]);
            for (std::string name; std::getline(list, name, ',');) passNames.push_back(name);
        } else if (a == "--dma-timing" && hasValue) ctx.robustDma = std::string(argv[++i]) != "model";
        else if (a == "--list-passes") {
            for (const Pass& p : allPasses()) std::cout << p.name << "\t" << p.description << "\n";
            return 0;
        } else if (a == "--check") checkOnly = true;
        else if (a == "-q") quiet = true;
        else if (a[0] != '-' && input.empty()) input = a;
        else {
            usage();
            return 2;
        }
    }
    if (input.empty()) {
        usage();
        return 2;
    }

    try {
        AsmProgram original = readAsmFile(input);
        SimResult before = simulate(original);
        if (checkOnly) {
            std::cout << input << ": " << before.cycles << " cycles, " << before.issued << " instructions issued ("
                      << before.delays << " delays)\n";
            printProblems("input", before);
            return before.violations.empty() && before.stopReason.empty() ? 0 : 1;
        }

        Code code = buildBlocks(original);
        runPasses(code, passNames, ctx);
        AsmProgram optimized = flatten(code);

        // Check the result, and (for robust schedules) that it still holds when DMA is slower than modeled.
        SimResult after = simulate(optimized);
        SimOptions slowDma;
        slowDma.dmaLatencyScale = 1.7;
        SimResult afterSlow = ctx.robustDma ? simulate(optimized, slowDma) : after;

        if (!output.empty()) {
            if (!writeFile(output, printAsm(optimized))) return 1;
        } else if (vizPath.empty()) {
            std::cout << printAsm(optimized);
        }
        if (!vizPath.empty() && !writeFile(vizPath, renderHtml(buildProgramView(input, original, code, before, after, ctx.log))))
            return 1;

        bool bad = !after.violations.empty() || !after.stopReason.empty() || !afterSlow.violations.empty();
        if (!quiet || bad) {
            for (const std::string& line : ctx.log) std::cerr << "  " << line << "\n";
            char buf[512];
            std::snprintf(buf, sizeof buf, "%s: %lld -> %lld cycles (%.2fx), %lld -> %lld instructions issued\n",
                          input.c_str(), before.cycles, after.cycles, (double)before.cycles / std::max(1LL, after.cycles),
                          before.issued, after.issued);
            std::cerr << buf;
        }
        printProblems("output", after);
        if (ctx.robustDma) printProblems("output with slower DMA", afterSlow);
        return bad ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << "atlas-opt: " << e.what() << "\n";
        return 1;
    }
}
