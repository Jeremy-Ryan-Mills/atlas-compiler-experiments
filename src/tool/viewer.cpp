#include "tool/viewer.h"

#include <algorithm>
#include <climits>
#include <sstream>

static int criticalPathLength(const DepGraph& g) {
    std::vector<int> h = criticalHeights(g);
    int best = 0;
    for (int x : h) best = std::max(best, x);
    return best;
}

// Edges implied by a longer path through other nodes add no constraint, so the
// viewer hides them by default.
static std::vector<bool> redundantEdges(const DepGraph& g) {
    int n = (int)g.nodes.size();
    // longest[a][c]: longest path from a to c using one or more edges (INT_MIN if none).
    std::vector<std::vector<int>> longest(n, std::vector<int>(n, INT_MIN));
    for (int a = n - 1; a >= 0; a--)
        for (int e : g.out[a]) {
            const Edge& ed = g.edges[e];
            longest[a][ed.to] = std::max(longest[a][ed.to], ed.distance);
            for (int c = ed.to + 1; c < n; c++)
                if (longest[ed.to][c] != INT_MIN)
                    longest[a][c] = std::max(longest[a][c], ed.distance + longest[ed.to][c]);
        }

    std::vector<bool> redundant(g.edges.size(), false);
    for (int e = 0; e < (int)g.edges.size(); e++) {
        const Edge& ed = g.edges[e];
        for (int f : g.out[ed.from]) {
            const Edge& first = g.edges[f];
            if (first.to == ed.to || first.to > ed.to) continue;
            int rest = longest[first.to][ed.to];
            if (rest != INT_MIN && first.distance + rest >= ed.distance) {
                redundant[e] = true;
                break;
            }
        }
    }
    return redundant;
}

static bool isScheduleArtifact(const Instr& in) {
    return (in.op->opClass == OpClass::Delay && !in.keep) || isNop(in);
}

static GraphView makeView(const std::vector<Instr>& seq, const std::vector<int>& seqCycles,
                          const RegValues& entry, int length, uint32_t dmaRegs) {
    GraphView v;
    std::vector<Instr> nodes;
    for (size_t i = 0; i < seq.size(); i++) {
        if (isScheduleArtifact(seq[i])) continue;  // delays and fillers show up as gaps instead
        nodes.push_back(seq[i]);
        v.cycles.push_back(seqCycles[i]);
    }
    v.graph = buildGraph(nodes, entry, dmaRegs);
    v.redundant = redundantEdges(v.graph);
    v.length = length;
    return v;
}

ProgramView buildProgramView(const std::string& source, const AsmProgram& original, const Code& optimized,
                             const SimResult& before, const SimResult& after) {
    ProgramView view;
    view.source = source;
    view.before = before;
    view.after = after;
    Code orig = buildBlocks(original);
    std::vector<RegValues> entryOrig = blockEntryValues(orig);
    std::vector<RegValues> entryOpt = blockEntryValues(optimized);
    uint32_t dmaRegs = dmaOperandRegisters(original.instrs);

    // Passes may merge blocks (e.g. unroll-loops), so pair each optimized block with the
    // original blocks its instructions came from, found by source line.
    std::vector<int> blockOfLine;
    for (size_t bi = 0; bi < orig.blocks.size(); bi++)
        for (const Instr& in : blockInstructions(orig.blocks[bi])) {
            if (in.line >= (int)blockOfLine.size()) blockOfLine.resize(in.line + 1, -1);
            blockOfLine[in.line] = (int)bi;
        }

    for (size_t bi = 0; bi < optimized.blocks.size(); bi++) {
        const Block& nb = optimized.blocks[bi];
        std::vector<int> sources;
        for (const Instr& in : blockInstructions(nb))
            if (in.line > 0 && in.line < (int)blockOfLine.size() && blockOfLine[in.line] >= 0)
                sources.push_back(blockOfLine[in.line]);
        std::sort(sources.begin(), sources.end());
        sources.erase(std::unique(sources.begin(), sources.end()), sources.end());

        BlockView bv;
        int first = sources.empty() ? -1 : sources[0];
        const Block* ob = first >= 0 ? &orig.blocks[first] : nullptr;
        bv.name = ob && !ob->labels.empty() ? ob->labels[0] : first == 0 ? "entry" : "block " + std::to_string(bi);
        if (sources.size() > 1) bv.name += " (+" + std::to_string(sources.size() - 1) + " merged)";

        // Merged blocks show their original blocks back to back, each as written once.
        std::vector<Instr> seq;
        for (int s : sources)
            for (const Instr& in : blockInstructions(orig.blocks[s])) seq.push_back(in);
        std::vector<int> cycles = asWrittenCycles(seq);
        int length = seq.empty() ? 0 : cycles.back() + naturalGap(seq.back());
        bv.before = makeView(seq, cycles, first >= 0 ? entryOrig[first] : unknownRegs(), length, dmaRegs);
        bv.lowerBound = criticalPathLength(bv.before.graph);

        std::vector<Instr> seq2 = blockInstructions(nb);
        std::vector<int> cycles2 = asWrittenCycles(seq2);
        if (nb.scheduled) {
            cycles2 = nb.issue;
            if (nb.terminator) cycles2.push_back(nb.terminatorCycle);
            if (hasDelaySlot(nb)) cycles2.push_back(nb.terminatorCycle + 1);
        }
        length = nb.scheduled ? nb.endCycle : seq2.empty() ? 0 : cycles2.back() + naturalGap(seq2.back());
        bv.after = makeView(seq2, cycles2, entryOpt[bi], length, dmaRegs);
        view.blocks.push_back(bv);
    }
    return view;
}

// JSON string literal; '<' is escaped so the data can sit inside a <script> tag.
static std::string js(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '<': out += "\\u003c"; break;
            default:
                if ((unsigned char)c < 0x20) out += ' ';
                else out += c;
        }
    }
    return out + "\"";
}


static void writeGraph(std::ostringstream& o, const GraphView& v) {
    o << "{\"length\":" << v.length << ",\"nodes\":[";
    for (size_t i = 0; i < v.graph.nodes.size(); i++) {
        const Instr& in = v.graph.nodes[i];
        o << (i ? "," : "") << "[" << js(formatInstr(in)) << "," << in.line << "," << (int)in.op->engine << ","
          << v.cycles[i] << "," << v.graph.footprints[i].doneAge << "]";
    }
    o << "],\"edges\":[";
    bool first = true;
    for (size_t e = 0; e < v.graph.edges.size(); e++) {
        if (v.redundant[e]) continue;
        const Edge& ed = v.graph.edges[e];
        o << (first ? "" : ",") << "[" << ed.from << "," << ed.to << "," << ed.distance << "," << (int)ed.kind << ","
          << js(ed.reason) << "]";
        first = false;
    }
    o << "]}";
}

static void writeSim(std::ostringstream& o, const SimResult& r) {
    o << "{\"cycles\":" << r.cycles << ",\"issued\":" << r.issued << ",\"delays\":" << r.delays << ",\"problems\":[";
    std::vector<std::string> problems = r.violations;
    if (!r.stopReason.empty()) problems.push_back("simulation stopped: " + r.stopReason);
    for (size_t i = 0; i < problems.size(); i++) o << (i ? "," : "") << js(problems[i]);
    o << "]}";
}

// Engine lanes follow the Engine enum order; edge colors follow EdgeKind.
static const char* kPage = R"HTML(<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --ground: #F5F7FA; --panel: #FFFFFF; --ink: #18212E; --muted: #5A6577; --grid: #EDF0F4; --rule: #D6DCE4;
  --accent: #0B7A84; --good: #2F7D4F; --warn: #B4412F;
  --e0: #6E7B8F; --e1: #2E8A5A; --e2: #B8562B; --e3: #C98A1E; --e4: #3B6DB5; --e5: #8756B0; --e6: #0B7A84;
  --k0: #2F66B3; --k1: #C46A1F; --k2: #8B4FB9; --k3: #C23B4B; --k4: #9AA3B0;
  --mono: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --ground: #10151D; --panel: #171E28; --ink: #E4E9F0; --muted: #9AA5B5; --grid: #1D2530; --rule: #2A3441;
    --accent: #3CC3CC; --good: #6CCB91; --warn: #F08A74;
    --e0: #9AA7BA; --e1: #57C08A; --e2: #EE8656; --e3: #F0B74C; --e4: #6FA0EC; --e5: #B990E3; --e6: #3CC3CC;
    --k0: #6FA0EC; --k1: #F0A25C; --k2: #B98BEA; --k3: #F07684; --k4: #6D7888;
  }
}
body { background: var(--ground); color: var(--ink); font: 14px/1.45 system-ui, sans-serif; margin: 0; }
.wrap { padding: 20px; display: grid; gap: 16px; max-width: 1600px; margin: 0 auto; }
h1 { font-size: 20px; margin: 0; }
h1 code { font: 18px var(--mono); color: var(--accent); }
.stats { display: flex; flex-wrap: wrap; gap: 28px; }
.stat small { display: block; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
.stat b { font: 500 22px var(--mono); }
.good { color: var(--good); }
.warn { border-left: 3px solid var(--warn); padding: 8px 12px; background: var(--panel); font: 12px var(--mono); white-space: pre-wrap; }
.controls { display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center; color: var(--muted); }
.legend { display: flex; gap: 12px; font-size: 12px; }
.legend i { width: 16px; height: 2px; display: inline-block; margin-right: 4px; vertical-align: middle; }
.main { display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 16px; align-items: start; }
@media (max-width: 900px) { .main { grid-template-columns: minmax(0, 1fr); } }
.panel { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; margin-bottom: 14px; }
.panel header { display: flex; justify-content: space-between; flex-wrap: wrap; padding: 8px 12px; border-bottom: 1px solid var(--rule); }
.panel header span { font: 12px var(--mono); color: var(--muted); }
.scroll { overflow-x: auto; }
svg text { font-family: var(--mono); }
.node { cursor: pointer; }
.details { position: sticky; top: 16px; padding: 12px 14px; font-size: 13px; display: grid; gap: 10px; }
.details h3 { margin: 0; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); font-weight: 500; }
.details code { font: 12px var(--mono); overflow-wrap: anywhere; }
.details ul { list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; font-size: 12px; }
.muted { color: var(--muted); }
</style>
<div class="wrap">
  <h1>Dependency graph of <code id="src"></code></h1>
  <div class="stats" id="stats"></div>
  <div id="warnings"></div>
  <div class="controls">
    <label>Block <select id="block"></select></label>
    <label>Zoom <input type="range" id="zoom" min="1" max="40" step="0.5"></label>
    <span class="legend" id="legend"></span>
  </div>
  <div class="main">
    <div>
      <section class="panel"><header><b>Before (as written)</b><span id="len-before"></span></header><div class="scroll" id="view-before"></div></section>
      <section class="panel"><header><b>After atlas-opt</b><span id="len-after"></span></header><div class="scroll" id="view-after"></div></section>
    </div>
    <aside class="panel details" id="details"></aside>
  </div>
</div>
<script>
const DATA = __DATA__;
const ENGINES = ["Scalar", "LSU", "MXU0", "MXU1", "VPU", "XLU", "DMA"];
const KINDS = ["RAW", "WAR", "WAW", "rule", "order"];
const NS = "http://www.w3.org/2000/svg";
const $ = id => document.getElementById(id);
const fmt = n => n.toLocaleString("en-US");
const esc = s => s.replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const state = { block: 0, zoom: 6, selected: null };
const panels = {};
const HINT = `<span class="muted">Select an instruction to see what it waits for and what waits for it.</span>`;

function el(tag, attrs, parent, text) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  parent.appendChild(e);
  return e;
}

// One lane per engine; instructions whose bars would overlap stack in sub-rows.
function layout(g, lanesUsed, ppc) {
  const LEFT = 64, ROW = 18, PAD = 8;
  const rows = ENGINES.map(() => []);
  const pos = g.nodes.map(([text, , eng, cycle, done]) => {
    const x = LEFT + cycle * ppc, barEnd = x + Math.max(3, (done + 1) * ppc);
    const end = Math.max(barEnd, x + 6 + text.split(" ")[0].length * 7.2);
    let r = rows[eng].findIndex(e => e + 3 <= x);
    if (r < 0) r = rows[eng].push(0) - 1;
    rows[eng][r] = end;
    return { x, row: r, eng, barEnd };
  });
  const top = [];
  let y = 26;
  ENGINES.forEach((_, i) => { top[i] = y; if (lanesUsed[i]) y += Math.max(1, rows[i].length) * ROW + 2 * PAD; });
  pos.forEach(p => { p.y = top[p.eng] + PAD + p.row * ROW + ROW / 2; });
  return { pos, top, height: y + 6, left: LEFT };
}

function drawPanel(side, g, other, lanesUsed) {
  const ppc = state.zoom, host = $("view-" + side);
  host.innerHTML = "";
  const L = layout(g, lanesUsed, ppc), maxLen = Math.max(g.length, other.length, 1);
  const width = L.left + maxLen * ppc + 60;
  const svg = el("svg", { width, height: L.height, viewBox: `0 0 ${width} ${L.height}` }, host);
  const defs = el("defs", {}, svg);
  KINDS.forEach((_, k) => {
    const m = el("marker", { id: `arrow-${side}-${k}`, viewBox: "0 0 6 6", refX: 5.5, refY: 3, markerWidth: 6, markerHeight: 6, orient: "auto" }, defs);
    el("path", { d: "M0,0 L6,3 L0,6 z", fill: `var(--k${k})` }, m);
  });
  const grid = el("g", {}, svg), edgesG = el("g", {}, svg), nodesG = el("g", {}, svg);
  const step = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000].find(s => s * ppc >= 60) || 10000;
  for (let c = 0; c <= maxLen; c += step) {
    const x = L.left + c * ppc;
    el("line", { x1: x, x2: x, y1: 18, y2: L.height, stroke: "var(--grid)" }, grid);
    el("text", { x: x + 2, y: 12, "font-size": 10, fill: "var(--muted)" }, grid, c);
  }
  const endX = L.left + g.length * ppc;
  el("line", { x1: endX, x2: endX, y1: 18, y2: L.height, stroke: "var(--accent)", "stroke-dasharray": "4 3" }, grid);
  el("text", { x: endX + 3, y: L.height - 4, "font-size": 10, fill: "var(--accent)" }, grid, "next block");
  ENGINES.forEach((name, i) => {
    if (!lanesUsed[i]) return;
    el("rect", { x: 0, y: L.top[i], width, height: 1, fill: "var(--rule)" }, grid);
    el("text", { x: 6, y: L.top[i] + 16, "font-size": 11, fill: `var(--e${i})` }, grid, name);
  });
  const edges = g.edges.map(([s, t, , kind]) => {
    const a = L.pos[s], b = L.pos[t], dx = Math.max(24, (b.x - a.x) / 2);
    const p = el("path", { d: `M${a.x},${a.y} C${a.x + dx},${a.y} ${b.x - dx},${b.y} ${b.x - 1},${b.y}`, fill: "none",
      stroke: `var(--k${kind})`, "stroke-width": 1.2, opacity: 0.55, "marker-end": `url(#arrow-${side}-${kind})` }, edgesG);
    return { p, s, t };
  });
  const nodes = g.nodes.map(([text, line, eng, cycle], i) => {
    const p = L.pos[i], n = el("g", { class: "node" }, nodesG);
    el("rect", { x: p.x, y: p.y - 5, width: Math.max(3, p.barEnd - p.x), height: 10, rx: 2, fill: `var(--e${eng})`, opacity: 0.18 }, n);
    el("rect", { x: p.x - 1.5, y: p.y - 7, width: 3, height: 14, rx: 1, fill: `var(--e${eng})` }, n);
    el("text", { x: p.x + 5, y: p.y + 4, "font-size": 11, fill: "var(--ink)" }, n, text.split(" ")[0]);
    el("title", {}, n, `${text}\nline ${line} · cycle ${cycle}`);
    n.addEventListener("click", () => select(side, i));
    n.addEventListener("mouseenter", () => highlight(side, i));
    n.addEventListener("mouseleave", () => highlight(side, null));
    return n;
  });
  panels[side] = { g, edges, nodes };
}

function highlight(side, i) {
  const P = panels[side];
  if (i === null && state.selected && state.selected.side === side) i = state.selected.index;
  const linked = new Set([i]);
  P.edges.forEach(e => {
    const on = e.s === i || e.t === i;
    if (on) linked.add(e.s).add(e.t);
    e.p.setAttribute("opacity", i === null ? 0.55 : on ? 1 : 0.06);
    e.p.setAttribute("stroke-width", on ? 2 : 1.2);
  });
  P.nodes.forEach((n, k) => n.setAttribute("opacity", i === null || linked.has(k) ? 1 : 0.3));
}

function select(side, i) {
  state.selected = { side, index: i };
  const g = panels[side].g, [text, line, eng, cycle, done] = g.nodes[i];
  const otherSide = side === "before" ? "after" : "before", og = panels[otherSide].g;
  const twin = og.nodes.findIndex(n => n[0] === text && n[1] === line);  // same instruction in the other panel
  highlight(side, i);
  highlight(otherSide, twin >= 0 ? twin : null);
  const list = (edges, from) => edges.length ? "<ul>" + edges.map(([s, t, d, k, why]) =>
    `<li><code>${esc(g.nodes[from ? s : t][0])}</code><br><span class="muted"><span style="color:var(--k${k})">${KINDS[k]}</span> · ${d} cycle${d === 1 ? "" : "s"} · ${esc(why)}</span></li>`
  ).join("") + "</ul>" : `<span class="muted">none</span>`;
  $("details").innerHTML = `
    <div><h3>${side} · ${ENGINES[eng]}</h3><code>${esc(text)}</code></div>
    <div class="muted">Line ${line} · issues at cycle ${cycle} · busy until cycle ${cycle + done}</div>
    <div><h3>Waits for</h3>${list(g.edges.filter(e => e[1] === i), true)}</div>
    <div><h3>Needed by</h3>${list(g.edges.filter(e => e[0] === i), false)}</div>`;
}

function render() {
  const blk = DATA.blocks[state.block];
  state.selected = null;
  $("details").innerHTML = HINT;
  $("len-before").textContent = `${fmt(blk.before.length)} cycles`;
  $("len-after").textContent = `${fmt(blk.after.length)} cycles · critical path ${fmt(blk.lb)}`;
  const lanes = ENGINES.map(() => 0);
  [blk.before, blk.after].forEach(g => g.nodes.forEach(n => { lanes[n[2]] = 1; }));
  drawPanel("before", blk.before, blk.after, lanes);
  drawPanel("after", blk.after, blk.before, lanes);
}

function fitZoom() {
  const blk = DATA.blocks[state.block];
  const avail = Math.max(300, $("view-before").clientWidth - 120);
  state.zoom = Math.max(1, Math.min(40, avail / Math.max(1, blk.before.length, blk.after.length)));
  $("zoom").value = state.zoom;
}

const b = DATA.before, a = DATA.after;
$("src").textContent = DATA.source;
$("stats").innerHTML = [
  ["Cycles", `${fmt(b.cycles)} → ${fmt(a.cycles)}`],
  ["Speedup", `<span class="good">${(b.cycles / Math.max(1, a.cycles)).toFixed(2)}×</span>`],
  ["Instructions issued", `${fmt(b.issued)} → ${fmt(a.issued)}`],
  ["Delays issued", `${fmt(b.delays)} → ${fmt(a.delays)}`],
].map(([l, v]) => `<div class="stat"><small>${l}</small><b>${v}</b></div>`).join("");
if (a.problems.length) $("warnings").innerHTML = `<div class="warn">The optimized program breaks timing rules:\n${esc(a.problems.join("\n"))}</div>`;
$("legend").innerHTML = KINDS.map((k, i) => `<span><i style="background:var(--k${i})"></i>${k}</span>`).join("");
DATA.blocks.forEach((blk, i) => {
  const o = document.createElement("option");
  o.value = i;
  o.textContent = `${blk.name}: ${fmt(blk.before.length)} → ${fmt(blk.after.length)} cycles`;
  $("block").appendChild(o);
  if (blk.before.length - blk.after.length > DATA.blocks[state.block].before.length - DATA.blocks[state.block].after.length) state.block = i;
});
$("block").value = state.block;
fitZoom();
render();
$("block").addEventListener("change", e => { state.block = +e.target.value; fitZoom(); render(); });
$("zoom").addEventListener("input", e => { state.zoom = +e.target.value; render(); });
</script>
)HTML";

std::string renderHtml(const ProgramView& view) {
    std::ostringstream o;
    o << "{\"source\":" << js(view.source) << ",\"before\":";
    writeSim(o, view.before);
    o << ",\"after\":";
    writeSim(o, view.after);
    o << ",\"blocks\":[";
    for (size_t b = 0; b < view.blocks.size(); b++) {
        const BlockView& bv = view.blocks[b];
        o << (b ? "," : "") << "{\"name\":" << js(bv.name) << ",\"lb\":" << bv.lowerBound << ",\"before\":";
        writeGraph(o, bv.before);
        o << ",\"after\":";
        writeGraph(o, bv.after);
        o << "}";
    }
    o << "]}";

    std::string title = view.source.substr(view.source.find_last_of('/') + 1);
    std::replace(title.begin(), title.end(), '<', '_');
    std::string page = kPage;
    page.replace(page.find("__TITLE__"), 9, title + " schedule graph");
    page.replace(page.find("__DATA__"), 8, o.str());
    return page;
}
