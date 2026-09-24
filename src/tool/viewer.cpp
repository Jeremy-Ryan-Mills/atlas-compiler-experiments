#include "tool/viewer.h"

#include <algorithm>
#include <climits>
#include <sstream>

static const Engine kEngines[] = {Engine::Scalar, Engine::Lsu, Engine::Mxu0, Engine::Mxu1,
                                  Engine::Vpu, Engine::Xlu, Engine::Dma};

static int engineIndex(Engine e) {
    for (int i = 0; i < 7; i++)
        if (kEngines[i] == e) return i;
    return 0;
}

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
                             const SimResult& before, const SimResult& after, const std::vector<std::string>& log) {
    ProgramView view;
    view.source = source;
    view.before = before;
    view.after = after;
    view.log = log;
    Code orig = buildBlocks(original);
    std::vector<RegValues> entryOrig = blockEntryValues(orig);
    std::vector<RegValues> entryOpt = blockEntryValues(optimized);
    uint32_t dmaRegs = dmaOperandRegisters(original.instrs);

    for (size_t bi = 0; bi < orig.blocks.size() && bi < optimized.blocks.size(); bi++) {
        const Block& ob = orig.blocks[bi];
        BlockView bv;
        bv.name = ob.labels.empty() ? "block " + std::to_string(bi) : ob.labels[0];
        if (bi == 0 && ob.labels.empty()) bv.name = "entry";

        std::vector<Instr> seq = blockInstructions(ob);
        std::vector<int> cycles = asWrittenCycles(seq);
        int length = seq.empty() ? 0 : cycles.back() + naturalGap(seq.back());
        bv.before = makeView(seq, cycles, entryOrig[bi], length, dmaRegs);
        bv.lowerBound = criticalPathLength(bv.before.graph);

        const Block& nb = optimized.blocks[bi];
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
        o << (i ? "," : "") << "[" << js(formatInstr(in)) << "," << in.line << "," << engineIndex(in.op->engine)
          << "," << v.cycles[i] << "," << v.graph.footprints[i].doneAge << "]";
    }
    o << "],\"edges\":[";
    for (size_t e = 0; e < v.graph.edges.size(); e++) {
        const Edge& ed = v.graph.edges[e];
        o << (e ? "," : "") << "[" << ed.from << "," << ed.to << "," << ed.distance << "," << (int)ed.kind << ","
          << js(ed.reason) << "," << (v.redundant[e] ? 1 : 0) << "]";
    }
    o << "]}";
}

static void writeSim(std::ostringstream& o, const SimResult& r) {
    o << "{\"cycles\":" << r.cycles << ",\"issued\":" << r.issued << ",\"delays\":" << r.delays << ",\"busy\":{";
    bool first = true;
    for (auto& [engine, cycles] : r.busyCycles) {
        o << (first ? "" : ",") << js(engine) << ":" << cycles;
        first = false;
    }
    o << "},\"violations\":[";
    for (size_t i = 0; i < r.violations.size(); i++) o << (i ? "," : "") << js(r.violations[i]);
    o << "],\"stop\":" << js(r.stopReason) << "}";
}

static const char* kPage = R"HTML(<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {
  --ground: #F5F7FA; --panel: #FFFFFF; --ink: #18212E; --muted: #5A6577; --faint: #8A94A4;
  --rule: #D6DCE4; --grid: #EDF0F4; --accent: #0B7A84; --accent-soft: #DDF1F2; --warn: #B4412F; --good: #2F7D4F;
  --e0: #6E7B8F; --e1: #2E8A5A; --e2: #B8562B; --e3: #C98A1E; --e4: #3B6DB5; --e5: #8756B0; --e6: #0B7A84;
  --k0: #2F66B3; --k1: #C46A1F; --k2: #8B4FB9; --k3: #C23B4B; --k4: #9AA3B0;
  --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --ground: #10151D; --panel: #171E28; --ink: #E4E9F0; --muted: #9AA5B5; --faint: #6D7888;
    --rule: #2A3441; --grid: #1D2530; --accent: #3CC3CC; --accent-soft: #163A3E; --warn: #F08A74; --good: #6CCB91;
    --e0: #9AA7BA; --e1: #57C08A; --e2: #EE8656; --e3: #F0B74C; --e4: #6FA0EC; --e5: #B990E3; --e6: #3CC3CC;
    --k0: #6FA0EC; --k1: #F0A25C; --k2: #B98BEA; --k3: #F07684; --k4: #6D7888;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --ground: #10151D; --panel: #171E28; --ink: #E4E9F0; --muted: #9AA5B5; --faint: #6D7888;
  --rule: #2A3441; --grid: #1D2530; --accent: #3CC3CC; --accent-soft: #163A3E; --warn: #F08A74; --good: #6CCB91;
  --e0: #9AA7BA; --e1: #57C08A; --e2: #EE8656; --e3: #F0B74C; --e4: #6FA0EC; --e5: #B990E3; --e6: #3CC3CC;
  --k0: #6FA0EC; --k1: #F0A25C; --k2: #B98BEA; --k3: #F07684; --k4: #6D7888;
}
body { background: var(--ground); color: var(--ink); font: 14px/1.45 var(--sans); margin: 0; }
.wrap { padding-inline: 20px; padding-block: 20px 40px; display: grid; gap: 18px; max-width: 1600px; margin: 0 auto; }
h1 { font-size: 20px; font-weight: 600; margin: 0; text-wrap: balance; }
h1 code { font: 500 18px var(--mono); color: var(--accent); }
.sub { color: var(--muted); margin: 2px 0 0; }
.stats { display: flex; flex-wrap: wrap; gap: 28px; align-items: flex-end; }
.stat .label { font-size: 11px; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
.stat .value { font: 500 22px var(--mono); font-variant-numeric: tabular-nums; }
.stat .value small { font-size: 13px; color: var(--muted); }
.stat .value.good { color: var(--good); }
.engines { display: grid; grid-template-columns: auto 1fr 1fr; gap: 3px 14px; font-size: 12px; align-items: center; max-width: 560px; }
.engines .hdr { color: var(--muted); font-size: 11px; letter-spacing: .06em; text-transform: uppercase; }
.bar { height: 8px; border-radius: 2px; background: var(--grid); position: relative; margin-right: 38px; }
.bar i { position: absolute; inset: 0 auto 0 0; border-radius: 2px; }
.bar b { position: absolute; left: calc(100% + 6px); top: -4px; font: 11px var(--mono); color: var(--muted); font-weight: 400; }
.eng { display: inline-flex; align-items: center; gap: 6px; }
.dot { width: 9px; height: 9px; border-radius: 2px; display: inline-block; }
.warn { border-left: 3px solid var(--warn); padding: 8px 12px; background: var(--panel); font: 12px var(--mono); }
.controls { display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center; position: sticky; top: env(safe-area-inset-top, 0px);
  background: var(--ground); padding-block: 8px; z-index: 5; border-bottom: 1px solid var(--rule); }
.controls label { display: inline-flex; gap: 6px; align-items: center; color: var(--muted); font-size: 13px; }
select, input[type=range] { font: 13px var(--sans); color: var(--ink); background: var(--panel); border: 1px solid var(--rule); border-radius: 4px; padding: 3px 6px; max-width: 100%; }
.seg { display: inline-flex; border: 1px solid var(--rule); border-radius: 4px; overflow: hidden; }
.seg button { font: 13px var(--sans); border: 0; background: var(--panel); color: var(--muted); padding: 4px 10px; cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--accent-soft); color: var(--ink); }
button:focus-visible, select:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.legend { display: flex; flex-wrap: wrap; gap: 12px; font-size: 12px; color: var(--muted); }
.legend span { display: inline-flex; gap: 5px; align-items: center; }
.legend i { width: 16px; height: 2px; display: inline-block; }
.main { display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 18px; align-items: start; }
@media (max-width: 900px) { .main { grid-template-columns: minmax(0, 1fr); } }
.panel { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; }
.panel + .panel { margin-top: 14px; }
.panel header { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 6px 16px; padding: 8px 12px; border-bottom: 1px solid var(--rule); }
.panel header h2 { font-size: 13px; font-weight: 600; margin: 0; }
.panel header span { font: 12px var(--mono); color: var(--muted); font-variant-numeric: tabular-nums; }
.scroll { overflow-x: auto; }
svg text { font-family: var(--mono); }
.details { position: sticky; top: 70px; padding: 12px 14px; font-size: 13px; display: grid; gap: 10px; }
.details h3 { margin: 0; font-size: 11px; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); font-weight: 500; }
.details code { font: 13px var(--mono); overflow-wrap: anywhere; }
.details ul { list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; }
.details li { font-size: 12px; line-height: 1.35; }
.details li code { font-size: 12px; }
.details .why { color: var(--muted); }
.empty { color: var(--muted); }
details.log summary { cursor: pointer; color: var(--muted); }
details.log pre { font: 12px var(--mono); white-space: pre-wrap; margin: 6px 0 0; }
.node { cursor: pointer; }
.node:focus-visible rect.mark { stroke: var(--accent); stroke-width: 2; }
@media (prefers-reduced-motion: no-preference) { .edge, .node { transition: opacity .12s; } }
</style>
<div class="wrap">
  <div>
    <h1>Dependency graph of <code id="src"></code></h1>
    <p class="sub">Each block's instructions before and after atlas-opt, placed at the cycle they issue. Cycle totals come from the rtl-match timing model.</p>
  </div>
  <div class="stats" id="stats"></div>
  <div id="warnings"></div>
  <div class="engines" id="engines"></div>
  <details class="log"><summary>Pass log</summary><pre id="log"></pre></details>
  <div class="controls">
    <label for="block">Block <select id="block"></select></label>
    <span class="seg" role="group" aria-label="Layout">
      <button type="button" id="lay-time" aria-pressed="true">Timeline</button>
      <button type="button" id="lay-layer" aria-pressed="false">Dependency layers</button>
    </span>
    <label for="zoom">Zoom <input type="range" id="zoom" min="1" max="40" step="0.5" value="6"></label>
    <label for="implied"><input type="checkbox" id="implied"> Show implied edges</label>
    <span class="legend" id="legend"></span>
  </div>
  <div class="main">
    <div>
      <section class="panel"><header><h2>Before (as written)</h2><span id="len-before"></span></header><div class="scroll" id="view-before"></div></section>
      <section class="panel"><header><h2>After atlas-opt</h2><span id="len-after"></span></header><div class="scroll" id="view-after"></div></section>
    </div>
    <aside class="panel details" id="details" aria-live="polite"><span class="empty">Select an instruction to see what it waits for and what waits for it.</span></aside>
  </div>
</div>
<script>
const DATA = __DATA__;
const ENGINES = ["Scalar", "LSU", "MXU0", "MXU1", "VPU", "XLU", "DMA"];
const KINDS = ["RAW", "WAR", "WAW", "rule", "order"];
const KIND_TEXT = ["read after write", "write after read", "write after write", "hardware rule", "ordering"];
const NS = "http://www.w3.org/2000/svg";
const $ = id => document.getElementById(id);
const state = { block: 0, layout: "time", implied: false, zoom: 6, selected: null };

function el(tag, attrs, parent, text) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}
const fmt = n => n.toLocaleString("en-US");

function renderSummary() {
  $("src").textContent = DATA.source;
  const b = DATA.before, a = DATA.after;
  const speed = a.cycles > 0 ? b.cycles / a.cycles : 1;
  const stats = [
    ["Cycles before", fmt(b.cycles), ""],
    ["Cycles after", fmt(a.cycles), a.cycles < b.cycles ? "good" : ""],
    ["Speedup", speed.toFixed(2) + "<small>×</small>", speed > 1 ? "good" : ""],
    ["Instructions issued", fmt(b.issued) + " <small>→</small> " + fmt(a.issued), ""],
    ["Delays issued", fmt(b.delays) + " <small>→</small> " + fmt(a.delays), ""],
  ];
  $("stats").innerHTML = stats.map(([l, v, c]) => `<div class="stat"><div class="label">${l}</div><div class="value ${c}">${v}</div></div>`).join("");
  const w = [];
  if (a.violations.length) w.push("The optimized program breaks timing rules:\n" + a.violations.join("\n"));
  if (b.violations.length) w.push("The original program already breaks timing rules:\n" + b.violations.join("\n"));
  if (a.stop) w.push("Simulation of the optimized program stopped: " + a.stop);
  $("warnings").innerHTML = w.map(t => `<div class="warn">${escapeHtml(t).replace(/\n/g, "<br>")}</div>`).join("");
  let rows = `<span class="hdr">Engine busy</span><span class="hdr">before</span><span class="hdr">after</span>`;
  ENGINES.forEach((name, i) => {
    const pb = b.cycles ? (b.busy[name] || 0) / b.cycles : 0, pa = a.cycles ? (a.busy[name] || 0) / a.cycles : 0;
    if (!pb && !pa) return;
    const bar = p => `<span class="bar"><i style="width:${(p * 100).toFixed(1)}%;background:var(--e${i})"></i><b>${(p * 100).toFixed(0)}%</b></span>`;
    rows += `<span class="eng"><span class="dot" style="background:var(--e${i})"></span>${name}</span>${bar(pb)}${bar(pa)}`;
  });
  $("engines").innerHTML = rows;
  $("log").textContent = DATA.log.join("\n");
  $("legend").innerHTML = KINDS.map((k, i) => `<span title="${KIND_TEXT[i]}"><i style="background:var(--k${i})"></i>${k}</span>`).join("");
  const sel = $("block");
  DATA.blocks.forEach((blk, i) => {
    const o = document.createElement("option");
    o.value = i;
    o.textContent = `${blk.name}: ${fmt(blk.before.length)} → ${fmt(blk.after.length)} cycles`;
    sel.appendChild(o);
  });
  let best = 0;
  DATA.blocks.forEach((blk, i) => { const g = blk.before.length - blk.after.length, bg = DATA.blocks[best].before.length - DATA.blocks[best].after.length; if (g > bg) best = i; });
  state.block = best;
  sel.value = best;
}

function escapeHtml(s) { return s.replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
const mnemonic = text => text.split(" ")[0];

// Timeline: x is the issue cycle, one lane per engine; overlapping instructions stack in sub-rows.
function layoutTimeline(g, lanesUsed, ppc) {
  const LEFT = 64, ROW = 18, LANE_PAD = 8, TOP = 26;
  const rowsPerLane = ENGINES.map(() => []);
  const pos = g.nodes.map(([text, line, eng, cycle, done]) => {
    const x = LEFT + cycle * ppc;
    const barEnd = x + Math.max(3, (done + 1) * ppc);
    const labelEnd = x + 6 + mnemonic(text).length * 7.2;
    const rows = rowsPerLane[eng];
    let r = rows.findIndex(end => end + 3 <= x);
    if (r < 0) { r = rows.length; rows.push(0); }
    rows[r] = Math.max(barEnd, labelEnd);
    return { x, row: r, eng, barEnd };
  });
  const laneTop = [];
  let y = TOP;
  ENGINES.forEach((_, i) => {
    laneTop[i] = y;
    if (lanesUsed[i]) y += Math.max(1, rowsPerLane[i].length, lanesUsed[i]) * ROW + LANE_PAD * 2;
  });
  pos.forEach(p => { p.y = laneTop[p.eng] + LANE_PAD + p.row * ROW + ROW / 2; });
  return { pos, laneTop, height: y + 6, left: LEFT, rows: rowsPerLane.map(r => r.length) };
}

// Layers: column = longest chain of dependences leading to the instruction.
function layoutLayers(g) {
  const n = g.nodes.length, layer = new Array(n).fill(0);
  g.edges.forEach(([s, t]) => { layer[t] = Math.max(layer[t], layer[s] + 1); });
  const COL = 210, ROWH = 26, count = [];
  const pos = g.nodes.map((_, i) => {
    const L = layer[i];
    count[L] = (count[L] || 0) + 1;
    return { x: 16 + L * COL, y: 16 + (count[L] - 1) * ROWH + 10, w: COL - 34 };
  });
  const width = 16 + (Math.max(0, ...layer) + 1) * COL;
  const height = 16 + Math.max(0, ...count.map(c => c || 0)) * ROWH + 10;
  return { pos, width, height };
}

function drawPanel(side, g, other, ppc, lanesUsed) {
  const host = $("view-" + side);
  host.innerHTML = "";
  const svg = el("svg", { role: "img", "aria-label": `${side} dependency graph` }, host);
  const defs = el("defs", {}, svg);
  KINDS.forEach((_, k) => {
    const m = el("marker", { id: `arrow-${side}-${k}`, viewBox: "0 0 6 6", refX: 5.5, refY: 3, markerWidth: 6, markerHeight: 6, orient: "auto" }, defs);
    el("path", { d: "M0,0 L6,3 L0,6 z", fill: `var(--k${k})` }, m);
  });
  const gEdges = el("g", {}, svg), gNodes = el("g", {}, svg);
  let pos, width, height;
  if (state.layout === "time") {
    const maxLen = Math.max(g.length, other.length, 1);
    const L = layoutTimeline(g, lanesUsed, ppc);
    pos = L.pos; height = L.height; width = L.left + maxLen * ppc + 60;
    const gGrid = el("g", {}, svg);
    svg.insertBefore(gGrid, gEdges);
    const steps = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000];
    const step = steps.find(s => s * ppc >= 60) || 20000;
    for (let c = 0; c <= maxLen; c += step) {
      const x = L.left + c * ppc;
      el("line", { x1: x, x2: x, y1: 18, y2: height, stroke: "var(--grid)", "stroke-width": 1 }, gGrid);
      el("text", { x: x + 2, y: 12, "font-size": 10, fill: "var(--faint)" }, gGrid, c);
    }
    const endX = L.left + g.length * ppc;
    el("line", { x1: endX, x2: endX, y1: 18, y2: height, stroke: "var(--accent)", "stroke-width": 1.5, "stroke-dasharray": "4 3" }, gGrid);
    el("text", { x: endX + 3, y: height - 4, "font-size": 10, fill: "var(--accent)" }, gGrid, "next block");
    ENGINES.forEach((name, i) => {
      if (!lanesUsed[i]) return;
      el("rect", { x: 0, y: L.laneTop[i], width: width, height: 1, fill: "var(--rule)" }, gGrid);
      el("text", { x: 6, y: L.laneTop[i] + 16, "font-size": 11, fill: `var(--e${i})`, "font-weight": 500 }, gGrid, name);
    });
  } else {
    const L = layoutLayers(g);
    pos = L.pos; width = L.width; height = L.height;
  }
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const edgeEls = [];
  g.edges.forEach(([s, t, d, kind, why, redundant], ei) => {
    if (redundant && !state.implied) return;
    const a = pos[s], b = pos[t];
    let x1, y1, x2, y2;
    if (state.layout === "time") { x1 = a.x; y1 = a.y; x2 = b.x; y2 = b.y; }
    else { x1 = a.x + a.w; y1 = a.y; x2 = b.x; y2 = b.y; }
    const dx = Math.max(24, Math.abs(x2 - x1) / 2);
    const p = el("path", {
      d: `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2 - 1},${y2}`, fill: "none",
      stroke: `var(--k${kind})`, "stroke-width": 1.2, opacity: redundant ? 0.25 : 0.55,
      "marker-end": `url(#arrow-${side}-${kind})`, class: "edge" }, gEdges);
    edgeEls.push({ p, s, t, base: redundant ? 0.25 : 0.55 });
  });

  const nodeEls = g.nodes.map(([text, line, eng, cycle, done], i) => {
    const p = pos[i];
    const node = el("g", { class: "node", tabindex: 0, role: "button", "aria-label": `${text}, issues at cycle ${cycle}` }, gNodes);
    if (state.layout === "time") {
      el("rect", { x: p.x, y: p.y - 5, width: Math.max(3, p.barEnd - p.x), height: 10, rx: 2, fill: `var(--e${eng})`, opacity: 0.18 }, node);
      el("rect", { class: "mark", x: p.x - 1.5, y: p.y - 7, width: 3, height: 14, rx: 1, fill: `var(--e${eng})` }, node);
      el("text", { x: p.x + 5, y: p.y + 4, "font-size": 11, fill: "var(--ink)" }, node, mnemonic(text));
    } else {
      el("rect", { class: "mark", x: p.x, y: p.y - 10, width: p.w, height: 20, rx: 3, fill: "var(--panel)", stroke: `var(--e${eng})` }, node);
      el("rect", { x: p.x, y: p.y - 10, width: 4, height: 20, rx: 1, fill: `var(--e${eng})` }, node);
      const label = text.length > 22 ? text.slice(0, 21) + "…" : text;
      el("text", { x: p.x + 9, y: p.y + 4, "font-size": 11, fill: "var(--ink)" }, node, label);
      el("text", { x: p.x + p.w - 4, y: p.y + 4, "font-size": 10, fill: "var(--faint)", "text-anchor": "end" }, node, cycle);
    }
    el("title", {}, node, `${text}\nline ${line} · cycle ${cycle}`);
    const choose = () => select(side, i);
    node.addEventListener("click", choose);
    node.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); choose(); } });
    node.addEventListener("mouseenter", () => highlight(side, i));
    node.addEventListener("mouseleave", () => highlight(side, null));
    return node;
  });
  panels[side] = { g, edgeEls, nodeEls };
}

const panels = {};

function highlight(side, i) {
  const P = panels[side];
  if (i === null && state.selected && state.selected.side === side) i = state.selected.index;
  P.edgeEls.forEach(e => {
    const on = i === null || e.s === i || e.t === i;
    e.p.setAttribute("opacity", i === null ? e.base : on ? 1 : 0.06);
    e.p.setAttribute("stroke-width", i !== null && on ? 2 : 1.2);
  });
  const linked = new Set([i]);
  if (i !== null) P.edgeEls.forEach(e => { if (e.s === i) linked.add(e.t); if (e.t === i) linked.add(e.s); });
  P.nodeEls.forEach((n, k) => n.setAttribute("opacity", i === null || linked.has(k) ? 1 : 0.3));
}

function select(side, i) {
  state.selected = { side, index: i };
  const g = panels[side].g, [text, line, eng, cycle, done] = g.nodes[i];
  const otherSide = side === "before" ? "after" : "before";
  const og = panels[otherSide].g;
  const twin = og.nodes.findIndex(n => n[0] === text && n[1] === line);
  highlight(side, i);
  if (twin >= 0) highlight(otherSide, twin); else highlight(otherSide, null);
  const list = (edges, dirFrom) => edges.length ? "<ul>" + edges.map(([s, t, d, k, why]) => {
    const n = g.nodes[dirFrom ? s : t];
    return `<li><code>${escapeHtml(n[0])}</code><br><span class="why"><span style="color:var(--k${k})">${KINDS[k]}</span> · ${d} cycle${d === 1 ? "" : "s"} · ${escapeHtml(why)}</span></li>`;
  }).join("") + "</ul>" : `<span class="empty">none</span>`;
  const into = g.edges.filter(e => e[1] === i), outOf = g.edges.filter(e => e[0] === i);
  const otherCycle = twin >= 0 ? og.nodes[twin][3] : null;
  $("details").innerHTML = `
    <div><h3>${side === "before" ? "Before" : "After"} · ${ENGINES[eng]}</h3><code>${escapeHtml(text)}</code></div>
    <div class="why">Source line ${line} · issues at cycle ${cycle} · busy until cycle ${cycle + done}${otherCycle !== null ? ` · cycle ${otherCycle} ${side === "before" ? "after" : "before"} optimization` : ""}</div>
    <div><h3>Waits for</h3>${list(into, true)}</div>
    <div><h3>Needed by</h3>${list(outOf, false)}</div>`;
}

function render() {
  const blk = DATA.blocks[state.block];
  if (!blk) return;
  state.selected = null;
  $("details").innerHTML = `<span class="empty">Select an instruction to see what it waits for and what waits for it.</span>`;
  $("len-before").textContent = `${fmt(blk.before.length)} cycles · ${blk.before.nodes.length} instructions`;
  $("len-after").textContent = `${fmt(blk.after.length)} cycles · ${blk.after.nodes.length} instructions · critical path ${fmt(blk.lb)}`;
  const lanes = ENGINES.map((_, i) => 0);
  [blk.before, blk.after].forEach(g => g.nodes.forEach(n => { lanes[n[2]] = 1; }));
  drawPanel("before", blk.before, blk.after, state.zoom, lanes);
  drawPanel("after", blk.after, blk.before, state.zoom, lanes);
  $("zoom").disabled = state.layout !== "time";
}

function fitZoom() {
  const blk = DATA.blocks[state.block];
  const avail = Math.max(300, $("view-before").clientWidth - 120);
  state.zoom = Math.max(1, Math.min(40, avail / Math.max(1, blk.before.length, blk.after.length)));
  $("zoom").value = state.zoom;
}

renderSummary();
fitZoom();
render();
$("block").addEventListener("change", e => { state.block = +e.target.value; fitZoom(); render(); });
$("zoom").addEventListener("input", e => { state.zoom = +e.target.value; render(); });
$("implied").addEventListener("change", e => { state.implied = e.target.checked; render(); });
["time", "layer"].forEach(k => $("lay-" + k).addEventListener("click", () => {
  state.layout = k;
  $("lay-time").setAttribute("aria-pressed", k === "time");
  $("lay-layer").setAttribute("aria-pressed", k === "layer");
  render();
}));
</script>
)HTML";

std::string renderHtml(const ProgramView& view) {
    std::ostringstream o;
    o << "{\"source\":" << js(view.source) << ",\"before\":";
    writeSim(o, view.before);
    o << ",\"after\":";
    writeSim(o, view.after);
    o << ",\"log\":[";
    for (size_t i = 0; i < view.log.size(); i++) o << (i ? "," : "") << js(view.log[i]);
    o << "],\"blocks\":[";
    for (size_t b = 0; b < view.blocks.size(); b++) {
        const BlockView& bv = view.blocks[b];
        o << (b ? "," : "") << "{\"name\":" << js(bv.name) << ",\"lb\":" << bv.lowerBound << ",\"before\":";
        writeGraph(o, bv.before);
        o << ",\"after\":";
        writeGraph(o, bv.after);
        o << "}";
    }
    o << "]}";

    std::string page = kPage;
    std::string title = view.source;
    size_t slash = title.find_last_of('/');
    if (slash != std::string::npos) title = title.substr(slash + 1);
    std::string escapedTitle;
    for (char c : title) escapedTitle += (c == '<' || c == '>' || c == '&') ? '_' : c;
    page.replace(page.find("__TITLE__"), 9, escapedTitle + " schedule graph");
    page.replace(page.find("__DATA__"), 8, o.str());
    return page;
}
