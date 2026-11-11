"""Generate a self-contained interactive HTML report for the sentiment arc predictions."""

import csv
import json

CSV_PATH = "outputs/sentiment_arc.csv"
OUT_PATH = "outputs/sentiment_arc_report.html"

MODEL = {
    "dataset": "Douban reviews (qhchina-data, douban.json)",
    "classes": "positive = 4-5 stars, negative = 1-2 stars (3 stars excluded)",
    "segmentation": "jieba",
    "features": "TF-IDF (TfidfVectorizer, 106,836 terms)",
    "classifier": "LogisticRegression",
    "train": 64000,
    "test": 16000,
    "accuracy": 0.8666,
    "precision": 0.8517,
    "recall": 0.8876,
    "f1": 0.8693,
    "tn": 6764, "fp": 1236, "fn": 899, "tp": 7101,
}

POS_WORDS = [
    ["我们", 4.06], ["一口气", 4.00], ["四星", 3.89], ["虽然", 3.51], ["震撼", 3.48],
    ["很棒", 3.44], ["一生", 3.39], ["真实", 3.34], ["不错", 3.21], ["慢慢", 3.12],
    ["停不下来", 2.97], ["细腻", 2.95], ["哭", 2.86], ["有时", 2.80], ["很", 2.78],
    ["五星", 2.77], ["最", 2.70], ["有时候", 2.70], ["完美", 2.69], ["与", 2.69],
]
NEG_WORDS = [
    ["抄袭", -8.47], ["失望", -8.13], ["看不下去", -6.90], ["恶心", -6.63], ["无聊", -6.47],
    ["矫情", -6.21], ["两星", -6.18], ["不", -6.17], ["一星", -6.12], ["女主", -5.27],
    ["作者", -5.25], ["实在", -5.22], ["垃圾", -5.15], ["郭敬明", -4.93], ["鸡汤", -4.90],
    ["烂", -4.86], ["浪费", -4.85], ["莫名其妙", -4.73], ["弃", -4.65], ["无病呻吟", -4.64],
]

TEMPLATE = r'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>《我城》 · Sentiment Arc Explorer</title>
<style>
:root {
  --pos: #16a34a; --neg: #dc2626; --ma: #2563eb;
  --ink: #111827; --muted: #6b7280; --line: #e5e7eb; --bg: #f3f4f6;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Noto Sans SC",
    "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  font-size: 15px; line-height: 1.55;
}
main { max-width: 1180px; margin: 0 auto; padding: 20px 16px 60px; }
header.hero { padding: 26px 16px 8px; text-align: center; }
header.hero h1 { margin: 0 0 4px; font-size: 26px; }
header.hero p.sub { margin: 0; color: var(--muted); }
.chips { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; margin: 16px 0 20px; }
.chip {
  background: #fff; border: 1px solid var(--line); border-radius: 999px;
  padding: 6px 14px; font-size: 13.5px; box-shadow: 0 1px 2px rgba(0,0,0,.04);
}
.chip b { font-variant-numeric: tabular-nums; }
.card {
  background: #fff; border: 1px solid var(--line); border-radius: 14px;
  box-shadow: 0 1px 3px rgba(0,0,0,.05); padding: 18px 20px; margin-bottom: 18px;
}
.card h2 { margin: 0 0 12px; font-size: 17px; }
.card-head { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; justify-content: space-between; }
.controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; font-size: 13px; color: var(--muted); }
.controls input[type=range] { width: 130px; vertical-align: middle; }
.controls label { white-space: nowrap; }
button, .btn {
  font: inherit; font-size: 13px; padding: 5px 12px; border-radius: 8px;
  border: 1px solid var(--line); background: #f9fafb; cursor: pointer;
}
button:hover { background: #eef2f7; }
#chartWrap { position: relative; }
#chart { display: block; width: 100%; cursor: crosshair; }
#tooltip {
  display: none; position: absolute; width: 240px; z-index: 5; pointer-events: none;
  background: rgba(17,24,39,.93); color: #f9fafb; font-size: 12.5px; line-height: 1.45;
  padding: 8px 10px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,.25);
}
.tick { font-size: 10px; fill: #9ca3af; }
.legend { display: flex; gap: 18px; flex-wrap: wrap; margin-top: 8px; font-size: 12.5px; color: var(--muted); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.sw { width: 18px; height: 3px; border-radius: 2px; display: inline-block; }
.dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.note { margin: 10px 0 0; font-size: 12.5px; color: var(--muted); }
.badge {
  color: #fff; border-radius: 999px; padding: 2px 12px; font-size: 13px;
  font-variant-numeric: tabular-nums; display: inline-block; margin-left: 8px;
}
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
@media (max-width: 860px) { .grid2 { grid-template-columns: 1fr; } }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 18px; }
.li {
  display: flex; gap: 8px; align-items: baseline; padding: 6px 8px; border-radius: 8px;
  cursor: pointer; font-size: 13.5px;
}
.li:hover { background: #f3f4f6; }
.li.active { background: #dbeafe; }
.li-n { color: var(--muted); min-width: 44px; font-variant-numeric: tabular-nums; }
.li-s { min-width: 58px; font-variant-numeric: tabular-nums; font-weight: 600; }
.li-t { color: #374151; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
h3.mini { margin: 10px 0 4px; font-size: 13.5px; }
h3.mini:first-child { margin-top: 0; }
.wrow { display: flex; align-items: center; gap: 8px; padding: 2.5px 0; font-size: 13px; }
.wword { min-width: 64px; }
.wbar { flex: 1; background: #f3f4f6; border-radius: 4px; height: 10px; overflow: hidden; }
.wbar i { display: block; height: 100%; border-radius: 4px; }
.wval { min-width: 48px; text-align: right; font-variant-numeric: tabular-nums; color: var(--muted); }
.wcols { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 26px; }
@media (max-width: 640px) { .wcols, .cols { grid-template-columns: 1fr; } }
.ptext {
  white-space: pre-wrap; font-size: 15.5px; line-height: 1.9; max-height: 42vh;
  overflow: auto; padding: 12px 14px; background: #fafafa; border: 1px solid var(--line);
  border-radius: 10px;
}
mark { border-radius: 3px; padding: 0 1px; }
.w-pos { background: #dcfce7; box-shadow: inset 0 -1.5px 0 var(--pos); }
.w-neg { background: #fee2e2; box-shadow: inset 0 -1.5px 0 var(--neg); }
.hit { background: #fef08a; outline: 1.5px solid #eab308; }
.muted { color: var(--muted); }
.mono { font-variant-numeric: tabular-nums; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; font-size: 12px; }
tr[data-n] { cursor: pointer; }
tr[data-n]:hover { background: #f9fafb; }
tr.active { background: #dbeafe; }
.bar { position: relative; width: 160px; height: 12px; background: #f3f4f6; border-radius: 3px; }
.bar::before { content: ""; position: absolute; left: 50%; top: -2px; bottom: -2px; width: 1px; background: #cbd5e1; }
.bar i { position: absolute; top: 0; bottom: 0; border-radius: 3px; }
details summary { cursor: pointer; font-weight: 600; }
.kv { display: grid; grid-template-columns: auto 1fr; gap: 4px 14px; font-size: 13.5px; }
.kv dt { color: var(--muted); }
.kv dd { margin: 0; }
.cm { border-collapse: collapse; font-size: 13px; margin-top: 8px; }
.cm td, .cm th { border: 1px solid var(--line); padding: 5px 12px; text-align: center; }
.cm th { background: #f9fafb; font-size: 12px; }
footer { text-align: center; color: var(--muted); font-size: 12.5px; padding: 8px 0 30px; }
input[type=search] {
  font: inherit; font-size: 13.5px; padding: 6px 12px; border: 1px solid var(--line);
  border-radius: 8px; width: 220px;
}
input[type=search]:focus { outline: 2px solid #bfdbfe; border-color: var(--ma); }
</style>
</head>
<body>
<header class="hero">
  <h1>《我城》（西西）— Sentiment Arc Explorer</h1>
  <p class="sub">Passage-level sentiment predicted with a TF-IDF + LogisticRegression model trained on Douban reviews · score = P(positive) − P(negative)</p>
</header>
<main>
  <div class="chips" id="chips"></div>

  <section class="card">
    <div class="card-head">
      <h2>Sentiment arc</h2>
      <div class="controls">
        <input id="q" type="search" placeholder="Search in passages… (Enter = first hit)">
        <span id="matchInfo" class="muted"></span>
        <label>from <input id="zStart" type="range" min="1" value="1"></label>
        <label>to <input id="zEnd" type="range" min="1"></label>
        <button id="zReset">reset zoom</button>
        <label><input id="maToggle" type="checkbox" checked> moving avg</label>
      </div>
    </div>
    <div id="chartWrap">
      <svg id="chart"></svg>
      <div id="tooltip"></div>
    </div>
    <div class="legend">
      <span><i class="sw" style="background:#9ca3af"></i> passage score</span>
      <span><i class="sw" style="background:var(--ma)"></i> 10-passage moving average</span>
      <span><i class="dot" style="background:var(--pos)"></i> positive (≥ 0)</span>
      <span><i class="dot" style="background:var(--neg)"></i> negative (&lt; 0)</span>
      <span class="muted">hover = preview · click = inspect · ←/→ = step</span>
    </div>
    <p class="note">Note: this ebook edition also contains front matter (contents, author bio) and appended essays by 何福仁; late passage numbers include that material, which explains some of the extremes there. The model was trained on film reviews, so scores are domain-shifted predictions, not ground truth.</p>
  </section>

  <section class="card">
    <div class="card-head">
      <h2 id="detailTitle">Selected passage</h2>
      <div class="controls">
        <button id="prevBtn">← prev</button>
        <button id="nextBtn">next →</button>
      </div>
    </div>
    <div id="detailBody"></div>
    <p class="note">Highlighted words are the model's strongest positive (green) and negative (red) TF-IDF weights.</p>
  </section>

  <div class="grid2">
    <section class="card">
      <h2>Most positive / negative passages</h2>
      <div class="cols">
        <div><h3 class="mini" style="color:var(--pos)">Top 5 positive</h3><div id="topList"></div></div>
        <div><h3 class="mini" style="color:var(--neg)">Top 5 negative</h3><div id="botList"></div></div>
      </div>
    </section>
    <section class="card">
      <h2>Model's strongest words (reference)</h2>
      <div class="wcols">
        <div><h3 class="mini" style="color:var(--pos)">Positive weights</h3><div id="posWords"></div></div>
        <div><h3 class="mini" style="color:var(--neg)">Negative weights</h3><div id="negWords"></div></div>
      </div>
    </section>
  </div>

  <div class="grid2">
    <section class="card">
      <h2>Distribution of passage scores</h2>
      <div id="hist"></div>
    </section>
    <section class="card">
      <h2>Model &amp; evaluation</h2>
      <dl class="kv" id="modelInfo"></dl>
      <table class="cm" id="cm"></table>
    </section>
  </div>

  <section class="card">
    <details>
      <summary>All passages (<span id="nShown"></span>)</summary>
      <table>
        <thead><tr><th style="width:44px">#</th><th style="width:170px">score</th><th style="width:70px">value</th><th>text (preview)</th></tr></thead>
        <tbody id="tbody"></tbody>
      </table>
    </details>
  </section>
</main>
<footer>Generated by <code>scripts/build_report.py</code> from <code>outputs/sentiment_arc.csv</code> · model: <code>outputs/sentiment_model.joblib</code></footer>

<script>
const PAYLOAD = __PAYLOAD__;
const { passages, model, posWords, negWords } = PAYLOAD;
const N = passages.length;
const MAW = 10;
const ma = [];
for (let i = MAW - 1; i < N; i++) {
  let s = 0;
  for (let j = i - MAW + 1; j <= i; j++) s += passages[j].score;
  ma.push({ n: i + 1, v: s / MAW });
}
let zs = 1, ze = N, sel = null, showMA = true, query = "";

const $ = id => document.getElementById(id);
const POS = "#16a34a", NEG = "#dc2626";
const colorOf = v => v >= 0 ? POS : NEG;
const fmt = v => (v > 0 ? "+" : "") + v.toFixed(3);
const esc = s => s.replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const preview = (t, n) => t.replace(/\s+/g, " ").trim().slice(0, n);

const WORDS = posWords.map(x => ({ w: x[0], p: 1 }))
  .concat(negWords.map(x => ({ w: x[0], p: -1 })))
  .sort((a, b) => b.w.length - a.w.length);
const reEsc = s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const wordAlts = WORDS.map(x => reEsc(x.w)).join("|");

function highlight(text, q) {
  const src = q ? "(" + wordAlts + ")|(" + reEsc(q) + ")" : "(" + wordAlts + ")";
  const re = new RegExp(src, "g");
  let out = "", last = 0, m;
  while ((m = re.exec(text))) {
    out += esc(text.slice(last, m.index));
    const w = WORDS.find(x => x.w === m[0]);
    const cls = w ? (w.p > 0 ? "w-pos" : "w-neg") : "hit";
    out += '<mark class="' + cls + '">' + esc(m[0]) + "</mark>";
    last = m.index + m[0].length;
  }
  return out + esc(text.slice(last));
}

const wrap = $("chartWrap"), svg = $("chart"), tip = $("tooltip");
const MG = { t: 14, r: 14, b: 30, l: 46 };

function renderChart() {
  const w = Math.max(320, wrap.clientWidth), h = 380;
  svg.setAttribute("viewBox", "0 0 " + w + " " + h);
  svg.setAttribute("width", w); svg.setAttribute("height", h);
  const iw = w - MG.l - MG.r, ih = h - MG.t - MG.b;
  const X = i => MG.l + (i - zs) / Math.max(1, ze - zs) * iw;
  const Y = v => MG.t + (1 - (v + 1) / 2) * ih;
  let g = "";
  for (const t of [-1, -0.5, 0, 0.5, 1]) {
    const y = Y(t).toFixed(1);
    g += '<line x1="' + MG.l + '" y1="' + y + '" x2="' + (w - MG.r) + '" y2="' + y + '" stroke="' +
      (t === 0 ? "#9ca3af" : "#e5e7eb") + '" stroke-width="1"' + (t === 0 ? ' stroke-dasharray="5 4"' : "") + "/>";
    g += '<text x="' + (MG.l - 8) + '" y="' + (Y(t) + 4).toFixed(1) + '" text-anchor="end" class="tick">' + t + "</text>";
  }
  const step = Math.max(1, Math.ceil((ze - zs + 1) / 12));
  for (let i = zs; i <= ze; i += step)
    g += '<text x="' + X(i).toFixed(1) + '" y="' + (h - 8) + '" text-anchor="middle" class="tick">' + i + "</text>";
  const pts = [];
  for (let i = zs; i <= ze; i++) pts.push([X(i), Y(passages[i - 1].score), i]);
  if (pts.length > 1)
    g += '<polyline points="' + pts.map(p => p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" ") +
      '" fill="none" stroke="#9ca3af" stroke-width="1.3" opacity="0.65"/>';
  for (const p of pts)
    g += '<circle cx="' + p[0].toFixed(1) + '" cy="' + p[1].toFixed(1) + '" r="' + (sel === p[2] ? 5.5 : 2.6) +
      '" fill="' + colorOf(passages[p[2] - 1].score) + '"' +
      (sel === p[2] ? ' stroke="#111827" stroke-width="1.5"' : ' opacity="0.85"') + "/>";
  if (showMA) {
    const mp = ma.filter(p => p.n >= zs && p.n <= ze);
    if (mp.length > 1)
      g += '<polyline points="' + mp.map(p => X(p.n).toFixed(1) + "," + Y(p.v).toFixed(1)).join(" ") +
        '" fill="none" stroke="#2563eb" stroke-width="2.4"/>';
  }
  if (sel)
    g += '<line x1="' + X(sel).toFixed(1) + '" y1="' + MG.t + '" x2="' + X(sel).toFixed(1) + '" y2="' + (h - MG.b) +
      '" stroke="#111827" stroke-width="1" opacity="0.25"/>';
  svg.innerHTML = g;
  const locate = ev => {
    const r = svg.getBoundingClientRect();
    const px = ev.clientX - r.left;
    return Math.min(ze, Math.max(zs, Math.round(zs + (px - MG.l) / Math.max(1, ze - zs) * (ze - zs))));
  };
  svg.onmousemove = ev => {
    const r = svg.getBoundingClientRect();
    const p = passages[locate(ev) - 1];
    tip.style.display = "block";
    tip.innerHTML = "<b>#" + p.n + "</b> <span style=\"color:" + (p.score >= 0 ? "#4ade80" : "#f87171") + "\">" +
      fmt(p.score) + "</span><br>" + esc(preview(p.text, 150));
    let tx = ev.clientX - r.left + 14;
    if (tx + 250 > w) tx = ev.clientX - r.left - 254;
    tip.style.left = tx + "px";
    tip.style.top = Math.max(0, Math.min(ev.clientY - r.top + 12, h - 80)) + "px";
  };
  svg.onmouseleave = () => tip.style.display = "none";
  svg.onclick = ev => select(locate(ev));
}

function renderDetail() {
  const d = $("detailBody");
  if (!sel) {
    d.innerHTML = '<p class="muted">Hover the chart and click a passage to inspect it here, or pick one from the lists below. Use the ←/→ arrow keys to step through the novel.</p>';
    $("detailTitle").textContent = "Selected passage";
    return;
  }
  const p = passages[sel - 1];
  $("detailTitle").innerHTML = "Passage #" + p.n + ' <span class="badge" style="background:' +
    colorOf(p.score) + '">' + fmt(p.score) + "</span>";
  d.innerHTML = '<div class="ptext">' + highlight(p.text, query) + "</div>";
}

function listItem(i) {
  const p = passages[i];
  return '<div class="li' + (sel === p.n ? " active" : "") + '" data-n="' + p.n + '"><span class="li-n">#' +
    p.n + '</span><span class="li-s" style="color:' + colorOf(p.score) + '">' + fmt(p.score) +
    '</span><span class="li-t">' + esc(preview(p.text, 46)) + "</span></div>";
}

function renderLists() {
  const idx = passages.map((_, i) => i);
  const byScore = idx.slice().sort((a, b) => passages[b].score - passages[a].score);
  $("topList").innerHTML = byScore.slice(0, 5).map(listItem).join("");
  $("botList").innerHTML = byScore.slice(-5).reverse().map(listItem).join("");
}

function renderWords() {
  const max = Math.max(...posWords.map(x => x[1]), ...negWords.map(x => Math.abs(x[1])));
  const row = (x, neg) => '<div class="wrow"><span class="wword">' + esc(x[0]) +
    '</span><span class="wbar"><i style="width:' + (Math.abs(x[1]) / max * 100).toFixed(1) +
    "%;background:" + (neg ? NEG : POS) + '"></i></span><span class="wval">' + x[1].toFixed(2) + "</span></div>";
  $("posWords").innerHTML = posWords.map(x => row(x, false)).join("");
  $("negWords").innerHTML = negWords.map(x => row(x, true)).join("");
}

function renderHist() {
  const B = 20, bins = new Array(B).fill(0);
  for (const p of passages)
    bins[Math.min(B - 1, Math.max(0, Math.floor((p.score + 1) / 2 * B)))]++;
  const bw = 26, gap = 4, h = 130, w = B * (bw + gap) + 24, mx = Math.max(...bins, 1);
  let s = '<svg viewBox="0 0 ' + w + " " + (h + 22) + '" width="100%" style="max-width:560px">';
  bins.forEach((c, k) => {
    const bh = c / mx * (h - 18);
    const x = 12 + k * (bw + gap), center = -1 + (k + 0.5) / B * 2;
    s += '<rect x="' + x + '" y="' + (h - bh) + '" width="' + bw + '" height="' + Math.max(bh, c ? 2 : 0) +
      '" rx="3" fill="' + colorOf(center) + '" opacity="0.85"/>';
    if (c) s += '<text x="' + (x + bw / 2) + '" y="' + (h - bh - 4) + '" text-anchor="middle" class="tick">' + c + "</text>";
    if (k % 5 === 0 || k === B - 1)
      s += '<text x="' + (x + bw / 2) + '" y="' + (h + 16) + '" text-anchor="middle" class="tick">' +
        (-1 + k / B * 2).toFixed(1) + "</text>";
  });
  s += '<line x1="12" y1="' + h + '" x2="' + (w - 12) + '" y2="' + h + '" stroke="#d1d5db"/></svg>';
  $("hist").innerHTML = s;
}

function renderModel() {
  const pct = v => (v * 100).toFixed(2) + "%";
  $("modelInfo").innerHTML =
    "<dt>Training data</dt><dd>" + esc(model.dataset) + " — " + esc(model.classes) + "</dd>" +
    "<dt>Pipeline</dt><dd>" + esc(model.segmentation) + " → " + esc(model.features) + " → " + esc(model.classifier) + "</dd>" +
    "<dt>Train / test</dt><dd>" + model.train + " / " + model.test + " reviews (stratified 80/20, random_state=42)</dd>" +
    "<dt>Accuracy</dt><dd><b>" + pct(model.accuracy) + "</b></dd>" +
    "<dt>Precision / Recall / F1</dt><dd>" + pct(model.precision) + " / " + pct(model.recall) + " / " + pct(model.f1) + "</dd>";
  $("cm").innerHTML =
    "<tr><th>confusion</th><th>pred neg</th><th>pred pos</th></tr>" +
    "<tr><th>true neg</th><td>" + model.tn + "</td><td>" + model.fp + "</td></tr>" +
    "<tr><th>true pos</th><td>" + model.fn + "</td><td>" + model.tp + "</td></tr>";
}

function renderTable() {
  const rows = [];
  let shown = 0;
  for (const p of passages) {
    if (query && !p.text.includes(query)) continue;
    shown++;
    const wpc = Math.abs(p.score) * 50;
    const left = p.score >= 0 ? 50 : 50 - wpc;
    rows.push('<tr data-n="' + p.n + '"' + (sel === p.n ? ' class="active"' : "") + "><td>" + p.n +
      '</td><td><div class="bar"><i style="left:' + left + "%;width:" + wpc + "%;background:" +
      colorOf(p.score) + '"></i></div></td><td class="mono" style="color:' + colorOf(p.score) + '">' +
      fmt(p.score) + "</td><td>" + highlight(preview(p.text, 64), query) + "</td></tr>");
  }
  $("tbody").innerHTML = rows.join("") || '<tr><td colspan="4" class="muted">no matches</td></tr>';
  $("nShown").textContent = shown + " of " + N + " shown";
  $("matchInfo").textContent = query ? shown + " hit" + (shown === 1 ? "" : "s") : "";
}

function updateChips() {
  const scores = passages.map(p => p.score);
  const mean = scores.reduce((a, b) => a + b, 0) / N;
  const pos = scores.filter(v => v > 0).length;
  const chips = [
    ["passages", N], ["mean score", fmt(mean)], ["max", fmt(Math.max(...scores))],
    ["min", fmt(Math.min(...scores))], ["positive passages", pos + " (" + (pos / N * 100).toFixed(0) + "%)"],
    ["test accuracy", (model.accuracy * 100).toFixed(1) + "%"],
  ];
  $("chips").innerHTML = chips.map(c => '<span class="chip">' + c[0] + ": <b>" + c[1] + "</b></span>").join("");
}

function select(n) {
  sel = n;
  renderChart(); renderDetail(); renderLists(); renderTable();
  const el = document.querySelector("tr[data-n='" + n + "']");
  if (el && el.scrollIntoView) el.scrollIntoView({ block: "nearest" });
}

document.body.addEventListener("click", ev => {
  const t = ev.target.closest("[data-n]");
  if (t) select(+t.dataset.n);
});
$("prevBtn").onclick = () => { if (sel && sel > 1) select(sel - 1); };
$("nextBtn").onclick = () => { if (sel && sel < N) select(sel + 1); };
document.addEventListener("keydown", ev => {
  if (/INPUT|TEXTAREA|SELECT/.test(ev.target.tagName)) return;
  if (ev.key === "ArrowRight" && sel && sel < N) select(sel + 1);
  if (ev.key === "ArrowLeft" && sel && sel > 1) select(sel - 1);
});
$("zStart").max = N; $("zEnd").max = N; $("zEnd").value = N;
$("zStart").oninput = e => { zs = Math.min(+e.target.value, ze - 1); e.target.value = zs; renderChart(); };
$("zEnd").oninput = e => { ze = Math.max(+e.target.value, zs + 1); e.target.value = ze; renderChart(); };
$("zReset").onclick = () => { zs = 1; ze = N; $("zStart").value = 1; $("zEnd").value = N; renderChart(); };
$("maToggle").onchange = e => { showMA = e.target.checked; renderChart(); };
$("q").addEventListener("input", e => { query = e.target.value.trim(); renderTable(); renderDetail(); });
$("q").addEventListener("keydown", e => {
  if (e.key === "Enter" && query) {
    const i = passages.findIndex(p => p.text.includes(query));
    if (i >= 0) select(i + 1);
  }
});
window.addEventListener("resize", renderChart);

updateChips(); renderWords(); renderHist(); renderModel(); renderLists(); renderTable(); renderDetail(); renderChart();
</script>
</body>
</html>
'''


def main():
    passages = []
    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            passages.append({"n": int(row["passage_number"]), "score": float(row["score"]), "text": row["text"]})
    payload = json.dumps(
        {"passages": passages, "model": MODEL, "posWords": POS_WORDS, "negWords": NEG_WORDS},
        ensure_ascii=False,
    ).replace("</", "<\\/")
    html = TEMPLATE.replace("__PAYLOAD__", payload)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {OUT_PATH} ({len(html) / 1024:.0f} KB, {len(passages)} passages)")


if __name__ == "__main__":
    main()
