# -*- coding: utf-8 -*-
"""trade_review v2 共享资产：review.css / review.js / 总览页模板 / Pair 页模板。

这些常量由 export_trade_review_html() 写入回测输出目录：
  trade_review_assets/review.css
  trade_review_assets/review.js
总览页与 Pair 页通过相对路径引用这两个文件。
"""

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
REVIEW_CSS = r"""
:root {
  color-scheme: light;
  --bg: #f5f7fb;
  --panel: #ffffff;
  --ink: #17202a;
  --muted: #667085;
  --line: #d7dde8;
  --blue: #2563eb;
  --orange: #f97316;
  --green: #16a34a;
  --red: #dc2626;
  --purple: #7c3aed;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
}
header {
  padding: 18px 28px 10px;
  border-bottom: 1px solid var(--line);
  background: #fff;
  position: sticky;
  top: 0;
  z-index: 10;
}
h1 { font-size: 22px; margin: 0 0 10px; font-weight: 720; }
h2 { font-size: 16px; margin: 0 0 8px; font-weight: 650; }
.topbar { display: grid; grid-template-columns: 1fr auto; gap: 16px; align-items: end; }
.metrics { display: flex; flex-wrap: wrap; gap: 8px; }
.metric {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 6px 10px;
  background: #fbfdff;
  font-size: 12px;
  color: var(--muted);
}
.metric b { display: block; font-size: 15px; color: var(--ink); margin-top: 2px; }
.buttons { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.buttons button, .buttons select {
  border: 1px solid var(--line);
  background: #fff;
  border-radius: 6px;
  padding: 6px 12px;
  font-size: 13px;
  cursor: pointer;
}
.buttons button.primary { background: var(--blue); color: #fff; border-color: var(--blue); }
.pair-jump { display: inline-flex; align-items: center; gap: 6px; color: var(--muted); font-size: 13px; }
.pair-jump select { min-width: 150px; }
main { display: grid; grid-template-columns: 1fr 320px; gap: 14px; padding: 16px 28px 28px; max-width: 1500px; margin: 0 auto; min-width: 0; }
.pair-review main { grid-template-columns: minmax(0, 1fr) 500px; max-width: 1800px; }
main > * { min-width: 0; }
.side { display: flex; flex-direction: column; gap: 14px; min-width: 0; }
.chart-stack { min-width: 0; }
.table-wrap { overflow-x: auto; max-width: 100%; }
@media (max-width: 1100px) {
  main { grid-template-columns: 1fr; }
  .side { position: static; }
  .topbar { grid-template-columns: 1fr; }
}
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 14px 16px;
}
.chart { width: 100%; height: 280px; display: block; }
.chart.tall { height: 340px; }
.legend { display: flex; gap: 12px; flex-wrap: wrap; color: var(--muted); font-size: 12px; margin-bottom: 4px; }
.dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; margin-right: 5px; vertical-align: middle; }
.range-panel {
  display: flex; gap: 14px; align-items: center; flex-wrap: wrap;
  background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 10px 14px; margin-bottom: 14px; font-size: 13px;
}
.range-panel label { display: flex; align-items: center; gap: 6px; color: var(--muted); }
.range-panel input[type="range"] { width: 190px; }
.boundary-value { color: var(--ink); font-variant-numeric: tabular-nums; }
.toggle-label { user-select: none; }
.summary { display: grid; gap: 6px; }
.summary-row { display: flex; justify-content: space-between; font-size: 13px; }
.summary-row span { color: var(--muted); }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { border-bottom: 1px solid var(--line); padding: 7px 10px; text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-weight: 600; background: #fafbfc; }
tbody tr { cursor: pointer; }
tbody tr:hover { background: #f8fafc; }
tbody tr.active-row { background: #eff6ff; }
.pos { color: var(--green); }
.neg { color: var(--red); }
.trade-list { display: flex; flex-direction: column; gap: 10px; max-height: 620px; overflow-y: auto; }
.pair-review .trade-list { max-height: 760px; }
.trade-card {
  border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px;
  font-size: 12.5px; line-height: 1.55; cursor: pointer; background: #fbfdff;
}
.trade-card:hover { border-color: var(--blue); }
.trade-card.selected {
  border: 3px solid var(--blue) !important;
  background: #dbeafe !important;
  box-shadow: inset 6px 0 0 var(--blue), 0 0 0 3px rgba(37, 99, 235, .14);
}
.trade-title { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px; }
.trade-title .selected-label {
  color: #fff; background: var(--blue); border-radius: 4px;
  padding: 2px 7px; font-weight: 700;
}
.trade-block { margin-top: 4px; padding-top: 4px; border-top: 1px dashed var(--line); }
.trade-pnl { font-weight: 700; }
.trade-scenario-title { display:flex; justify-content:space-between; gap:8px; align-items:baseline; margin-bottom:5px; }
.trade-scenario-title span { color:var(--muted); font-size:11px; text-align:right; }
.scenario-table { table-layout:fixed; width:100%; font-size:11px; }
.scenario-table th, .scenario-table td { padding:4px 3px; text-align:right; overflow:hidden; text-overflow:ellipsis; }
.scenario-table th:first-child, .scenario-table td:first-child { text-align:left; width:18%; }
.scenario-table th:nth-child(2), .scenario-table td:nth-child(2),
.scenario-table th:nth-child(3), .scenario-table td:nth-child(3) { width:27%; }
.scenario-table th:last-child, .scenario-table td:last-child { width:28%; }
.scenario-table tbody tr { cursor:default; }
.scenario-table tbody tr:hover { background:#f8fafc; }
.scenario-actual { margin-top:6px; padding:7px 8px; background:#f8fafc; border:1px solid var(--line); border-radius:5px; }
.scenario-unavailable { color:var(--muted); font-size:11.5px; }
.hint { color: var(--muted); font-size: 12px; margin: 4px 0; }
#loadError {
  display: none;
  background: #fef2f2; color: #b91c1c;
  padding: 10px 20px; font-size: 13px; line-height: 1.6;
  border-bottom: 1px solid #fecaca; white-space: pre-wrap;
}
.back-link { display: inline-block; margin-bottom: 6px; color: var(--blue); text-decoration: none; font-size: 13px; }
.back-link:hover { text-decoration: underline; }
.buttons .back-link {
  margin-bottom: 0; border: 1px solid var(--line); border-radius: 6px;
  padding: 6px 12px; background: #fff; text-decoration: none;
}
.buttons .back-link:hover { border-color: var(--blue); text-decoration: none; }
.buttons .back-link[hidden] { display: none; }
"""


# ---------------------------------------------------------------------------
# JS：共享核心库 + 两个页面的初始化入口
# ---------------------------------------------------------------------------
REVIEW_JS = r"""
"use strict";

// ---------- 列式打包数据还原 ----------
function unpackSeries(d, keys) {
  if (Array.isArray(d)) return d;
  if (!d || !d.t) return [];
  const t = d.t;
  let ts = d.t0 || 0;
  const out = new Array(t.length);
  for (let i = 0; i < t.length; i++) {
    ts += t[i];
    const row = { ts };
    for (const k of keys) row[k] = d[k] ? d[k][i] : undefined;
    out[i] = row;
  }
  return out;
}

// ---------- 二分 / 抽稀 / 极值 ----------
function lowerBound(arr, ts) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid].ts < ts) lo = mid + 1; else hi = mid; }
  return lo;
}
function upperBound(arr, ts) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid].ts <= ts) lo = mid + 1; else hi = mid; }
  return lo;
}
function decimate(data, maxPoints) {
  if (data.length <= maxPoints) return data;
  const step = data.length / maxPoints;
  const out = [];
  for (let i = 0; i < data.length; i += step) out.push(data[Math.floor(i)]);
  if (out[out.length - 1] !== data[data.length - 1]) out.push(data[data.length - 1]);
  return out;
}
function minMax(values) {
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  return [mn, mx];
}
function nearestPoint(arr, ts) {
  if (!arr.length) return null;
  const idx = lowerBound(arr, ts);
  let best = null, bestDist = Infinity;
  for (const i of [idx - 1, idx]) {
    if (i < 0 || i >= arr.length) continue;
    const d = Math.abs(arr[i].ts - ts);
    if (d < bestDist) { bestDist = d; best = arr[i]; }
  }
  return best || arr[0];
}

// ---------- 通用工具 ----------
function pct(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return (value * 100).toFixed(2) + "%";
}
function num(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return Number(value).toLocaleString("en-US", { maximumFractionDigits: digits });
}
function dt(ts) {
  return new Date(ts).toISOString().slice(0, 16).replace("T", " ");
}
function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}
function fmtDuration(minutes) {
  if (minutes === null || minutes === undefined || Number.isNaN(minutes)) return "-";
  if (minutes >= 1440) return (minutes / 1440).toFixed(1) + " 天";
  if (minutes >= 60) return (minutes / 60).toFixed(1) + " 小时";
  return minutes.toFixed(0) + " 分钟";
}
function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.floor(rect.width * dpr);
  canvas.height = Math.floor(rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: rect.width, h: rect.height };
}
function showLoadError(msg) {
  const el = document.getElementById("loadError");
  if (el) {
    el.style.display = "block";
    el.textContent = msg;
  }
}

// ---------- 图表绘制 ----------
function drawLineChart(canvasId, data, series, options = {}) {
  const canvas = document.getElementById(canvasId);
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const pad = { left: 62, right: 72, top: 14, bottom: 34 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;
  const allValues = [];
  for (const s of series) {
    for (const p of data) {
      const v = p[s.key];
      if (v !== null && v !== undefined && !Number.isNaN(v)) allValues.push(v);
    }
  }
  const thresholdRows = options.thresholds || [];
  if (thresholdRows.length) allValues.push(...thresholdRows.map(item => Number(item.value)));
  if (!data.length || !allValues.length) {
    ctx.fillStyle = "#667085";
    ctx.font = "13px Microsoft YaHei, Arial";
    ctx.fillText("没有数据", pad.left, pad.top + 20);
    return;
  }
  let yMin = Infinity, yMax = -Infinity;
  for (const v of allValues) {
    if (v < yMin) yMin = v;
    if (v > yMax) yMax = v;
  }
  if (yMin === yMax) { yMin -= 1; yMax += 1; }
  const yPad = (yMax - yMin) * 0.08;
  yMin -= yPad; yMax += yPad;
  const x = ts => pad.left + (ts - options.viewStart) / (options.viewEnd - options.viewStart) * plotW;
  const y = value => pad.top + (yMax - value) / (yMax - yMin) * plotH;

  ctx.strokeStyle = "#d7dde8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top + plotH);
  ctx.stroke();

  ctx.font = "12px Microsoft YaHei, Arial";
  for (let i = 0; i <= 4; i++) {
    const value = yMin + (yMax - yMin) * i / 4;
    const yy = y(value);
    ctx.strokeStyle = "#eef1f6";
    ctx.beginPath();
    ctx.moveTo(pad.left, yy);
    ctx.lineTo(pad.left + plotW, yy);
    ctx.stroke();
    ctx.fillStyle = "#667085";
    ctx.textAlign = "left";
    ctx.fillText(options.yFormat ? options.yFormat(value) : value.toFixed(2), 6, yy + 4);
  }
  ctx.textAlign = "left";
  for (const tick of [options.viewStart, (options.viewStart + options.viewEnd) / 2, options.viewEnd]) {
    ctx.fillStyle = "#667085";
    ctx.fillText(dt(tick).slice(5), x(tick) - 40, pad.top + plotH + 22);
  }

  if (thresholdRows.length) {
    for (const threshold of thresholdRows) {
      const value = Number(threshold.value);
      const yy = y(value);
      const isEntry = threshold.kind === "entry";
      ctx.setLineDash(isEntry ? [7, 4] : [3, 4]);
      ctx.strokeStyle = isEntry ? "#475569" : "#94a3b8";
      ctx.lineWidth = isEntry ? 1.5 : 1;
      ctx.beginPath();
      ctx.moveTo(pad.left, yy);
      ctx.lineTo(pad.left + plotW, yy);
      ctx.stroke();
      const label = threshold.label || value.toFixed(2);
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(255,255,255,0.86)";
      ctx.fillRect(pad.left + plotW - ctx.measureText(label).width - 11, clamp(yy - 20, pad.top + 2, pad.top + plotH - 18), ctx.measureText(label).width + 10, 16);
      ctx.fillStyle = isEntry ? "#334155" : "#64748b";
      ctx.fillText(label, pad.left + plotW - ctx.measureText(label).width - 6, clamp(yy - 9, pad.top + 12, pad.top + plotH - 6));
    }
    ctx.setLineDash([]);
  }

  for (const s of series) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 0.4;
    ctx.setLineDash(s.dash ? [6, 5] : []);
    ctx.beginPath();
    let started = false;
    for (const p of data) {
      const v = p[s.key];
      if (v === null || v === undefined || Number.isNaN(v)) { started = false; continue; }
      const xx = x(p.ts), yy = y(v);
      if (!started) { ctx.moveTo(xx, yy); started = true; }
      else ctx.lineTo(xx, yy);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  if (options.trades) {
    for (const t of options.trades) {
      const xx = x(t.ts);
      const isOpen = t.action !== "close";
      ctx.fillStyle = isOpen ? "#16a34a" : "#dc2626";
      ctx.globalAlpha = 0.72;
      ctx.beginPath();
      const point = nearestPoint(data, t.ts);
      const value = point && point[options.valueKey || series[0].key];
      if (value !== null && value !== undefined && !Number.isNaN(value)) {
        const yy = y(Number(value));
        if (isOpen) {
          ctx.moveTo(xx, yy - 6);
          ctx.lineTo(xx - 4, yy + 3);
          ctx.lineTo(xx + 4, yy + 3);
        } else {
          ctx.moveTo(xx, yy + 6);
          ctx.lineTo(xx - 4, yy - 3);
          ctx.lineTo(xx + 4, yy - 3);
        }
        ctx.closePath();
        ctx.fill();
      } else if (isOpen) {
        const yy = pad.top + plotH - 7;
        ctx.moveTo(xx, yy - 6); ctx.lineTo(xx - 4, yy + 3); ctx.lineTo(xx + 4, yy + 3);
        ctx.closePath(); ctx.fill();
      } else {
        const yy = pad.top + 7;
        ctx.moveTo(xx, yy + 6); ctx.lineTo(xx - 4, yy - 3); ctx.lineTo(xx + 4, yy - 3);
        ctx.closePath(); ctx.fill();
      }
      ctx.globalAlpha = 1;
    }
  }
  ctx.textAlign = "left";
}

function drawDualAxisIndependentChart(canvasId, data, series, options = {}) {
  // 左右轴各自独立缩放（用于原始价格：两侧价格量级可能相差巨大，如 BTC vs MSTR）
  const canvas = document.getElementById(canvasId);
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const pad = { left: 62, right: 72, top: 14, bottom: 34 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;
  const valid = value => value !== null && value !== undefined && !Number.isNaN(value);

  const leftVals = [], rightVals = [];
  for (const p of data) {
    if (valid(p[series.left.key])) leftVals.push(Number(p[series.left.key]));
    if (valid(p[series.right.key])) rightVals.push(Number(p[series.right.key]));
  }
  if (!data.length || !leftVals.length || !rightVals.length) {
    ctx.fillStyle = "#667085";
    ctx.font = "13px Microsoft YaHei, Arial";
    ctx.fillText("没有数据", pad.left, pad.top + 20);
    return;
  }
  let lMin = Infinity, lMax = -Infinity;
  for (const v of leftVals) { if (v < lMin) lMin = v; if (v > lMax) lMax = v; }
  let rMin = Infinity, rMax = -Infinity;
  for (const v of rightVals) { if (v < rMin) rMin = v; if (v > rMax) rMax = v; }
  if (lMin === lMax) { lMin -= 1; lMax += 1; }
  if (rMin === rMax) { rMin -= 1; rMax += 1; }
  const lPad = (lMax - lMin) * 0.08;
  const rPad = (rMax - rMin) * 0.08;
  lMin -= lPad; lMax += lPad;
  rMin -= rPad; rMax += rPad;

  const x = ts => pad.left + (ts - options.viewStart) / (options.viewEnd - options.viewStart) * plotW;
  const yLeft = v => pad.top + (lMax - v) / (lMax - lMin) * plotH;
  const yRight = v => pad.top + (rMax - v) / (rMax - rMin) * plotH;

  ctx.strokeStyle = "#d7dde8";
  ctx.lineWidth = 1;
  ctx.strokeRect(pad.left, pad.top, plotW, plotH);
  ctx.font = "12px Microsoft YaHei, Arial";
  // 左右各 4 格刻度；刻度位置用 yLeft/yRight 计算，与曲线方向一致（价格越高越靠上）
  for (let i = 0; i <= 4; i++) {
    const lv = lMin + (lMax - lMin) * i / 4;
    const rv = rMin + (rMax - rMin) * i / 4;
    const yyLeft = yLeft(lv);
    const yyRight = yRight(rv);
    ctx.strokeStyle = "#eef1f6";
    ctx.beginPath();
    ctx.moveTo(pad.left, yyLeft);
    ctx.lineTo(pad.left + plotW, yyLeft);
    ctx.stroke();
    ctx.fillStyle = "#667085";
    ctx.textAlign = "left";
    ctx.fillText(options.leftFormat ? options.leftFormat(lv) : lv.toFixed(4), 6, yyLeft + 4);
    ctx.textAlign = "right";
    ctx.fillText(options.rightFormat ? options.rightFormat(rv) : rv.toFixed(4), w - 6, yyRight + 4);
  }
  ctx.textAlign = "left";
  for (const tick of [options.viewStart, (options.viewStart + options.viewEnd) / 2, options.viewEnd]) {
    ctx.fillStyle = "#667085";
    ctx.fillText(dt(tick).slice(5), x(tick) - 40, pad.top + plotH + 22);
  }

  const drawSeries = (item, yFn) => {
    ctx.strokeStyle = item.color;
    ctx.lineWidth = 0.4;
    ctx.beginPath();
    let started = false;
    for (const p of data) {
      const value = p[item.key];
      if (!valid(value)) { started = false; continue; }
      const xx = x(p.ts);
      const yy = yFn(Number(value));
      if (!started) { ctx.moveTo(xx, yy); started = true; }
      else ctx.lineTo(xx, yy);
    }
    ctx.stroke();
  };
  drawSeries(series.left, yLeft);
  drawSeries(series.right, yRight);

  if (options.trades) {
    for (const t of options.trades) {
      const xx = x(t.ts);
      const isOpen = t.action !== "close";
      ctx.fillStyle = isOpen ? "#16a34a" : "#dc2626";
      ctx.globalAlpha = 0.72;
      ctx.beginPath();
      const point = nearestPoint(data, t.ts);
      const values = [
        point && point[series.left.key],
        point && point[series.right.key],
      ];
      const yFns = [yLeft, yRight];
      values.forEach((value, index) => {
        if (!valid(value)) return;
        const yy = yFns[index](Number(value));
        if (isOpen) {
          ctx.moveTo(xx, yy - 6); ctx.lineTo(xx - 4, yy + 3); ctx.lineTo(xx + 4, yy + 3);
        } else {
          ctx.moveTo(xx, yy + 6); ctx.lineTo(xx - 4, yy - 3); ctx.lineTo(xx + 4, yy - 3);
        }
        ctx.closePath();
        ctx.fill();
      });
      ctx.globalAlpha = 1;
    }
  }
  ctx.textAlign = "left";
}

function drawDualAxisAnchoredLogChart(canvasId, data, series, options = {}) {
  const canvas = document.getElementById(canvasId);
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const pad = { left: 62, right: 72, top: 14, bottom: 34 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;
  const valid = value => value !== null && value !== undefined && !Number.isNaN(value);
  const anchor = data.find(p => valid(p[series.left.key]) && valid(p[series.right.key]));

  if (!data.length || !anchor) {
    ctx.fillStyle = "#667085";
    ctx.font = "13px Microsoft YaHei, Arial";
    ctx.fillText("没有数据", pad.left, pad.top + 20);
    return;
  }
  const leftBase = Number(anchor[series.left.key]);
  const rightBase = Number(anchor[series.right.key]);
  const deltas = [0];
  for (const p of data) {
    if (valid(p[series.left.key])) deltas.push(Number(p[series.left.key]) - leftBase);
    if (valid(p[series.right.key])) deltas.push(Number(p[series.right.key]) - rightBase);
  }
  let deltaMin = Infinity, deltaMax = -Infinity;
  for (const v of deltas) {
    if (v < deltaMin) deltaMin = v;
    if (v > deltaMax) deltaMax = v;
  }
  if (deltaMin === deltaMax) { deltaMin -= 0.01; deltaMax += 0.01; }
  const deltaPad = (deltaMax - deltaMin) * 0.08;
  deltaMin -= deltaPad; deltaMax += deltaPad;

  const x = ts => pad.left + (ts - options.viewStart) / (options.viewEnd - options.viewStart) * plotW;
  const y = delta => pad.top + (deltaMax - delta) / (deltaMax - deltaMin) * plotH;

  ctx.strokeStyle = "#d7dde8";
  ctx.lineWidth = 1;
  ctx.strokeRect(pad.left, pad.top, plotW, plotH);
  ctx.font = "12px Microsoft YaHei, Arial";
  for (let i = 0; i <= 4; i++) {
    const delta = deltaMin + (deltaMax - deltaMin) * i / 4;
    const yy = y(delta);
    ctx.strokeStyle = "#eef1f6";
    ctx.beginPath();
    ctx.moveTo(pad.left, yy);
    ctx.lineTo(pad.left + plotW, yy);
    ctx.stroke();
    ctx.fillStyle = "#667085";
    ctx.textAlign = "left";
    ctx.fillText((leftBase + delta).toFixed(4), 6, yy + 4);
    ctx.textAlign = "right";
    ctx.fillText((rightBase + delta).toFixed(4), w - 6, yy + 4);
  }
  const startY = y(0);
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = "#94a3b8";
  ctx.beginPath();
  ctx.moveTo(pad.left, startY);
  ctx.lineTo(pad.left + plotW, startY);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.textAlign = "left";
  for (const tick of [options.viewStart, (options.viewStart + options.viewEnd) / 2, options.viewEnd]) {
    ctx.fillStyle = "#667085";
    ctx.fillText(dt(tick).slice(5), x(tick) - 40, pad.top + plotH + 22);
  }

  const drawSeries = (item, base) => {
    ctx.strokeStyle = item.color;
    ctx.lineWidth = 0.4;
    ctx.beginPath();
    let started = false;
    for (const p of data) {
      const value = p[item.key];
      if (!valid(value)) { started = false; continue; }
      const xx = x(p.ts);
      const yy = y(Number(value) - base);
      if (!started) { ctx.moveTo(xx, yy); started = true; }
      else ctx.lineTo(xx, yy);
    }
    ctx.stroke();
  };
  drawSeries(series.left, leftBase);
  drawSeries(series.right, rightBase);

  if (options.trades) {
    for (const t of options.trades) {
      const xx = x(t.ts);
      const isOpen = t.action !== "close";
      ctx.fillStyle = isOpen ? "#16a34a" : "#dc2626";
      ctx.globalAlpha = 0.72;
      ctx.beginPath();
      const point = nearestPoint(data, t.ts);
      const values = [point && point[series.left.key], point && point[series.right.key]];
      values.forEach((value, index) => {
        if (!valid(value)) return;
        const yy = y(Number(value) - (index === 0 ? leftBase : rightBase));
        if (isOpen) {
          ctx.moveTo(xx, yy - 6); ctx.lineTo(xx - 4, yy + 3); ctx.lineTo(xx + 4, yy + 3);
        } else {
          ctx.moveTo(xx, yy + 6); ctx.lineTo(xx - 4, yy - 3); ctx.lineTo(xx + 4, yy - 3);
        }
        ctx.closePath();
        ctx.fill();
      });
      ctx.globalAlpha = 1;
    }
  }
  ctx.textAlign = "left";
}

function drawEquityDrawdownChart(canvasId, data, series, options = {}) {
  // 双面板：上 70% 权益/贡献曲线，下 30% 回撤填充（回撤 = value/peak - 1）
  const canvas = document.getElementById(canvasId);
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const pad = { left: 62, right: 72, top: 14, bottom: 34 };
  const plotW = w - pad.left - pad.right;
  const splitY = pad.top + (h - pad.top - pad.bottom) * 0.70;
  const plotH = h - pad.top - pad.bottom;

  const valid = value => value !== null && value !== undefined && !Number.isNaN(value);
  const key = series[0].key;
  const values = [];
  for (const p of data) {
    if (valid(p[key])) values.push(Number(p[key]));
  }
  if (!data.length || !values.length) {
    ctx.fillStyle = "#667085";
    ctx.font = "13px Microsoft YaHei, Arial";
    ctx.fillText("没有数据", pad.left, pad.top + 20);
    return;
  }
  let vMin = Infinity, vMax = -Infinity;
  for (const v of values) {
    if (v < vMin) vMin = v;
    if (v > vMax) vMax = v;
  }
  if (vMin === vMax) { vMin -= 1; vMax += 1; }
  const vPad = (vMax - vMin) * 0.08;
  vMin -= vPad; vMax += vPad;

  const x = ts => pad.left + (ts - options.viewStart) / (options.viewEnd - options.viewStart) * plotW;
  const yValue = v => pad.top + (vMax - v) / (vMax - vMin) * (splitY - pad.top);

  // 回撤：优先用后端提供的全局累计峰值（peak）计算 dd = value/peak - 1，
  // 与顶部指标（全局最大回撤）口径一致；无 peak 字段时回退为窗口内重算峰值
  const ddSeries = [];
  let localPeak = null;
  for (const p of data) {
    let dd = null;
    if (valid(p[key])) {
      const v = Number(p[key]);
      if (p.peak !== null && p.peak !== undefined && p.peak > 0) {
        dd = v / Number(p.peak) - 1;
      } else {
        localPeak = localPeak === null ? v : Math.max(localPeak, v);
        dd = localPeak > 0 ? v / localPeak - 1 : 0;
      }
    }
    ddSeries.push(dd);
  }
  let ddMin = 0;
  for (const dd of ddSeries) {
    if (dd !== null && dd < ddMin) ddMin = dd;
  }
  if (ddMin > -0.001) ddMin = -0.001;
  const yDd = dd => (splitY + 6) + (0 - dd) / (0 - ddMin) * (pad.top + plotH - splitY - 10);

  ctx.strokeStyle = "#d7dde8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top + plotH);
  ctx.stroke();
  // 分隔线
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = "#cbd5e1";
  ctx.beginPath();
  ctx.moveTo(pad.left, splitY);
  ctx.lineTo(pad.left + plotW, splitY);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.font = "12px Microsoft YaHei, Arial";
  ctx.fillStyle = "#667085";
  ctx.textAlign = "left";
  for (let i = 0; i <= 4; i++) {
    const value = vMin + (vMax - vMin) * i / 4;
    const yy = yValue(value);
    ctx.strokeStyle = "#eef1f6";
    ctx.beginPath();
    ctx.moveTo(pad.left, yy);
    ctx.lineTo(pad.left + plotW, yy);
    ctx.stroke();
    ctx.fillStyle = "#667085";
    // 最低刻度紧邻回撤区，文字画在刻度线上方，避免与回撤区 0% 标签重叠
    const labelY = i === 4 ? yy - 4 : yy + 4;
    ctx.fillText(options.yFormat ? options.yFormat(value) : value.toFixed(2), 6, labelY);
  }
  // 回撤区网格（0 与最低点）
  ctx.fillStyle = "#b91c1c";
  ctx.fillText("0%", 6, yDd(0) + 12);
  ctx.fillText((ddMin * 100).toFixed(1) + "%", 6, yDd(ddMin) + 4);
  for (const tick of [options.viewStart, (options.viewStart + options.viewEnd) / 2, options.viewEnd]) {
    ctx.fillStyle = "#667085";
    ctx.fillText(dt(tick).slice(5), x(tick) - 40, pad.top + plotH + 22);
  }

  // 权益曲线
  ctx.strokeStyle = series[0].color;
  ctx.lineWidth = 0.4;
  ctx.beginPath();
  let started = false;
  for (const p of data) {
    const v = p[key];
    if (!valid(v)) { started = false; continue; }
    const xx = x(p.ts), yy = yValue(Number(v));
    if (!started) { ctx.moveTo(xx, yy); started = true; }
    else ctx.lineTo(xx, yy);
  }
  ctx.stroke();

  // 回撤填充
  ctx.beginPath();
  ctx.moveTo(x(data[0].ts), yDd(0));
  for (let i = 0; i < data.length; i++) {
    const dd = ddSeries[i];
    if (dd === null) continue;
    ctx.lineTo(x(data[i].ts), yDd(dd));
  }
  ctx.lineTo(x(data[data.length - 1].ts), yDd(0));
  ctx.closePath();
  ctx.fillStyle = "rgba(220, 38, 38, 0.28)";
  ctx.fill();

  if (options.trades) {
    for (const t of options.trades) {
      const xx = x(t.ts);
      const isOpen = t.action !== "close";
      ctx.fillStyle = isOpen ? "#16a34a" : "#dc2626";
      ctx.globalAlpha = 0.72;
      const point = nearestPoint(data, t.ts);
      const value = point && point[key];
      let yy = null;
      if (valid(value)) yy = yValue(Number(value));
      ctx.beginPath();
      if (yy !== null) {
        if (isOpen) {
          ctx.moveTo(xx, yy - 6); ctx.lineTo(xx - 4, yy + 3); ctx.lineTo(xx + 4, yy + 3);
        } else {
          ctx.moveTo(xx, yy + 6); ctx.lineTo(xx - 4, yy - 3); ctx.lineTo(xx + 4, yy - 3);
        }
        ctx.closePath();
        ctx.fill();
      } else if (isOpen) {
        const yy2 = splitY - 7;
        ctx.moveTo(xx, yy2 - 6); ctx.lineTo(xx - 4, yy2 + 3); ctx.lineTo(xx + 4, yy2 + 3);
        ctx.closePath(); ctx.fill();
      } else {
        const yy2 = splitY + 7;
        ctx.moveTo(xx, yy2 + 6); ctx.lineTo(xx - 4, yy2 - 3); ctx.lineTo(xx + 4, yy2 - 3);
        ctx.closePath(); ctx.fill();
      }
      ctx.globalAlpha = 1;
    }
  }
  ctx.textAlign = "left";
}

function drawHoldingDistribution(bins, hintEl) {
  const canvas = document.getElementById("holdingChart");
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  if (!bins || !bins.length) {
    ctx.fillStyle = "#667085";
    ctx.fillText("没有数据", 62, 30);
    if (hintEl) hintEl.textContent = "";
    return;
  }
  const n = bins.length;
  const pad = { left: 46, right: 46, top: 46, bottom: 42 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;
  const maxPct = Math.max(10, Math.ceil(Math.max(...bins.map(b => b.pct)) / 10) * 10);
  const x = i => pad.left + (i + 0.5) * plotW / n;
  const yLeft = v => pad.top + (maxPct - v) / maxPct * plotH;
  const yRight = v => pad.top + (100 - v) / 100 * plotH;
  const barW = plotW / n * 0.62;
  ctx.font = "11px Microsoft YaHei, Arial";
  ctx.strokeStyle = "#eef1f6";
  ctx.fillStyle = "#667085";
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const value = maxPct * i / 4;
    const yy = yLeft(value);
    ctx.beginPath();
    ctx.moveTo(pad.left, yy);
    ctx.lineTo(pad.left + plotW, yy);
    ctx.stroke();
    ctx.fillText(value.toFixed(0) + "%", pad.left - 5, yy + 4);
    ctx.fillStyle = "#b45309";
    ctx.fillText((i * 25).toFixed(0) + "%", pad.left + plotW + 41, yy + 4);
    ctx.fillStyle = "#667085";
  }
  ctx.fillStyle = "rgba(37, 99, 235, 0.78)";
  for (let i = 0; i < n; i++) {
    const b = bins[i];
    const cx = x(i);
    if (b.count > 0) {
      const barTop = yLeft(b.pct);
      ctx.fillRect(cx - barW / 2, barTop, barW, yLeft(0) - barTop);
      ctx.fillStyle = "#1e293b";
      ctx.textAlign = "center";
      if (barTop > pad.top + 38) {
        ctx.fillText(b.count + "笔", cx, barTop - 6);
        ctx.fillText(b.pct.toFixed(1) + "%", cx, barTop - 18);
      } else {
        ctx.fillStyle = "#ffffff";
        ctx.fillText(b.count + "笔", cx, barTop + 13);
      }
      ctx.fillStyle = "#667085";
    }
    ctx.textAlign = "center";
    ctx.fillText(b.label, cx, pad.top + plotH + 16);
  }
}

// ---------- 时间窗口控件（滑块） ----------
function initRangeControls(opts) {
  const leftInput = document.getElementById("windowLeft");
  const rightInput = document.getElementById("windowRight");
  const leftText = document.getElementById("leftBoundaryText");
  const rightText = document.getElementById("rightBoundaryText");
  const minWindowMs = 60 * 60 * 1000;
  const { tsMin, tsMax } = opts;
  const totalMs = Math.max(1, tsMax - tsMin);
  // 滑块按分钟映射：每格 1 分钟（全年约 52.6 万格），可用方向键逐分钟微调
  const sliderMax = Math.max(1000, Math.ceil(totalMs / 60000));
  leftInput.max = sliderMax;
  rightInput.max = sliderMax;
  const sliderStepMs = totalMs / sliderMax;
  const minSteps = Math.max(1, Math.ceil(minWindowMs / sliderStepMs));
  const initial = opts.initial;
  const state = {
    viewStart: initial && initial.start !== null && initial.start !== undefined ? clamp(initial.start, tsMin, tsMax) : tsMin,
    viewEnd: initial && initial.end !== null && initial.end !== undefined
      ? clamp(initial.end, tsMin, tsMax)
      : Math.min(tsMax, tsMin + 14 * 24 * 60 * 60 * 1000)
  };
  // 初始窗口（用于“重置”按钮恢复；Pair 页为默认交易窗口）
  const initialStart = initial && initial.start !== null && initial.start !== undefined ? clamp(initial.start, tsMin, tsMax) : tsMin;
  const initialEnd = initial && initial.end !== null && initial.end !== undefined
    ? clamp(initial.end, tsMin, tsMax)
    : Math.min(tsMax, tsMin + 14 * 24 * 60 * 60 * 1000);

  function tsToSlider(ts) { return Math.round((ts - tsMin) / totalMs * sliderMax); }
  function sliderToTs(value) { return tsMin + Number(value) / sliderMax * totalMs; }
  function clampView(start, end) {
    const minSize = Math.min(minWindowMs, totalMs);
    if (end - start < minSize) end = start + minSize;
    if (start < tsMin) { end += tsMin - start; start = tsMin; }
    if (end > tsMax) { start -= end - tsMax; end = tsMax; }
    return [clamp(start, tsMin, tsMax), clamp(end, tsMin, tsMax)];
  }
  function syncInputs() {
    leftInput.value = tsToSlider(state.viewStart);
    rightInput.value = tsToSlider(state.viewEnd);
    leftText.textContent = dt(state.viewStart);
    rightText.textContent = dt(state.viewEnd);
  }
  function setView(start, end) {
    [state.viewStart, state.viewEnd] = clampView(start, end);
    state.viewStart = sliderToTs(tsToSlider(state.viewStart));
    state.viewEnd = sliderToTs(tsToSlider(state.viewEnd));
    [state.viewStart, state.viewEnd] = clampView(state.viewStart, state.viewEnd);
    syncInputs();
    opts.onChange(state.viewStart, state.viewEnd);
  }
  function updateFromInputs(changedSide, commit) {
    let leftValue = Number(leftInput.value);
    let rightValue = Number(rightInput.value);
    const originalLeftValue = leftValue;
    const originalRightValue = rightValue;
    if (!commit) {
      // 预览：只更新本侧日期文字，不移动对侧滑块，也不钳制窗口
      const previewTs = sliderToTs(changedSide === "left" ? leftValue : rightValue);
      if (changedSide === "left") leftText.textContent = dt(previewTs);
      else rightText.textContent = dt(previewTs);
      return;
    }
    // 松手提交：完整钳制（窗口最小 1 小时）并触发重绘
    if (rightValue - leftValue < minSteps) {
      if (changedSide === "left") leftValue = rightValue - minSteps;
      else rightValue = leftValue + minSteps;
    }
    if (leftValue < 0) { rightValue -= leftValue; leftValue = 0; }
    if (rightValue > sliderMax) { leftValue -= rightValue - sliderMax; rightValue = sliderMax; }
    leftValue = clamp(leftValue, 0, sliderMax);
    rightValue = clamp(rightValue, 0, sliderMax);
    leftInput.value = leftValue;
    rightInput.value = rightValue;
    const nextStart = changedSide === "left" || leftValue !== originalLeftValue
      ? sliderToTs(leftValue)
      : state.viewStart;
    const nextEnd = changedSide === "right" || rightValue !== originalRightValue
      ? sliderToTs(rightValue)
      : state.viewEnd;
    setView(nextStart, nextEnd);
  }
  const bindRange = (input, side) => {
    input.addEventListener("input", () => updateFromInputs(side, false));
    input.addEventListener("change", () => updateFromInputs(side, true));
  };
  bindRange(leftInput, "left");
  bindRange(rightInput, "right");
  document.getElementById("fullBtn")?.addEventListener("click", () => setView(tsMin, tsMax));
  document.getElementById("resetZoom")?.addEventListener("click", () => setView(initialStart, initialEnd));
  syncInputs();
  state.setView = setView;
  return state;
}

// ---------- rAF 合并渲染 ----------
function makeScheduler(renderFn) {
  let queued = false;
  return function scheduleRender() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; renderFn(); });
  };
}
"""


# ---------------------------------------------------------------------------
# 组合总览页模板
# ---------------------------------------------------------------------------
OVERVIEW_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__ - 组合总览</title>
  <link rel="stylesheet" href="trade_review_assets/review.css?v=20260831-2" />
  <script src="trade_review_assets/review.js?v=20260831-2"></script>
</head>
<body>
  <div id="loadError"></div>
  <header>
    <h1>BTC-COIN 回测组合总览</h1>
    <div class="topbar">
      <div class="metrics" id="metrics"></div>
      <div class="buttons">
        <a class="back-link" id="runIndexLink" href="/" hidden>切换回测</a>
        <label class="pair-jump">Pair
          <select id="pairSelect" aria-label="选择 Pair"></select>
        </label>
        <button id="openPairBtn">打开 Pair 分析</button>
        <button id="fullBtn">全区间</button>
        <button id="resetZoom">重置 14 天</button>
        <button class="primary" id="drawdownBtn">跳到最大回撤</button>
        <button id="gainBtn">跳到最高收益段</button>
      </div>
    </div>
  </header>
  <main>
    <section class="chart-stack" style="display:flex;flex-direction:column;gap:14px;">
      <div class="range-panel">
        <label>左边界 <span class="boundary-value" id="leftBoundaryText"></span>
          <input id="windowLeft" type="range" min="0" max="1000000" value="0" /></label>
        <label>右边界 <span class="boundary-value" id="rightBoundaryText"></span>
          <input id="windowRight" type="range" min="0" max="1000000" value="200" /></label>
      </div>
      <div class="panel">
        <h2>组合资金曲线与回撤</h2>
        <div class="legend">
          <span><i class="dot" style="background: var(--blue)"></i>组合权益</span>
          <span><i class="dot" style="background: var(--red)"></i>回撤（右轴）</span>
        </div>
        <canvas class="chart tall" id="equityChart"></canvas>
      </div>
      <div class="panel">
        <h2>持仓时间分布</h2>
        <canvas class="chart" id="holdingChart"></canvas>
        <div class="hint" id="holdingHint"></div>
      </div>
      <div class="panel">
        <h2>Pair 汇总表</h2>
        <div class="hint">收益率和最大回撤率的分母都是组合初始资金。点击任一 Pair 进入该 Pair 的独立复盘页面。</div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>交易对</th><th>交易次数</th><th>已平/未平</th><th>胜率</th><th>盈亏比</th>
                <th>累计净盈亏</th><th>收益贡献</th><th>最大回撤</th>
                <th>平均持仓</th><th>中位持仓</th><th>最长持仓</th>
                <th>手续费</th><th>滑点</th><th>资金费率</th>
              </tr>
            </thead>
            <tbody id="pairSummaryBody"></tbody>
          </table>
        </div>
      </div>
    </section>
    <aside class="side">
      <div class="panel">
        <h2>当前窗口</h2>
        <div class="summary" id="windowSummary"></div>
      </div>
    </aside>
  </main>
  <script>
    const payload = __PAYLOAD__;
    const points = unpackSeries(payload.points, ["equity", "gross_exposure_ratio", "net_exposure_ratio", "peak"]);
    const state = initRangeControls({
      tsMin: payload.range.start,
      tsMax: payload.range.end,
      onChange: () => scheduleRender()
    });
    const scheduleRender = makeScheduler(renderAll);
    const metricsEl = document.getElementById("metrics");
    const windowSummary = document.getElementById("windowSummary");
    const pairSummaryBody = document.getElementById("pairSummaryBody");
    const holdingDist = payload.holding_distribution || null;
    const pairRows = payload.pairs || [];

    function initPairJump() {
      const select = document.getElementById("pairSelect");
      if (!select) return;
      select.innerHTML = pairRows.map(row =>
        `<option value="${esc(row.pair_id)}">${esc(row.pair_id)}</option>`
      ).join("");
      document.getElementById("openPairBtn")?.addEventListener("click", () => {
        const pairId = select.value;
        if (pairId) window.location.href = "trade_review_pairs/" + encodeURIComponent(pairId) + ".html";
      });
    }

    function renderMetrics() {
      const m = payload.metrics;
      const rows = [
        ["初始权益", num(m.initial_equity, 0)],
        ["最终权益", num(m.final_equity, 0)],
        ["总收益", pct(m.total_return)],
        ["CAGR", pct(m.cagr)],
        ["Sharpe", num(m.sharpe, 3)],
        ["Calmar", num(m.calmar, 3)],
        ["最大回撤", pct(m.max_drawdown)],
        ["交易笔数", num(m.trade_count, 0)],
        ["胜率", pct(m.win_rate)],
        ["盈亏比", num(m.profit_loss_ratio, 2)],
        ["日均换手", num(m.daily_turnover, 2)],
        ["平均持仓", fmtDuration(m.average_holding_minutes)],
        ["总手续费", num(m.total_fee, 0)],
        ["总滑点", num(m.total_slippage, 0)],
        ["资金费率", num(m.funding_fee, 0)]
      ];
      metricsEl.innerHTML = rows.map(([k, v]) => `<span class="metric">${k}<b>${v}</b></span>`).join("");
    }

    function visiblePoints() {
      return points.slice(lowerBound(points, state.viewStart), upperBound(points, state.viewEnd));
    }

    function renderAll() {
      const data = visiblePoints();
      const equities = data.map(p => p.equity).filter(v => v !== null && v !== undefined);
      if (data.length && equities.length) {
        const [mn, mx] = minMax(equities);
        windowSummary.innerHTML = [
          ["窗口收益", pct(data[data.length - 1].equity / data[0].equity - 1)],
          ["权益最低/最高", num(mn, 0) + " / " + num(mx, 0)],
          ["点数", num(data.length, 0)]
        ].map(([k, v]) => `<div class="summary-row"><span>${k}</span><b>${v}</b></div>`).join("");
      } else {
        windowSummary.innerHTML = "";
      }
      const withDrawdown = [];
      let peak = null;
      for (const p of data) {
        let dd = null;
        if (p.equity !== null && p.equity !== undefined) {
          peak = peak === null ? p.equity : Math.max(peak, p.equity);
          dd = peak > 0 ? p.equity / peak - 1 : 0;
        }
        withDrawdown.push({ ...p, drawdown: dd });
      }
      drawEquityDrawdownChart("equityChart", withDrawdown, [
        { key: "equity", color: "#2563eb", label: "权益" }
      ], {
        viewStart: state.viewStart,
        viewEnd: state.viewEnd,
        yFormat: v => num(v, 0),
        trades: [],
        valueKey: "equity"
      });
      drawHoldingDistribution(holdingDist ? holdingDist.bins : null, document.getElementById("holdingHint"));
      renderPairSummaryTable();
    }

    function renderPairSummaryTable() {
      if (!pairRows.length) {
        pairSummaryBody.innerHTML = `<tr><td colspan="14">没有交易对汇总数据</td></tr>`;
        return;
      }
      pairSummaryBody.innerHTML = pairRows.map(row => {
        const href = "trade_review_pairs/" + encodeURIComponent(row.pair_id) + ".html";
        const contribution = row.contribution_to_initial_equity;
        return `
          <tr data-href="${href}">
            <td><a href="${href}">${esc(row.pair_id)}</a></td>
            <td>${num(row.total_positions, 0)}</td>
            <td>${num(row.closed_positions, 0)} / ${num(row.open_positions, 0)}</td>
            <td>${pct(row.win_rate)}</td>
            <td>${num(row.profit_loss_ratio, 2)}</td>
            <td class="${signedClass(row.total_pnl)}">${num(row.total_pnl, 0)}</td>
            <td class="${signedClass(contribution)}">${pct(contribution)}</td>
            <td>${pct(row.max_drawdown)}</td>
            <td>${fmtDuration(row.avg_holding_minutes)}</td>
            <td>${fmtDuration(row.median_holding_minutes)}</td>
            <td>${fmtDuration(row.max_holding_minutes)}</td>
            <td>${num(row.total_fee, 0)}</td>
            <td>${num(row.total_slippage, 0)}</td>
            <td>${num(row.funding_fee, 0)}</td>
          </tr>`;
      }).join("");
    }

    function esc(value) {
      return String(value ?? "").replace(/[&<>"']/g, ch => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[ch]));
    }
    function signedClass(value) {
      if (value === null || value === undefined || Number.isNaN(value)) return "";
      return value > 0 ? "pos" : (value < 0 ? "neg" : "");
    }

    renderMetrics();
    if (window.location.pathname.startsWith("/runs/")) {
      document.getElementById("runIndexLink").hidden = false;
    }
    initPairJump();
    document.getElementById("drawdownBtn").addEventListener("click", () => {
      const w = payload.windows.drawdown;
      if (w && w.start !== null && w.end !== null) setViewExternal(w.start, w.end);
    });
    document.getElementById("gainBtn").addEventListener("click", () => {
      const w = payload.windows.gain;
      if (w && w.start !== null && w.end !== null) setViewExternal(w.start, w.end);
    });
    function setViewExternal(start, end) {
      state.setView(start, end);
    }
    scheduleRender();
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Pair 独立复盘页模板
# ---------------------------------------------------------------------------
PAIR_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__PAIR_ID__ 独立复盘 - __TITLE__</title>
  <link rel="stylesheet" href="../trade_review_assets/review.css?v=20260831-2" />
  <script src="../trade_review_assets/review.js?v=20260901-1"></script>
</head>
<body class="pair-review">
  <div id="loadError"></div>
  <header>
    <h1><span id="pairTitle">__PAIR_ID__</span> 独立复盘</h1>
    <div class="topbar">
      <div class="metrics" id="metrics"></div>
      <div class="buttons">
        <a class="back-link" id="runIndexLink" href="/" hidden>切换回测</a>
        <a class="back-link" href="../trade_review.html">← 返回组合总览</a>
        <button id="fullBtn">全区间</button>
        <button id="resetZoom">重置默认窗口</button>
        <button class="primary" id="drawdownBtn">跳到最大回撤</button>
      </div>
    </div>
  </header>
  <main>
    <section class="chart-stack" style="display:flex;flex-direction:column;gap:14px;">
      <div class="range-panel">
        <label>左边界 <span class="boundary-value" id="leftBoundaryText"></span>
          <input id="windowLeft" type="range" min="0" max="1000000" value="0" /></label>
        <label>右边界 <span class="boundary-value" id="rightBoundaryText"></span>
          <input id="windowRight" type="range" min="0" max="1000000" value="200" /></label>
        <label class="toggle-label"><input id="showTradeMarkers" type="checkbox" checked /> 交易标记</label>
      </div>
      <div class="panel">
        <h2>Pair 贡献资金曲线与回撤</h2>
        <div class="legend">
          <span><i class="dot" style="background: var(--blue)"></i>贡献权益（组合初始权益 + 该 Pair 累计净盈亏）</span>
          <span><i class="dot" style="background: var(--red)"></i>回撤（右轴）</span>
          <span class="hint" style="margin-left:8px;">该曲线是收益贡献曲线，不代表给该 Pair 独立分配了一份完整本金。</span>
        </div>
        <canvas class="chart tall" id="contribChart"></canvas>
      </div>
      <div class="panel">
        <h2><span id="rawPriceTitle">Pair</span> 原始价格双轴走势</h2>
        <div class="legend">
          <span><i class="dot" style="background: var(--blue)"></i><span id="rawXLegend">X</span>（左轴）</span>
          <span><i class="dot" style="background: var(--orange)"></i><span id="rawYLegend">Y</span>（右轴）</span>
        </div>
        <canvas class="chart" id="rawPriceChart"></canvas>
      </div>
      <div class="panel">
        <h2>归一化价格（窗口首个有效价格为 100）</h2>
        <div class="legend">
          <span><i class="dot" style="background: var(--blue)"></i><span id="normXLegend">X</span></span>
          <span><i class="dot" style="background: var(--orange)"></i><span id="normYLegend">Y</span></span>
        </div>
        <canvas class="chart" id="normalizedPriceChart"></canvas>
      </div>
      <div class="panel">
        <h2>Log Price 双轴走势</h2>
        <div class="legend">
          <span><i class="dot" style="background: var(--blue)"></i><span id="logXLegend">X</span>（左轴）</span>
          <span><i class="dot" style="background: var(--orange)"></i><span id="logYLegend">Y</span>（右轴）</span>
        </div>
        <canvas class="chart" id="logPriceChart"></canvas>
      </div>
      <div class="panel">
        <h2><span id="theoreticalYTitle">Y</span> 真实价格与理论价格</h2>
        <div class="legend">
          <span><i class="dot" style="background: var(--orange)"></i><span id="actualYLegend">Y</span> 真实价格</span>
          <span><i class="dot" style="background: var(--blue)"></i><span id="theoreticalYLegend">Y</span> 理论价格</span>
          <span class="hint" id="theoreticalPriceHint">理论价格由当分钟已生效的 Alpha、Beta 和 X 价格计算。</span>
        </div>
        <canvas class="chart" id="theoreticalYChart"></canvas>
      </div>
      <div class="panel">
        <h2>每分钟 Z-score 与交易阈值</h2>
        <div class="legend">
          <span><i class="dot" style="background: var(--purple)"></i>Z-score</span>
          <span><i class="dot" style="background: var(--green)"></i>开仓</span>
          <span><i class="dot" style="background: var(--red)"></i>平仓</span>
        </div>
        <canvas class="chart" id="zscoreChart"></canvas>
      </div>
    </section>
    <aside class="side">
      <div class="panel">
        <h2>当前窗口</h2>
        <div class="summary" id="windowSummary"></div>
      </div>
      <div class="panel">
        <h2>交易卡片</h2>
        <div class="hint">全部交易始终展示；点击卡片把全部图表定位到该笔交易及其训练窗口附近。</div>
        <div class="trade-list" id="tradeList"></div>
      </div>
    </aside>
  </main>
  <script>
    const pairId = "__PAIR_ID__";
    const config = __PAIR_CONFIG__;
    const scheduleRender = makeScheduler(renderAll);
    const state = initRangeControls({
      tsMin: config.range.start,
      tsMax: config.range.end,
      initial: config.default_window || null,
      onChange: () => { ensureMonthsLoaded(state.viewStart, state.viewEnd).then(scheduleRender); }
    });

    const dataRoot = "../trade_review_data/pairs/" + encodeURIComponent(pairId) + "/";
    const monthCache = {};
    let allPoints = [];
    let summary = config.summary || {};
    let selectedTradeId = null;

    function monthKeys(startTs, endTs) {
      const keys = [];
      const d = new Date(startTs);
      const end = new Date(endTs);
      d.setUTCDate(1);
      while (d <= end) {
        keys.push(d.getUTCFullYear() + "-" + String(d.getUTCMonth() + 1).padStart(2, "0"));
        d.setUTCMonth(d.getUTCMonth() + 1);
      }
      return keys;
    }
    async function ensureMonthsLoaded(startTs, endTs) {
      const keys = monthKeys(startTs, endTs);
      const missing = keys.filter(k => !monthCache[k]);
      if (missing.length) {
        if (location.protocol === "file:") {
          showLoadError("Pair 分钟数据需要 HTTP 服务才能加载。请运行：\npython scripts/serve_trade_review.py --run-dir <回测目录> --port 8000");
          return;
        }
        for (const k of missing) {
          try {
            const r = await fetch(dataRoot + k + ".json");
            if (!r.ok) throw new Error("HTTP " + r.status);
            monthCache[k] = await r.json();
          } catch (err) {
            showLoadError("加载 " + k + " 数据失败: " + err.message + "\n请确认已通过 HTTP 服务访问本页面。");
            return;
          }
        }
      }
      // 全量重建：合并所有已缓存的月份，避免跨月拖动时部分月份消失
      const cached = Object.values(monthCache);
      allPoints = cached.length ? unpackMonths(cached) : [];
    }
    function unpackMonths(months) {
      const out = [];
      for (const m of months) {
        out.push(...unpackSeries(m, ["x_close", "y_close", "y_theoretical", "zscore", "pnl", "cum_funding", "peak"]));
      }
      for (const p of out) {
        p.contrib_equity = config.initial_equity + (p.pnl === null || p.pnl === undefined ? 0 : p.pnl);
      }
      return out.sort((a, b) => a.ts - b.ts);
    }
    function visiblePoints() {
      return allPoints.slice(lowerBound(allPoints, state.viewStart), upperBound(allPoints, state.viewEnd));
    }
    function computeWindowDerived(data) {
      // 归一化以窗口内首个有效价格为 100；log 用自然对数
      let firstX = null, firstY = null;
      for (const p of data) {
        if (firstX === null && p.x_close !== null && p.x_close !== undefined && p.x_close > 0) firstX = p.x_close;
        if (firstY === null && p.y_close !== null && p.y_close !== undefined && p.y_close > 0) firstY = p.y_close;
        if (firstX !== null && firstY !== null) break;
      }
      return data.map(p => {
        const xc = p.x_close, yc = p.y_close;
        return {
          ...p,
          x_norm: (xc !== null && xc !== undefined && firstX) ? xc / firstX * 100 : null,
          y_norm: (yc !== null && yc !== undefined && firstY) ? yc / firstY * 100 : null,
          x_log: (xc !== null && xc !== undefined && xc > 0) ? Math.log(xc) : null,
          y_log: (yc !== null && yc !== undefined && yc > 0) ? Math.log(yc) : null
        };
      });
    }
    function visibleTrades() {
      // 按持仓区间与窗口重叠判断：跨窗口持仓（开仓在窗口前、平仓在窗口内/未平仓）也要显示
      const start = state.viewStart, end = state.viewEnd;
      return (config.trades || []).filter(t => {
        const open = t.open_ts !== null && t.open_ts !== undefined ? t.open_ts : t.ts;
        if (open === null || open === undefined) return false;
        const close = t.close_ts !== null && t.close_ts !== undefined ? t.close_ts : end;
        return open <= end && close >= start;
      });
    }
    function visiblePairEvents() {
      const start = state.viewStart, end = state.viewEnd;
      return (config.events || []).filter(t => t.ts >= start && t.ts <= end);
    }
    function renderMetrics() {
      const s = summary;
      const rows = [
        ["X / Y", s.x_symbol + " / " + s.y_symbol],
        ["交易次数", num(s.total_positions, 0)],
        ["已平 / 未平", num(s.closed_positions, 0) + " / " + num(s.open_positions, 0)],
        ["胜率", pct(s.win_rate)],
        ["盈亏比", num(s.profit_loss_ratio, 2)],
        ["累计净盈亏", num(s.total_pnl, 0)],
        ["收益贡献", pct(s.contribution_to_initial_equity)],
        ["最大回撤", pct(s.max_drawdown)],
        ["平均 / 中位 / 最长持仓", fmtDuration(s.avg_holding_minutes) + " / " + fmtDuration(s.median_holding_minutes) + " / " + fmtDuration(s.max_holding_minutes)],
        ["手续费 / 滑点 / 资金费", num(s.total_fee, 0) + " / " + num(s.total_slippage, 0) + " / " + num(s.funding_fee, 0)]
      ];
      document.getElementById("metrics").innerHTML = rows.map(([k, v]) => `<span class="metric">${k}<b>${v}</b></span>`).join("");
      document.getElementById("pairTitle").textContent = pairId + "（" + s.x_symbol + " / " + s.y_symbol + "）";
      document.getElementById("rawPriceTitle").textContent = s.x_symbol + " / " + s.y_symbol;
      ["rawXLegend", "normXLegend", "logXLegend"].forEach(id => document.getElementById(id).textContent = s.x_symbol);
      ["rawYLegend", "normYLegend", "logYLegend"].forEach(id => document.getElementById(id).textContent = s.y_symbol);
      document.getElementById("theoreticalYTitle").textContent = s.y_symbol;
      document.getElementById("actualYLegend").textContent = s.y_symbol;
      document.getElementById("theoreticalYLegend").textContent = s.y_symbol;
      const hint = document.getElementById("theoreticalPriceHint");
      if (config.regression_method === "log_price") {
        hint.textContent = "Log-price 理论价格：exp(Alpha + Beta × log(X))。参数使用当分钟已生效值。";
      } else if (config.regression_method === "price") {
        hint.textContent = "Price 理论价格：Alpha + Beta × X。参数使用当分钟已生效值。";
      } else {
        hint.textContent = "当前模型基于收益率，无法从单点 Alpha、Beta 直接得到理论 Y 价格。";
      }
    }
    function renderAll() {
      const data = computeWindowDerived(visiblePoints());
      const showMarkers = document.getElementById("showTradeMarkers").checked;
      const markers = {
        trades: showMarkers ? visiblePairEvents() : [],
        viewStart: state.viewStart,
        viewEnd: state.viewEnd
      };
      drawEquityDrawdownChart("contribChart", data, [
        { key: "contrib_equity", color: "#2563eb", label: "贡献权益" }
      ], { ...markers, yFormat: v => num(v, 0) });
      drawDualAxisIndependentChart("rawPriceChart", data, {
        left: { key: "x_close", color: "#2563eb" }, right: { key: "y_close", color: "#f97316" }
      }, { ...markers, markerMode: "points" });
      drawDualAxisAnchoredLogChart("normalizedPriceChart", data, {
        left: { key: "x_norm", color: "#2563eb" }, right: { key: "y_norm", color: "#f97316" }
      }, { ...markers, markerMode: "points" });
      drawDualAxisAnchoredLogChart("logPriceChart", data, {
        left: { key: "x_log", color: "#2563eb" }, right: { key: "y_log", color: "#f97316" }
      }, { ...markers, markerMode: "points" });
      drawLineChart("theoreticalYChart", data, [
        { key: "y_close", color: "#f97316", label: "真实价格" },
        { key: "y_theoretical", color: "#2563eb", label: "理论价格", dash: true }
      ], { ...markers, valueKey: "y_close", yFormat: v => num(v, 6) });
      drawLineChart("zscoreChart", data, [
        { key: "zscore", color: "#7c3aed", label: "Z-score" }
      ], { ...markers, thresholds: config.thresholds || [], valueKey: "zscore", yFormat: v => num(v, 2) });
      renderWindowSummary(data);
      renderTradeList();
    }
    function renderWindowSummary(data) {
      const el = document.getElementById("windowSummary");
      if (!data.length) { el.innerHTML = ""; return; }
      const first = data[0], last = data[data.length - 1];
      el.innerHTML = [
        ["窗口贡献收益", pct(last.contrib_equity / first.contrib_equity - 1)],
        ["点数", num(data.length, 0)],
        ["窗口内交易", num(visibleTrades().length, 0)]
      ].map(([k, v]) => `<div class="summary-row"><span>${k}</span><b>${v}</b></div>`).join("");
    }
    function renderTradeScenario(t) {
      const analysis = t.scenario_analysis;
      if (!analysis) return "";
      if (!analysis.available) {
        return `<div class="trade-block"><div class="trade-scenario-title"><b>开仓收益情景</b></div>` +
          `<div class="scenario-unavailable">${esc(analysis.reason || "当前交易无法计算静态价格情景。")}</div></div>`;
      }
      const rows = (analysis.scenarios || []).map(row => `
        <tr>
          <td>${pct(row.x_move)}</td>
          <td title="${num(row.x_price, 10)}">${num(row.x_price, 6)}</td>
          <td title="${num(row.y_target_price, 10)}">${num(row.y_target_price, 6)}</td>
          <td class="${signedClass(row.gross_return)}">${pct(row.gross_return)}</td>
        </tr>`).join("");
      const actual = analysis.actual;
      let actualHtml = "";
      if (actual) {
        const exact = actual.theoretical_at_actual_x;
        const neighbors = [actual.lower_scenario, actual.upper_scenario]
          .filter(Boolean)
          .map(row => `${pct(row.x_move)} → ${pct(row.gross_return)}`)
          .join("；");
        actualHtml = `
          <div class="scenario-actual">
            <b>实际平仓落点</b><br>
            X实际变动 <b>${pct(actual.x_move)}</b>（${esc(actual.scenario_band || "-")}）<br>
            相邻情景收益 ${neighbors || "-"}<br>
            该X位置理论收益 <span class="trade-pnl ${signedClass(exact && exact.gross_return)}">${exact ? pct(exact.gross_return) : "-"}</span><br>
            实际毛收益 <span class="trade-pnl ${signedClass(actual.actual_gross_return)}">${pct(actual.actual_gross_return)}</span>，
            实际净收益 <span class="trade-pnl ${signedClass(actual.actual_net_return)}">${pct(actual.actual_net_return)}</span><br>
            毛收益与理论差 ${pct(actual.gross_return_gap)}
          </div>`;
      }
      return `
        <div class="trade-block">
          <div class="trade-scenario-title">
            <b>开仓收益情景</b>
            <span>假设回到 Z=${num(analysis.target_z, 2)}；不计成本</span>
          </div>
          <div class="table-wrap"><table class="scenario-table">
            <thead><tr><th>X变动</th><th>X平仓价</th><th>理论Y价</th><th>组合收益</th></tr></thead>
            <tbody>${rows}</tbody>
          </table></div>
          ${actualHtml}
        </div>`;
    }
    function renderTradeList() {
      const el = document.getElementById("tradeList");
      const trades = config.trades || [];
      if (!trades.length) {
        el.innerHTML = `<div class="hint">这个 Pair 没有交易。</div>`;
        return;
      }
      el.innerHTML = trades.map(t => {
        const closed = t.close_ts !== null && t.close_ts !== undefined;
        const tradeId = t.trade_id || `${t.pair_id || pairId}:${t.open_ts || t.ts}`;
        const isSelected = selectedTradeId === tradeId;
        return `
        <div class="trade-card${isSelected ? " selected" : ""}" data-trade-id="${esc(tradeId)}" data-ts="${t.ts}">
          <div class="trade-title"><b>${esc(t.pair_id || pairId)}</b><span class="${isSelected ? "selected-label" : ""}">${isSelected ? "当前查看 · " : ""}${closed ? "已平仓" : "未平仓"}</span></div>
          <div>${esc(t.direction || "")}</div>
          <div class="trade-block"><b class="open">开仓</b> ${esc(t.open_time || "-")}<br>
            信号 Z-score ${num(t.open_zscore, 2)}，成交分钟 Z-score ${num(t.open_fill_zscore, 2)}<br>
            Alpha ${num(t.open_alpha, 6)}，Beta ${num(t.open_beta, 4)}<br>
            残差均值 ${num(t.open_spread_mean, 6)}，标准差 ${num(t.open_spread_std, 6)}<br>
            原始价格 ${esc(summary.x_symbol || "X")} ${num(t.open_x_price, 8)} / ${esc(summary.y_symbol || "Y")} ${num(t.open_y_price, 8)}<br>
            原始价差（Y-X） ${num(t.open_raw_spread, 8)}<br>
            两腿：${esc(t.open_legs || "-")}
          </div>
          ${renderTradeScenario(t)}
          <div class="trade-block"><b class="close">平仓</b> ${esc(t.close_time || "未平仓")}<br>
            信号 Z-score ${num(t.close_zscore, 2)}，成交分钟 Z-score ${num(t.close_fill_zscore, 2)}<br>
            Alpha ${num(t.close_alpha, 6)}，Beta ${num(t.close_beta, 4)}<br>
            原始价格 ${esc(summary.x_symbol || "X")} ${num(t.close_x_price, 8)} / ${esc(summary.y_symbol || "Y")} ${num(t.close_y_price, 8)}<br>
            原始价差（Y-X） ${num(t.close_raw_spread, 8)}<br>
            两腿：${esc(t.close_legs || "-")}
          </div>
          <div class="trade-block">持仓 ${fmtDuration(t.holding_minutes)}，对冲比例 ${num(t.target_hedge_ratio, 3)}<br>
            不计成本收益 <span class="trade-pnl ${signedClass(t.gross_pnl)}">${num(t.gross_pnl, 2)}</span>（${pct(t.gross_return)}）<br>
            手续费 ${num(t.fee, 2)}，滑点 ${num(t.slippage, 2)}，资金费率 ${num(t.funding_fee, 2)}<br>
            实际净收益 <span class="trade-pnl ${signedClass(t.net_pnl)}">${num(t.net_pnl, 2)}</span>（${pct(t.net_return)}）
          </div>
        </div>`;
      }).join("");
      el.querySelectorAll(".trade-card").forEach(card => {
        card.addEventListener("click", () => {
          selectedTradeId = card.dataset.tradeId;
          renderTradeList();
          const trade = (config.trades || []).find(item => (item.trade_id || `${item.pair_id || pairId}:${item.open_ts || item.ts}`) === selectedTradeId);
          if (!trade) return;
          const contextMs = Math.max(0, Number(config.model_lookback_bars || 0)) * 60 * 1000;
          const openTs = Number(trade.open_ts ?? trade.ts);
          const closeTs = trade.close_ts !== null && trade.close_ts !== undefined
            ? Number(trade.close_ts)
            : Number(config.range.end);
          setPairView(openTs - contextMs, closeTs);
        });
      });
    }
    function setPairView(start, end) {
      state.setView(start, end);
    }
    document.getElementById("drawdownBtn").addEventListener("click", () => {
      const w = summary.max_drawdown_window;
      if (w && w.start !== null) setPairView(w.start, w.end);
    });
    function esc(value) {
      return String(value ?? "").replace(/[&<>"']/g, ch => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[ch]));
    }
    function signedClass(value) {
      if (value === null || value === undefined || Number.isNaN(value)) return "";
      return value > 0 ? "pos" : (value < 0 ? "neg" : "");
    }
    if (window.location.pathname.startsWith("/runs/")) {
      document.getElementById("runIndexLink").hidden = false;
    }
    renderMetrics();
    document.getElementById("showTradeMarkers").addEventListener("change", scheduleRender);
    ensureMonthsLoaded(state.viewStart, state.viewEnd).then(scheduleRender);
  </script>
</body>
</html>
"""
