from __future__ import annotations

from dataclasses import asdict, is_dataclass
from math import isfinite
from pathlib import Path
import json


def export_equity_summary_html(result, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": "Equity Curve",
        "metrics": _summary_metrics(getattr(result, "metrics", {}) or {}),
        "equity": _equity_points(getattr(result, "equity_curve", [])),
    }
    output_path.write_text(render_equity_summary_html(payload), encoding="utf-8")
    return output_path


def render_equity_summary_html(payload: dict) -> str:
    data = json.dumps(_json_safe(payload), ensure_ascii=False, separators=(",", ":"))
    return HTML_TEMPLATE.replace("__PAYLOAD__", data)


def _equity_points(rows) -> list[dict]:
    points = []
    peak = None
    for row in _rows(rows):
        ts = _safe_int(row.get("ts"))
        equity = _safe_float(row.get("equity"))
        if ts is None or equity is None:
            continue
        peak = equity if peak is None else max(peak, equity)
        points.append({"ts": ts, "equity": equity, "peak": peak})
    return sorted(points, key=lambda item: item["ts"])


def _summary_metrics(metrics: dict) -> list[dict]:
    items = [
        ("Final Equity", "final_equity", "money"),
        ("Total Return", "total_return", "percent"),
        ("Sharpe", "sharpe", "number"),
        ("Max Drawdown", "max_drawdown", "percent"),
        ("Trades", "trade_count", "integer"),
        ("Win Rate", "win_rate", "percent"),
        ("Total Fee", "total_fee", "money"),
        ("Total Slippage", "total_slippage", "money"),
    ]
    result = []
    for label, key, kind in items:
        value = metrics.get(key)
        if value is None:
            continue
        result.append({"label": label, "value": value, "kind": kind})
    return result


def _rows(rows) -> list[dict]:
    result = []
    for row in rows or []:
        if is_dataclass(row):
            result.append(asdict(row))
        elif isinstance(row, dict):
            result.append(row)
        else:
            result.append(vars(row))
    return result


def _safe_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_safe(value):
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if isfinite(value) else None
    if hasattr(value, "value"):
        return value.value
    return value


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Equity Curve</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d7dde8;
      --grid: #e7ebf2;
      --blue: #2563eb;
      --red: #dc2626;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, "Segoe UI", Arial, sans-serif;
    }
    .wrap {
      width: min(1320px, calc(100vw - 32px));
      margin: 20px auto 28px;
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-end;
      margin-bottom: 12px;
    }
    h1 {
      margin: 0;
      font-size: 24px;
      font-weight: 720;
      letter-spacing: 0;
    }
    .hint {
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }
    .metric b {
      font-size: 17px;
      font-weight: 700;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }
    .toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 8px;
    }
    .cursor {
      color: var(--muted);
      font-size: 13px;
    }
    button {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      padding: 6px 10px;
      cursor: pointer;
    }
    button:hover { border-color: #9aa7bb; }
    canvas {
      width: 100%;
      height: 560px;
      display: block;
      cursor: grab;
    }
    canvas:active { cursor: grabbing; }
    @media (max-width: 760px) {
      .wrap { width: calc(100vw - 20px); margin-top: 12px; }
      header { display: block; }
      .hint { margin-top: 4px; white-space: normal; }
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      canvas { height: 420px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>Equity Curve</h1>
      <div class="hint">滚轮缩放，拖动平移，双击重置</div>
    </header>
    <section class="metrics" id="metrics"></section>
    <section class="panel">
      <div class="toolbar">
        <div class="cursor" id="cursorInfo">-</div>
        <button id="resetBtn">Reset</button>
      </div>
      <canvas id="chart"></canvas>
    </section>
  </div>
  <script>
    const payload = __PAYLOAD__;
    const equity = payload.equity || [];
    const MAX_RENDER_POINTS = 4000;
    let fullStart = equity.length ? equity[0].ts : 0;
    let fullEnd = equity.length ? equity[equity.length - 1].ts : 1;
    if (fullEnd <= fullStart) fullEnd = fullStart + 1;
    let viewStart = fullStart;
    let viewEnd = fullEnd;
    let dragging = false;
    let dragX = 0;
    let dragStart = viewStart;
    let dragEnd = viewEnd;
    let hoverTs = null;

    const canvas = document.getElementById("chart");
    const cursorInfo = document.getElementById("cursorInfo");

    function init() {
      renderMetrics();
      canvas.addEventListener("wheel", onWheel, { passive: false });
      canvas.addEventListener("mousedown", onMouseDown);
      canvas.addEventListener("mousemove", onMouseMove);
      canvas.addEventListener("mouseleave", () => { hoverTs = null; draw(); });
      canvas.addEventListener("dblclick", resetView);
      window.addEventListener("mouseup", () => { dragging = false; });
      window.addEventListener("resize", draw);
      document.getElementById("resetBtn").addEventListener("click", resetView);
      draw();
    }

    function renderMetrics() {
      const root = document.getElementById("metrics");
      root.innerHTML = "";
      for (const item of payload.metrics || []) {
        const div = document.createElement("div");
        div.className = "metric";
        div.innerHTML = `<span>${item.label}</span><b>${formatValue(item.value, item.kind)}</b>`;
        root.appendChild(div);
      }
    }

    function draw() {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);
      const area = { x: 62, y: 18, w: rect.width - 112, h: rect.height - 58 };
      drawFrame(ctx, area);
      const data = visible();
      if (data.length < 2) {
        cursorInfo.textContent = "No equity data";
        return;
      }
      const scale = yScale(data, 0.08);
      drawYGrid(ctx, area, scale, formatMoney);
      drawXGrid(ctx, area);
      drawLine(ctx, area, data, d => d.equity, scale, "#2563eb", false);
      drawHover(ctx, area, scale);
    }

    function visible() {
      const startIdx = lowerBound(viewStart);
      const endIdx = upperBound(viewEnd);
      const count = Math.max(0, endIdx - startIdx);
      if (count <= 0) return [];
      if (count <= MAX_RENDER_POINTS) {
        return equity.slice(startIdx, endIdx);
      }
      const step = Math.ceil(count / MAX_RENDER_POINTS);
      const result = [];
      for (let i = startIdx; i < endIdx; i += step) {
        result.push(equity[i]);
      }
      const tail = equity[endIdx - 1];
      if (tail && result[result.length - 1]?.ts !== tail.ts) {
        result.push(tail);
      }
      return result;
    }

    function drawFrame(ctx, area) {
      ctx.strokeStyle = "#d7dde8";
      ctx.lineWidth = 1;
      ctx.strokeRect(area.x, area.y, area.w, area.h);
    }

    function drawYGrid(ctx, area, scale, formatter) {
      ctx.save();
      ctx.strokeStyle = "#e7ebf2";
      ctx.fillStyle = "#667085";
      ctx.font = "12px Inter, Segoe UI, Arial";
      for (let i = 0; i <= 5; i++) {
        const value = scale.min + (scale.max - scale.min) * i / 5;
        const y = yPos(value, area, scale);
        ctx.beginPath();
        ctx.moveTo(area.x, y);
        ctx.lineTo(area.x + area.w, y);
        ctx.stroke();
        ctx.fillText(formatter(value), 8, y + 4);
      }
      ctx.restore();
    }

    function drawXGrid(ctx, area) {
      ctx.save();
      ctx.strokeStyle = "#eef1f6";
      ctx.fillStyle = "#667085";
      ctx.font = "12px Inter, Segoe UI, Arial";
      for (let i = 0; i <= 6; i++) {
        const ts = viewStart + (viewEnd - viewStart) * i / 6;
        const x = xPos(ts, area);
        ctx.beginPath();
        ctx.moveTo(x, area.y);
        ctx.lineTo(x, area.y + area.h);
        ctx.stroke();
        ctx.fillText(formatDate(ts), Math.min(x - 34, area.x + area.w - 72), area.y + area.h + 26);
      }
      ctx.restore();
    }

    function drawLine(ctx, area, data, valueAccessor, scale, color, dashed) {
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = dashed ? 1.2 : 2;
      if (dashed) ctx.setLineDash([5, 4]);
      ctx.beginPath();
      data.forEach((d, i) => {
        const x = xPos(d.ts, area);
        const y = yPos(valueAccessor(d), area, scale);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.restore();
    }

    function drawHover(ctx, area, scale) {
      const point = nearest(equity, hoverTs || (viewStart + viewEnd) / 2);
      if (!point) return;
      const x = xPos(point.ts, area);
      const y = yPos(point.equity, area, scale);
      ctx.save();
      ctx.strokeStyle = "#111827";
      ctx.globalAlpha = 0.22;
      ctx.beginPath();
      ctx.moveTo(x, area.y);
      ctx.lineTo(x, area.y + area.h);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#2563eb";
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
      const dd = point.peak ? point.equity / point.peak - 1 : 0;
      cursorInfo.textContent = `${formatDateTime(point.ts)}  Equity ${formatMoney(point.equity)}  Drawdown ${formatPercent(dd)}`;
    }

    function yScale(points, pad) {
      let min = Infinity;
      let max = -Infinity;
      for (const point of points || []) {
        const values = [point.equity, point.peak];
        for (const value of values) {
          if (!Number.isFinite(value)) continue;
          if (value < min) min = value;
          if (value > max) max = value;
        }
      }
      if (!Number.isFinite(min) || !Number.isFinite(max)) { min = 0; max = 1; }
      const span = Math.max(1e-9, max - min);
      return { min: min - span * pad, max: max + span * pad };
    }

    function xPos(ts, area) {
      return area.x + ((ts - viewStart) / Math.max(1, viewEnd - viewStart)) * area.w;
    }

    function yPos(value, area, scale) {
      return area.y + area.h - ((value - scale.min) / Math.max(1e-12, scale.max - scale.min)) * area.h;
    }

    function onWheel(event) {
      event.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const ratio = clamp((event.clientX - rect.left - 62) / Math.max(1, rect.width - 112), 0, 1);
      const anchor = viewStart + (viewEnd - viewStart) * ratio;
      const factor = event.deltaY > 0 ? 1.22 : 0.82;
      const left = (anchor - viewStart) * factor;
      const right = (viewEnd - anchor) * factor;
      setView(anchor - left, anchor + right);
    }

    function onMouseDown(event) {
      dragging = true;
      dragX = event.clientX;
      dragStart = viewStart;
      dragEnd = viewEnd;
    }

    function onMouseMove(event) {
      const rect = canvas.getBoundingClientRect();
      const ratio = clamp((event.clientX - rect.left - 62) / Math.max(1, rect.width - 112), 0, 1);
      hoverTs = viewStart + (viewEnd - viewStart) * ratio;
      if (dragging) {
        const dx = event.clientX - dragX;
        const shift = -dx / Math.max(1, rect.width - 112) * (dragEnd - dragStart);
        setView(dragStart + shift, dragEnd + shift, false);
      }
      draw();
    }

    function setView(start, end, redraw = true) {
      const span = Math.max(60000, end - start);
      if (span >= fullEnd - fullStart) {
        viewStart = fullStart;
        viewEnd = fullEnd;
      } else {
        viewStart = clamp(start, fullStart, fullEnd - span);
        viewEnd = viewStart + span;
      }
      if (redraw) draw();
    }

    function resetView() {
      setView(fullStart, fullEnd);
    }

    function nearest(data, ts) {
      if (!data.length || !Number.isFinite(ts)) return null;
      let lo = 0, hi = data.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (data[mid].ts < ts) lo = mid + 1;
        else hi = mid;
      }
      const a = data[lo];
      const b = data[Math.max(0, lo - 1)];
      if (!b) return a;
      return Math.abs(a.ts - ts) < Math.abs(b.ts - ts) ? a : b;
    }

    function lowerBound(ts) {
      let lo = 0, hi = equity.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (equity[mid].ts < ts) lo = mid + 1;
        else hi = mid;
      }
      return lo;
    }

    function upperBound(ts) {
      let lo = 0, hi = equity.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (equity[mid].ts <= ts) lo = mid + 1;
        else hi = mid;
      }
      return lo;
    }

    function clamp(value, min, max) {
      return Math.max(min, Math.min(max, value));
    }

    function formatValue(value, kind) {
      if (kind === "money") return formatMoney(value);
      if (kind === "percent") return formatPercent(value);
      if (kind === "integer") return Math.round(value).toLocaleString();
      return Number(value).toFixed(3);
    }

    function formatMoney(value) {
      return Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 });
    }

    function formatPercent(value) {
      return `${(Number(value) * 100).toFixed(2)}%`;
    }

    function formatDate(ts) {
      return new Date(ts).toLocaleDateString();
    }

    function formatDateTime(ts) {
      return new Date(ts).toLocaleString();
    }

    init();
  </script>
</body>
</html>
"""
