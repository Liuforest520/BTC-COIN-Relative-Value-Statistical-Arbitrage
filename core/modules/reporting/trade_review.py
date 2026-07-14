from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from html import escape
from math import isfinite
from json import dumps, loads
from pathlib import Path

import polars as pl
import yaml


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "results" / "review" / "trade_review.html"


def latest_backtest_dir() -> Path:
    backtest_root = ROOT / "results" / "backtests"
    runs = [path for path in backtest_root.iterdir() if path.is_dir()]
    if not runs:
        raise FileNotFoundError("results/backtests 下面没有回测结果")
    return max(runs, key=lambda path: path.stat().st_mtime)


def read_frame(path: Path, columns: list[str] | None = None) -> pl.DataFrame:
    if not path.exists():
        return _empty_frame(columns)
    try:
        frame = pl.read_csv(path, infer_schema_length=10000)
    except Exception:
        return _empty_frame(columns)
    if columns is None:
        return frame
    for column in columns:
        if column not in frame.columns:
            frame = frame.with_columns(pl.lit(None).alias(column))
    return frame.select(columns)


def _empty_frame(columns: list[str] | None = None) -> pl.DataFrame:
    values = {}
    for column in columns or []:
        dtype = pl.Int64 if column == "ts" else pl.Null
        values[column] = pl.Series(column, [], dtype=dtype)
    return pl.DataFrame(values)


def sample_frame(frame: pl.DataFrame, max_points: int) -> pl.DataFrame:
    if frame.height <= max_points:
        return frame
    step = max(1, frame.height // max_points)
    return frame.with_row_index("row_id").filter(pl.col("row_id") % step == 0).drop("row_id")


def fmt_time(ts: int | float | None) -> str:
    if ts is None:
        return ""
    return datetime.fromtimestamp(float(ts) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def drawdown_window(equity: pl.DataFrame) -> dict:
    if equity.is_empty() or "ts" not in equity.columns or "equity" not in equity.columns:
        return empty_window("最大回撤")
    frame = equity.with_columns(
        pl.col("equity").cum_max().alias("peak"),
    ).with_columns(
        (pl.col("equity") / pl.col("peak") - 1).alias("drawdown"),
    ).drop_nulls(["ts", "equity", "drawdown"])
    if frame.is_empty():
        return empty_window("最大回撤")
    trough = frame.sort("drawdown").row(0, named=True)
    peak_before = frame.filter(pl.col("ts") <= trough["ts"]).filter(pl.col("equity") == trough["peak"]).tail(1)
    start_ts = peak_before["ts"][0] if peak_before.height else frame["ts"][0]
    end_ts = trough["ts"]
    padding = max(12 * 60 * 60 * 1000, int((end_ts - start_ts) * 0.25))
    return {
        "name": "最大回撤",
        "start": max(int(frame["ts"].min()), int(start_ts - padding)),
        "end": min(int(frame["ts"].max()), int(end_ts + padding)),
        "peak_ts": int(start_ts),
        "trough_ts": int(end_ts),
        "drawdown": float(trough["drawdown"]),
        "description": f"{fmt_time(start_ts)} 到 {fmt_time(end_ts)}，资金从阶段高点回落 {abs(float(trough['drawdown'])):.2%}",
    }


def best_gain_window(equity: pl.DataFrame) -> dict:
    if equity.is_empty() or "ts" not in equity.columns or "equity" not in equity.columns:
        return empty_window("最高收益段")
    values = equity.select(["ts", "equity"]).to_dicts()
    if not values:
        return empty_window("最高收益段")
    min_idx = 0
    best_start = 0
    best_end = 0
    best_gain = 0.0
    for idx, row in enumerate(values):
        current_gain = row["equity"] / values[min_idx]["equity"] - 1
        if current_gain > best_gain:
            best_gain = current_gain
            best_start = min_idx
            best_end = idx
        if row["equity"] < values[min_idx]["equity"]:
            min_idx = idx

    if best_gain <= 0:
        start_ts = int(equity["ts"].min())
        end_ts = int(equity["ts"].max())
        return {
            "name": "无正收益区间",
            "start": start_ts,
            "end": end_ts,
            "low_ts": start_ts,
            "high_ts": end_ts,
            "gain": 0.0,
            "description": "该回测区间内没有正收益窗口",
        }

    start_ts = int(values[best_start]["ts"])
    end_ts = int(values[best_end]["ts"])
    padding = max(12 * 60 * 60 * 1000, int((end_ts - start_ts) * 0.15))
    return {
        "name": "最高收益段",
        "start": max(int(equity["ts"].min()), start_ts - padding),
        "end": min(int(equity["ts"].max()), end_ts + padding),
        "low_ts": start_ts,
        "high_ts": end_ts,
        "gain": float(best_gain),
        "description": f"{fmt_time(start_ts)} 到 {fmt_time(end_ts)}，资金从阶段低点上涨 {best_gain:.2%}",
    }


def empty_window(name: str) -> dict:
    return {
        "name": name,
        "start": None,
        "end": None,
        "description": "没有足够数据",
    }


def trade_rows(trades: pl.DataFrame) -> list[dict]:
    if trades.is_empty():
        return []
    grouped = trades.group_by(["group_id", "action", "ts", "position_id"]).agg(
        pl.col("symbol").alias("symbols"),
        pl.col("side").alias("sides"),
        pl.col("notional").sum().alias("gross_notional"),
        pl.col("fee").sum().alias("fee"),
        pl.col("slippage").sum().alias("slippage"),
        pl.col("target_hedge_ratio").drop_nulls().first().alias("target_hedge_ratio"),
    ).sort("ts")

    rows = []
    for row in grouped.to_dicts():
        legs = []
        for symbol, side in zip(row["symbols"], row["sides"]):
            legs.append(f"{symbol} {side}")
        rows.append(
            {
                "ts": int(row["ts"]),
                "time": fmt_time(row["ts"]),
                "group_id": row["group_id"],
                "position_id": row["position_id"],
                "action": row["action"],
                "legs": " / ".join(legs),
                "gross_notional": float(row["gross_notional"] or 0),
                "fee": float(row["fee"] or 0),
                "slippage": float(row["slippage"] or 0),
                "target_hedge_ratio": float(row["target_hedge_ratio"]) if row["target_hedge_ratio"] is not None else None,
            }
        )
    return rows


def threshold_rows(run_dir: Path) -> list[dict]:
    config_path = run_dir / "config.yaml"
    default_entry_z = 2.0
    default_exit_z = 0.3
    entry_rule_method = "fixed_z"

    if not config_path.exists():
        entry_z = default_entry_z
        exit_z = default_exit_z
    else:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        active_setup = raw.get("active_setup")
        setup = raw.get("setups", {}).get(active_setup, {})
        signal_name = setup.get("signal")
        signal = raw.get("signals", {}).get(signal_name, {})
        entry_z = float(signal.get("entry_z", default_entry_z))
        exit_z = float(signal.get("exit_z", default_exit_z))
        entry_rule_method = signal.get("entry_rule", {}).get("method", "fixed_z")

    rows = [
        {"value": exit_z, "label": f"平仓 +{exit_z:.2f}", "kind": "exit"},
        {"value": -exit_z, "label": f"平仓 -{exit_z:.2f}", "kind": "exit"},
    ]
    if entry_rule_method != "rolling_quantile":
        rows.extend(
            [
                {"value": entry_z, "label": f"开仓 +{entry_z:.2f}", "kind": "entry"},
                {"value": -entry_z, "label": f"开仓 -{entry_z:.2f}", "kind": "entry"},
            ]
        )
    return rows


def build_payload(run_dir: Path, max_points: int) -> dict:
    equity = read_frame(run_dir / "equity_curve.csv", ["ts", "equity"]).sort("ts")
    signal = read_frame(
        run_dir / "signal_curve.csv",
        [
            "ts",
            "x_close",
            "y_close",
            "zscore",
            "entry_z_upper",
            "entry_z_lower",
            "action",
            "side",
            "position_side",
            "hedge_ratio",
            "spread",
        ],
    ).sort("ts")
    position = read_frame(
        run_dir / "position_curve.csv",
        ["ts", "gross_exposure_ratio", "net_exposure_ratio"],
    ).sort("ts")
    trades = read_frame(
        run_dir / "trades.csv",
        ["group_id", "symbol", "action", "side", "quantity", "price", "notional", "fee", "slippage", "ts", "position_id", "target_hedge_ratio"],
    ).sort("ts")
    try:
        metrics = loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    except Exception:
        metrics = {}

    if equity.is_empty() or "ts" not in equity.columns:
        raise ValueError(f"{run_dir} has no equity_curve.csv data")

    equity_sample = sample_frame(equity, max_points)
    signal_sample = sample_frame(signal, max_points)
    position_sample = sample_frame(position, max_points)
    btc_base = _first_positive(signal, "x_close")
    coin_base = _first_positive(signal, "y_close")

    merged = (
        equity_sample.join(signal_sample, on="ts", how="left")
        .join(position_sample, on="ts", how="left")
        .sort("ts")
    )

    points = []
    for row in merged.to_dicts():
        x_close = row.get("x_close")
        y_close = row.get("y_close")
        points.append(
            {
                "ts": int(row["ts"]),
                "time": fmt_time(row["ts"]),
                "equity": _float(row.get("equity")),
                "btc": _float(x_close),
                "coin": _float(y_close),
                "btc_norm": _float(x_close / btc_base if x_close and btc_base else None),
                "coin_norm": _float(y_close / coin_base if y_close and coin_base else None),
                "zscore": _float(row.get("zscore")),
                "entry_z_upper": _float(row.get("entry_z_upper")),
                "entry_z_lower": _float(row.get("entry_z_lower")),
                "spread": _float(row.get("spread")),
                "action": row.get("action"),
                "side": row.get("side"),
                "position_side": row.get("position_side"),
                "hedge_ratio": _float(row.get("hedge_ratio")),
                "gross_exposure_ratio": _float(row.get("gross_exposure_ratio")),
                "net_exposure_ratio": _float(row.get("net_exposure_ratio")),
            }
        )

    trades_grouped = trade_rows(trades)
    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "metrics": metrics,
        "points": points,
        "trades": trades_grouped,
        "thresholds": threshold_rows(run_dir),
        "windows": {
            "drawdown": drawdown_window(equity),
            "gain": best_gain_window(equity),
        },
        "range": {
            "start": int(equity["ts"].min()),
            "end": int(equity["ts"].max()),
        },
    }


def _float(value) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
        if not isfinite(value):
            return None
        return value
    except (TypeError, ValueError):
        return None


def _first_positive(frame: pl.DataFrame, column: str) -> float | None:
    if frame.is_empty() or column not in frame.columns:
        return None
    values = frame.select(pl.col(column).cast(pl.Float64, strict=False).alias(column)).drop_nulls(column)
    values = values.filter(pl.col(column) > 0)
    if values.is_empty():
        return None
    return float(values[column][0])


def render_html(payload: dict) -> str:
    payload_json = dumps(_json_safe(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    title = f"BTC-COIN 交易复盘 - {payload['run_name']}"
    return HTML_TEMPLATE.replace("__TITLE__", escape(title)).replace("__PAYLOAD__", payload_json)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if isfinite(value):
            return value
        if value > 0:
            return "Infinity"
        if value < 0:
            return "-Infinity"
        return None
    return value


def export_trade_review_html(run_dir: Path, output: Path | None = None, max_points: int = 9000) -> Path:
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir

    output = output or (run_dir / "trade_review.html")
    if not output.is_absolute():
        output = ROOT / output

    payload = build_payload(run_dir, max_points)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(payload), encoding="utf-8")
    return output


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-points", type=int, default=9000)
    args = parser.parse_args()

    run_dir = args.run_dir or latest_backtest_dir()
    output = export_trade_review_html(run_dir, args.output, args.max_points)
    print(output)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
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
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    header {
      padding: 22px 28px 12px;
      border-bottom: 1px solid var(--line);
      background: #fff;
      position: sticky;
      top: 0;
      z-index: 10;
    }
    h1 {
      font-size: 22px;
      margin: 0 0 12px;
      letter-spacing: 0;
    }
    .topbar {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: end;
    }
    .metrics {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px 10px;
      background: #fbfdff;
      font-size: 13px;
      color: var(--muted);
    }
    .metric b {
      color: var(--ink);
      font-weight: 650;
      margin-left: 4px;
    }
    .buttons {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    button {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      padding: 7px 10px;
      cursor: pointer;
      font-size: 13px;
    }
    button.primary {
      background: var(--blue);
      border-color: var(--blue);
      color: white;
    }
    main {
      padding: 18px 28px 28px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 16px;
    }
    .chart-stack {
      display: grid;
      gap: 12px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-width: 0;
    }
    .panel h2 {
      margin: 0 0 8px;
      font-size: 15px;
    }
    .range-panel {
      display: grid;
      grid-template-columns: 1fr 1fr auto auto;
      gap: 10px;
      align-items: end;
      margin-bottom: 12px;
    }
    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
    }
    input[type="range"] {
      width: 100%;
    }
    .boundary-value {
      color: var(--ink);
      font-size: 12px;
      font-weight: 650;
    }
    .toggle-label {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 32px;
      white-space: nowrap;
      color: var(--ink);
    }
    .chart {
      width: 100%;
      height: 230px;
      display: block;
      border-top: 1px solid #eef1f6;
    }
    .small-chart { height: 205px; }
    .hint {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }
    .side {
      position: sticky;
      top: 112px;
      align-self: start;
      display: grid;
      gap: 12px;
    }
    .summary {
      display: grid;
      gap: 8px;
      font-size: 13px;
    }
    .summary-row {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      border-bottom: 1px solid #eef1f6;
      padding-bottom: 6px;
    }
    .trade-list {
      max-height: 520px;
      overflow: auto;
      display: grid;
      gap: 8px;
      padding-right: 2px;
    }
    .trade-card {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 9px;
      background: #fbfdff;
      font-size: 12px;
      line-height: 1.45;
      cursor: pointer;
    }
    .trade-card:hover {
      border-color: var(--blue);
    }
    .trade-card b {
      font-size: 13px;
    }
    .open { color: var(--green); }
    .close { color: var(--red); }
    .legend {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }
    .dot {
      width: 9px;
      height: 9px;
      display: inline-block;
      border-radius: 50%;
      margin-right: 5px;
      vertical-align: middle;
    }
    @media (max-width: 1100px) {
      main { grid-template-columns: 1fr; }
      .side { position: static; }
      .topbar { grid-template-columns: 1fr; }
      .buttons { justify-content: flex-start; }
    }
  </style>
</head>
<body>
  <header>
    <h1>BTC-COIN 回测交易复盘</h1>
    <div class="topbar">
      <div class="metrics" id="metrics"></div>
      <div class="buttons">
        <button class="primary" id="drawdownBtn">跳到最大回撤</button>
        <button id="gainBtn">跳到最高收益段</button>
        <button id="fullBtn">全区间</button>
      </div>
    </div>
  </header>
  <main>
    <section class="chart-stack">
      <div class="panel">
        <div class="range-panel">
          <label>左边界
            <span class="boundary-value" id="leftBoundaryText"></span>
            <input id="windowLeft" type="range" min="0" max="1000" value="0" />
          </label>
          <label>右边界
            <span class="boundary-value" id="rightBoundaryText"></span>
            <input id="windowRight" type="range" min="0" max="1000" value="200" />
          </label>
          <label class="toggle-label">
            <input id="showTradeMarkers" type="checkbox" checked />
            交易标记
          </label>
          <button id="resetZoom">重置</button>
        </div>
        <div class="hint" id="windowText"></div>
      </div>

      <div class="panel">
        <h2>资金曲线</h2>
        <div class="legend">
          <span><i class="dot" style="background: var(--blue)"></i>账户权益</span>
          <span><i class="dot" style="background: var(--green)"></i>底部小三角为开仓</span>
          <span><i class="dot" style="background: var(--red)"></i>顶部小三角为平仓</span>
        </div>
        <canvas class="chart" id="equityChart"></canvas>
      </div>

      <div class="panel">
        <h2>BTC / COIN 归一化价格走势</h2>
        <div class="legend">
          <span><i class="dot" style="background: var(--blue)"></i>BTC</span>
          <span><i class="dot" style="background: var(--orange)"></i>COIN</span>
        </div>
        <canvas class="chart" id="priceChart"></canvas>
      </div>

      <div class="panel">
        <h2>z-score 与持仓方向</h2>
        <div class="legend">
          <span><i class="dot" style="background: var(--purple)"></i>z-score</span>
          <span>横线标出开仓/平仓阈值</span>
        </div>
        <canvas class="chart small-chart" id="zChart"></canvas>
      </div>
    </section>

    <aside class="side">
      <div class="panel">
        <h2>当前窗口</h2>
        <div class="summary" id="windowSummary"></div>
      </div>
      <div class="panel">
        <h2>窗口内交易</h2>
        <div class="hint">点击某笔交易，会把窗口移动到它附近。</div>
        <div class="trade-list" id="tradeList"></div>
      </div>
    </aside>
  </main>

  <script>
    const payload = __PAYLOAD__;
    const points = payload.points;
    const trades = payload.trades;
    const thresholds = payload.thresholds || [
      { value: 2, label: "开仓 +2.00", kind: "entry" },
      { value: -2, label: "开仓 -2.00", kind: "entry" },
      { value: 0.3, label: "平仓 +0.30", kind: "exit" },
      { value: -0.3, label: "平仓 -0.30", kind: "exit" }
    ];
    const tsMin = payload.range.start;
    const tsMax = payload.range.end;
    const totalMs = tsMax - tsMin;
    const dayMs = 24 * 60 * 60 * 1000;
    let viewStart = tsMin;
    let viewEnd = Math.min(tsMax, tsMin + 14 * dayMs);

    const metricsEl = document.getElementById("metrics");
    const leftInput = document.getElementById("windowLeft");
    const rightInput = document.getElementById("windowRight");
    const leftBoundaryText = document.getElementById("leftBoundaryText");
    const rightBoundaryText = document.getElementById("rightBoundaryText");
    const markerToggle = document.getElementById("showTradeMarkers");
    const windowText = document.getElementById("windowText");
    const windowSummary = document.getElementById("windowSummary");
    const tradeList = document.getElementById("tradeList");
    const sliderMax = 1000;
    const minWindowMs = 60 * 60 * 1000;

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
    function setView(start, end) {
      const minSize = Math.min(minWindowMs, totalMs);
      if (end - start < minSize) end = start + minSize;
      if (start < tsMin) {
        end += tsMin - start;
        start = tsMin;
      }
      if (end > tsMax) {
        start -= end - tsMax;
        end = tsMax;
      }
      viewStart = clamp(start, tsMin, tsMax);
      viewEnd = clamp(end, tsMin, tsMax);
      syncInputs();
      renderAll();
    }
    function syncInputs() {
      leftInput.value = tsToSlider(viewStart);
      rightInput.value = tsToSlider(viewEnd);
      leftBoundaryText.textContent = dt(viewStart);
      rightBoundaryText.textContent = dt(viewEnd);
    }
    function tsToSlider(ts) {
      return Math.round((ts - tsMin) / Math.max(1, totalMs) * sliderMax);
    }
    function sliderToTs(value) {
      return tsMin + Number(value) / sliderMax * totalMs;
    }
    function updateViewFromBoundaryInputs(changedSide) {
      let leftValue = Number(leftInput.value);
      let rightValue = Number(rightInput.value);
      const minSteps = Math.max(1, Math.round(minWindowMs / Math.max(1, totalMs) * sliderMax));

      if (rightValue - leftValue < minSteps) {
        if (changedSide === "left") {
          leftValue = rightValue - minSteps;
        } else {
          rightValue = leftValue + minSteps;
        }
      }
      if (leftValue < 0) {
        rightValue -= leftValue;
        leftValue = 0;
      }
      if (rightValue > sliderMax) {
        leftValue -= rightValue - sliderMax;
        rightValue = sliderMax;
      }

      leftValue = clamp(leftValue, 0, sliderMax);
      rightValue = clamp(rightValue, 0, sliderMax);
      leftInput.value = leftValue;
      rightInput.value = rightValue;
      setView(sliderToTs(leftValue), sliderToTs(rightValue));
    }

    function initMetrics() {
      const m = payload.metrics;
      const rows = [
        ["最终权益", num(m.final_equity)],
        ["总收益", pct(m.total_return)],
        ["Sharpe", num(m.sharpe, 3)],
        ["最大回撤", pct(m.max_drawdown)],
        ["交易笔数", num(m.trade_count, 0)],
        ["日均换手", num(m.daily_turnover, 2)]
      ];
      metricsEl.innerHTML = rows.map(([k, v]) => `<span class="metric">${k}<b>${v}</b></span>`).join("");
    }

    function visiblePoints() {
      return points.filter(p => p.ts >= viewStart && p.ts <= viewEnd);
    }
    function visibleTrades() {
      return trades.filter(t => t.ts >= viewStart && t.ts <= viewEnd);
    }

    function renderAll() {
      const data = visiblePoints();
      const tradeData = visibleTrades();
      windowText.textContent = `${dt(viewStart)}  到  ${dt(viewEnd)}，窗口内 ${data.length} 个采样点，${tradeData.length} 组交易`;
      renderSummary(data, tradeData);
      renderTradeList(tradeData);
      drawLineChart("equityChart", data, [
        { key: "equity", color: "#2563eb", label: "权益" }
      ], { trades: markerToggle.checked ? tradeData : [], yFormat: v => num(v, 0) });
      drawLineChart("priceChart", data, [
        { key: "btc_norm", color: "#2563eb", label: "BTC" },
        { key: "coin_norm", color: "#f97316", label: "COIN" }
      ], { trades: markerToggle.checked ? tradeData : [], yFormat: v => v.toFixed(3) });
      drawLineChart("zChart", data, [
        { key: "zscore", color: "#7c3aed", label: "z-score" },
        { key: "entry_z_upper", color: "#16a34a", label: "entry upper", dash: true },
        { key: "entry_z_lower", color: "#dc2626", label: "entry lower", dash: true }
      ], { trades: markerToggle.checked ? tradeData : [], thresholds, yFormat: v => v.toFixed(2) });
    }

    function renderSummary(data, tradeData) {
      if (!data.length) {
        windowSummary.innerHTML = "";
        return;
      }
      const first = data[0], last = data[data.length - 1];
      const equities = data.map(p => p.equity).filter(v => v !== null);
      const zscores = data.map(p => p.zscore).filter(v => v !== null);
      const ret = first.equity ? last.equity / first.equity - 1 : null;
      const minEq = Math.min(...equities);
      const maxEq = Math.max(...equities);
      const maxAbsZ = zscores.length ? Math.max(...zscores.map(v => Math.abs(v))) : null;
      const rows = [
        ["窗口收益", pct(ret)],
        ["权益最低/最高", `${num(minEq, 0)} / ${num(maxEq, 0)}`],
        ["最大 |z-score|", num(maxAbsZ, 2)],
        ["交易组数", num(tradeData.length, 0)],
        ["开仓/平仓", `${tradeData.filter(t => t.action === "open").length} / ${tradeData.filter(t => t.action === "close").length}`]
      ];
      windowSummary.innerHTML = rows.map(([k, v]) => `<div class="summary-row"><span>${k}</span><b>${v}</b></div>`).join("");
    }

    function renderTradeList(tradeData) {
      if (!tradeData.length) {
        tradeList.innerHTML = `<div class="hint">这个窗口里没有交易。</div>`;
        return;
      }
      tradeList.innerHTML = tradeData.map(t => `
        <div class="trade-card" data-ts="${t.ts}">
          <b class="${t.action}">${t.action === "open" ? "开仓" : "平仓"}</b>
          <span>${t.time}</span><br>
          ${t.legs}<br>
          名义金额 ${num(t.gross_notional, 0)}，手续费 ${num(t.fee, 2)}，滑点 ${num(t.slippage, 2)}<br>
          hedge ratio ${num(t.target_hedge_ratio, 3)}
        </div>`).join("");
      tradeList.querySelectorAll(".trade-card").forEach(card => {
        card.addEventListener("click", () => {
          const center = Number(card.dataset.ts);
          setView(center - 12 * 60 * 60 * 1000, center + 12 * 60 * 60 * 1000);
        });
      });
    }

    function drawLineChart(canvasId, data, series, options = {}) {
      const canvas = document.getElementById(canvasId);
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.floor(rect.width * dpr);
      canvas.height = Math.floor(rect.height * dpr);
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const w = rect.width, h = rect.height;
      ctx.clearRect(0, 0, w, h);

      const pad = { left: 62, right: 18, top: 14, bottom: 34 };
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
        ctx.fillText("没有数据", pad.left, pad.top + 20);
        return;
      }
      let yMin = Math.min(...allValues), yMax = Math.max(...allValues);
      if (yMin === yMax) { yMin -= 1; yMax += 1; }
      const yPad = (yMax - yMin) * 0.08;
      yMin -= yPad; yMax += yPad;
      const x = ts => pad.left + (ts - viewStart) / (viewEnd - viewStart) * plotW;
      const y = value => pad.top + (yMax - value) / (yMax - yMin) * plotH;

      ctx.strokeStyle = "#d7dde8";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.left, pad.top);
      ctx.lineTo(pad.left, pad.top + plotH);
      ctx.lineTo(pad.left + plotW, pad.top + plotH);
      ctx.stroke();

      ctx.fillStyle = "#667085";
      ctx.font = "12px Microsoft YaHei, Arial";
      for (let i = 0; i <= 4; i++) {
        const value = yMin + (yMax - yMin) * i / 4;
        const yy = y(value);
        ctx.strokeStyle = "#eef1f6";
        ctx.beginPath();
        ctx.moveTo(pad.left, yy);
        ctx.lineTo(pad.left + plotW, yy);
        ctx.stroke();
        ctx.fillText(options.yFormat ? options.yFormat(value) : value.toFixed(2), 6, yy + 4);
      }
      const labels = [viewStart, (viewStart + viewEnd) / 2, viewEnd];
      for (const tick of labels) {
        ctx.fillText(dt(tick).slice(5), x(tick) - 40, pad.top + plotH + 22);
      }

      if (thresholdRows.length) {
        ctx.font = "12px Microsoft YaHei, Arial";
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
          const textWidth = ctx.measureText(label).width;
          const labelX = pad.left + plotW - textWidth - 6;
          const labelY = clamp(yy - 8, pad.top + 4, pad.top + plotH - 8);
          ctx.setLineDash([]);
          ctx.globalAlpha = 0.92;
          ctx.fillStyle = "rgba(255, 255, 255, 0.86)";
          ctx.fillRect(labelX - 5, labelY - 11, textWidth + 10, 16);
          ctx.globalAlpha = 1;
          ctx.fillStyle = isEntry ? "#334155" : "#64748b";
          ctx.fillText(label, labelX, labelY);
        }
        ctx.setLineDash([]);
        ctx.lineWidth = 1;
      }

      for (const s of series) {
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 1.8;
        ctx.setLineDash(s.dash ? [6, 5] : []);
        ctx.beginPath();
        let started = false;
        for (const p of data) {
          const v = p[s.key];
          if (v === null || v === undefined || Number.isNaN(v)) {
            started = false;
            continue;
          }
          const xx = x(p.ts), yy = y(v);
          if (!started) {
            ctx.moveTo(xx, yy);
            started = true;
          } else {
            ctx.lineTo(xx, yy);
          }
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }

      if (options.trades) {
        for (const t of options.trades) {
          const xx = x(t.ts);
          ctx.fillStyle = t.action === "open" ? "#16a34a" : "#dc2626";
          ctx.globalAlpha = 0.72;
          ctx.beginPath();
          if (t.action === "open") {
            const yy = pad.top + plotH - 7;
            ctx.moveTo(xx, yy - 6);
            ctx.lineTo(xx - 4, yy + 3);
            ctx.lineTo(xx + 4, yy + 3);
          } else {
            const yy = pad.top + 7;
            ctx.moveTo(xx, yy + 6);
            ctx.lineTo(xx - 4, yy - 3);
            ctx.lineTo(xx + 4, yy - 3);
          }
          ctx.closePath();
          ctx.fill();
          ctx.globalAlpha = 1;
        }
      }
    }

    leftInput.addEventListener("input", () => updateViewFromBoundaryInputs("left"));
    rightInput.addEventListener("input", () => updateViewFromBoundaryInputs("right"));
    markerToggle.addEventListener("change", renderAll);
    document.getElementById("resetZoom").addEventListener("click", () => setView(tsMin, Math.min(tsMax, tsMin + 14 * dayMs)));
    document.getElementById("fullBtn").addEventListener("click", () => setView(tsMin, tsMax));
    document.getElementById("drawdownBtn").addEventListener("click", () => {
      const w = payload.windows.drawdown;
      setView(w.start, w.end);
    });
    document.getElementById("gainBtn").addEventListener("click", () => {
      const w = payload.windows.gain;
      setView(w.start, w.end);
    });
    window.addEventListener("resize", renderAll);

    initMetrics();
    setView(payload.windows.drawdown.start, payload.windows.drawdown.end);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
