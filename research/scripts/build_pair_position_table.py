"""Per-pair table from a finished backtest: leg position notional (USDT) + returns.

Columns written to documents/pair_oi_return_table.csv:
    symbol_left, symbol_right, tradfi, oi_left, oi_right, 年化收益率

* oi_left / oi_right -- the notional (USDT) that leg carried, averaged over its
  **open fills**: every fill with ``action=open`` contributes
  ``notional = price * quantity`` in USDT and the column is the plain mean of
  those values (no time weighting).  The time-weighted mean is kept as an extra
  column in the enriched table.
* 年化收益率 -- the pair's CAGR from the run's per-pair contribution curve
  (curve_charts/pairs/pair_chart_metrics.csv).

Usage:
    python research/scripts/build_pair_position_table.py \
        --run-dir results/backtests/tls_log_price_full97_L2D_U4H_E3p0_X0p5_freeze
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

COLUMNS = ["symbol_left", "symbol_right", "tradfi", "oi_left", "oi_right", "年化收益率"]
MILLISECONDS_PER_DAY = 86_400_000


def load_pairs(run_dir: Path) -> list[dict]:
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    return config["setups"][config["active_setup"]]["pairs"]


def load_pair_metrics(run_dir: Path) -> dict[str, dict]:
    path = run_dir / "curve_charts" / "pairs" / "pair_chart_metrics.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["pair_id"]: row for row in csv.DictReader(handle)}


def run_window(run_dir: Path) -> tuple[float, float]:
    """Return (initial equity, window length in days) of the backtest."""
    initial = 100_000.0
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            import json

            initial = float(json.loads(metrics_path.read_text(encoding="utf-8"))
                            .get("initial_equity") or initial)
        except Exception:
            pass
    first = last = None
    with (run_dir / "equity_curve.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = row.get("ts")
            if not raw:
                continue
            try:
                ts = int(float(raw))
            except ValueError:
                continue
            if first is None:
                first = ts
            last = ts
    if first is None or last is None or last <= first:
        return initial, 0.0
    return initial, (last - first) / MILLISECONDS_PER_DAY


def load_concentration(path: Path) -> dict[str, float]:
    """symbol -> top10_filter (share of supply held by the top 10 addresses)."""
    values: dict[str, float] = {}
    if not path.exists():
        return values
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        reader.fieldnames = [name.strip() for name in reader.fieldnames or []]
        for row in reader:
            symbol = str(row.get("symbol") or "").strip()
            raw = str(row.get("top10_filter") or "").strip()
            if symbol and raw:
                try:
                    values[symbol] = float(raw)
                except ValueError:
                    pass
    return values


def passes_top10(pair: dict, concentration: dict[str, float], threshold: float) -> bool:
    """TradFi pairs always pass; a token leg with data must be <= threshold.

    A leg without concentration data is exempt (TradFi-mapped contracts have no
    holder distribution, and the Arkham export misses a few tokens).
    """
    if "tradfi" in (pair.get("tags") or []):
        return True
    for symbol in (pair["x_symbol"], pair["y_symbol"]):
        value = concentration.get(symbol)
        if value is not None and value > threshold:
            return False
    return True


def annualize(net_pnl: float, base: float, days: float) -> float | None:
    """CAGR of a contribution of ``net_pnl`` on ``base`` capital over ``days``."""
    if days <= 1 or base <= 0 or net_pnl <= -base:
        return None
    return (1.0 + net_pnl / base) ** (365.0 / days) - 1.0


def leg_exposure(run_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    """pair_id -> symbol -> per-leg open-fill statistics.

    ``notional_mean`` is the plain average of every open fill's USDT notional,
    ``notional_tw`` additionally weights each fill by how long it was held.
    """
    positions: dict[tuple[str, str], dict] = {}
    with (run_dir / "trades.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            pair_id, position_id = row.get("pair_id"), row.get("position_id")
            symbol = row.get("symbol")
            if not pair_id or not symbol:
                continue
            try:
                ts = int(float(row.get("ts") or 0))
                notional = abs(float(row.get("notional") or 0.0))
            except ValueError:
                continue
            action = (row.get("action") or "").lower()
            key = (pair_id, str(position_id or f"{ts}_{symbol}"))
            record = positions.setdefault(
                key, {"pair_id": pair_id, "open_ts": None, "close_ts": None,
                      "legs": defaultdict(float), "leg_fills": defaultdict(int)}
            )
            if action == "open":
                record["legs"][symbol] += notional
                record["leg_fills"][symbol] += 1
                record["open_ts"] = ts if record["open_ts"] is None else min(record["open_ts"], ts)
            else:
                record["close_ts"] = ts if record["close_ts"] is None else max(record["close_ts"], ts)

    window_end = max((record["close_ts"] or record["open_ts"] or 0) for record in positions.values()) if positions else 0
    blank = {"notional_tw": 0.0, "notional_mean": 0.0, "notional_total": 0.0,
             "minutes": 0.0, "fills": 0.0}
    exposure: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: dict(blank)))
    for record in positions.values():
        if record["open_ts"] is None:
            continue
        end = record["close_ts"] if record["close_ts"] is not None else window_end
        minutes = max(1.0, (end - record["open_ts"]) / 60_000.0)
        for symbol, notional in record["legs"].items():
            if notional <= 0:
                continue
            cell = exposure[record["pair_id"]][symbol]
            cell["notional_tw"] += notional * minutes
            cell["notional_mean"] += notional
            cell["notional_total"] += notional
            cell["minutes"] += minutes
            cell["fills"] += record["leg_fills"].get(symbol, 0)
    for per_symbol in exposure.values():
        for cell in per_symbol.values():
            cell["notional_tw"] = cell["notional_tw"] / cell["minutes"] if cell["minutes"] else 0.0
            cell["notional_mean"] = cell["notional_mean"] / cell["fills"] if cell["fills"] else 0.0
    return exposure


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path,
                        default=PROJECT_ROOT / "research" / "outputs" / "pair_sizing_table"
                        / "pair_oi_return_table.csv")
    parser.add_argument("--enriched-out", type=Path, default=None,
                        help="optional extra table with helper columns; skipped by default")
    parser.add_argument(
        "--annualization",
        choices=("full_window", "active_window"),
        default="full_window",
        help="how 年化收益率 is computed: over the whole backtest window (default, "
             "contributions are comparable and additive) or over each pair's own "
             "active window (inflates pairs that traded only briefly)",
    )
    parser.add_argument("--concentration-csv", type=Path,
                        default=PROJECT_ROOT / "documents" / "df_top10_bn.csv",
                        help="top-10 holder concentration table (column top10_filter)")
    parser.add_argument("--top10-max", type=float, nargs="*", default=[],
                        help="also write one filtered table per threshold, e.g. "
                             "--top10-max 0.2 0.4 0.6 (TradFi pairs are always kept)")
    parser.add_argument("--compare-runs", type=Path, nargs="*", default=[])
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    pairs = load_pairs(run_dir)
    metrics = load_pair_metrics(run_dir)
    exposure = leg_exposure(run_dir)
    extra = {}
    for extra_run in args.compare_runs:
        extra_run = extra_run if extra_run.is_absolute() else PROJECT_ROOT / extra_run
        extra[extra_run.name] = load_pair_metrics(extra_run)

    rows = []
    initial_equity, window_days = run_window(run_dir)
    print(f"window   : {window_days:.1f} days   initial equity: {initial_equity:,.0f}"
          f"   annualization: {args.annualization}")
    for pair in pairs:
        pair_id = pair["pair_id"]
        row = metrics.get(pair_id, {})
        left_cell = exposure.get(pair_id, {}).get(pair["x_symbol"], {})
        right_cell = exposure.get(pair_id, {}).get(pair["y_symbol"], {})
        active_cagr = None
        if row.get("net_pnl") and row.get("window_start") and row.get("window_end"):
            try:
                net = float(row["net_pnl"])
                active_days = (int(float(row["window_end"]))
                               - int(float(row["window_start"]))) / MILLISECONDS_PER_DAY
                if args.annualization == "active_window":
                    active_cagr = annualize(net, initial_equity, active_days)
                elif active_cagr is None:
                    active_cagr = annualize(net, initial_equity, window_days)
            except (TypeError, ValueError):
                active_cagr = None
        rows.append({
            "pair_id": pair_id,
            "symbol_left": pair["x_symbol"],
            "symbol_right": pair["y_symbol"],
            "tradfi": "True" if "tradfi" in (pair.get("tags") or []) else "False",
            "oi_left": left_cell.get("notional_mean", 0.0),
            "oi_right": right_cell.get("notional_mean", 0.0),
            "left_tw": left_cell.get("notional_tw", 0.0),
            "right_tw": right_cell.get("notional_tw", 0.0),
            "left_fills": int(left_cell.get("fills", 0)),
            "right_fills": int(right_cell.get("fills", 0)),
            "left_minutes": left_cell.get("minutes", 0.0),
            "right_minutes": right_cell.get("minutes", 0.0),
            "cagr": "" if active_cagr is None else f"{active_cagr:.6f}",
            "net_pnl": row.get("net_pnl") or "",
            "sharpe": row.get("sharpe") or "",
            "max_drawdown": row.get("max_drawdown") or "",
            "trades": row.get("closed_positions") or "",
            "turnover": row.get("turnover") or "",
        })

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    if args.enriched_out is not None:
        args.enriched_out.parent.mkdir(parents=True, exist_ok=True)

    def write_table(path: Path, selected: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(COLUMNS)
            for row in selected:
                writer.writerow([
                    row["symbol_left"], row["symbol_right"], row["tradfi"],
                    f"{row['oi_left']:.2f}" if row["oi_left"] else "",
                    f"{row['oi_right']:.2f}" if row["oi_right"] else "",
                    row["cagr"],
                ])

    write_table(args.csv_out, rows)

    if args.top10_max:
        concentration = load_concentration(args.concentration_csv)
        if not concentration:
            print(f"warning: no concentration data in {args.concentration_csv}, "
                  "skipping the filtered tables")
        for threshold in args.top10_max:
            tag = f"{threshold:.1f}".replace(".", "p")
            kept_pairs = [pair for pair in pairs
                          if passes_top10(pair, concentration, threshold)]
            kept_ids = {pair["pair_id"] for pair in kept_pairs}
            selected = [row for row in rows if row["pair_id"] in kept_ids]
            filtered_path = args.csv_out.with_name(
                f"{args.csv_out.stem}_top10le{tag}{args.csv_out.suffix}"
            )
            write_table(filtered_path, selected)
            tradfi_count = sum(1 for row in selected if row["tradfi"] == "True")
            with_trades = sum(1 for row in selected if row["cagr"] != "")
            print(f"top10 <= {threshold}: {len(selected)} rows"
                  f" (TradFi {tradfi_count}, crypto {len(selected) - tradfi_count},"
                  f" with trades {with_trades})  ->  {filtered_path.name}")

    if args.enriched_out is not None:
        header = ["pair_id", *COLUMNS, "左腿开仓次数", "右腿开仓次数",
                  "左腿时间加权持仓额", "右腿时间加权持仓额", "左腿持仓分钟", "右腿持仓分钟",
                  "每万美金槽位年化", "净盈亏_usdt", "sharpe", "max_drawdown", "交易数", "换手名义",
                  "traded"] + [f"年化收益率_{name[:26]}" for name in extra]
        args.enriched_out.parent.mkdir(parents=True, exist_ok=True)
        with args.enriched_out.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for row in rows:
                writer.writerow([
                    row["pair_id"], row["symbol_left"], row["symbol_right"], row["tradfi"],
                    f"{row['oi_left']:.2f}" if row["oi_left"] else "",
                    f"{row['oi_right']:.2f}" if row["oi_right"] else "",
                    row["cagr"],
                    row["left_fills"] or "", row["right_fills"] or "",
                    f"{row['left_tw']:.2f}" if row["left_tw"] else "",
                    f"{row['right_tw']:.2f}" if row["right_tw"] else "",
                    f"{row['left_minutes']:.0f}" if row["left_minutes"] else "",
                    f"{row['right_minutes']:.0f}" if row["right_minutes"] else "",
                    row["cagr"], row["net_pnl"], row["sharpe"], row["max_drawdown"],
                    row["trades"], row["turnover"], "yes" if row["cagr"] != "" else "no",
                    *[extra[name].get(row["pair_id"], {}).get("cagr", "") for name in extra],
                ])

    traded = [row for row in rows if row["cagr"] != ""]
    print(f"run      : {run_dir.name}")
    print(f"pairs    : {len(rows)}   with trades: {len(traded)}")
    print(f"wrote    : {args.csv_out}")
    if args.enriched_out is not None:
        print(f"wrote    : {args.enriched_out}")
    print("\n按净盈亏排序前 12（含两腿每次开仓平均持仓额 USDT）：")
    print("{:<15} {:<12} {:>12} {:<12} {:>12} {:>10} {:>9}".format(
        "pair_id", "left", "oi_left", "right", "oi_right", "年化", "净盈亏"))
    for row in sorted(traded, key=lambda item: -float(item["net_pnl"] or 0))[:12]:
        print("{:<15} {:<12} {:>12,.0f} {:<12} {:>12,.0f} {:>10} {:>9,.0f}".format(
            row["pair_id"], row["symbol_left"], row["oi_left"], row["symbol_right"],
            row["oi_right"], f"{float(row['cagr']):.1%}", float(row["net_pnl"])))
    left_total = sum(row["oi_left"] for row in traded)
    right_total = sum(row["oi_right"] for row in traded)
    print(f"\n所有有成交币对的两腿开仓均值合计: left {left_total:,.0f}  right {right_total:,.0f} USDT")
    left_fills = sum(row["left_fills"] for row in traded)
    right_fills = sum(row["right_fills"] for row in traded)
    print(f"开仓次数: left {left_fills}  right {right_fills}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
