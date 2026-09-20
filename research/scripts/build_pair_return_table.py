"""Pair return table (no network): symbols, TradFi flag, per-pair return, OI placeholders.

Columns written to documents/pair_oi_return_table.csv (exactly as requested):
    symbol_left, symbol_right, tradfi, oi_left, oi_right, 年化收益率

* tradfi     -- pair carries the ``tradfi`` tag in the run config
* oi_left / oi_right -- contract open interest in USDT; the columns are present
  but empty here because the Binance fetch is done separately (slow network):
  run research/scripts/build_pair_oi_return_table.py to fill them.
* 年化收益率  -- the pair's CAGR from the run's per-pair contribution curve
  (curve_charts/pairs/pair_chart_metrics.csv, i.e. equity = initial + pair pnl).
  Empty when the pair never traded inside the backtest window.

Usage:
    python research/scripts/build_pair_return_table.py \
        --run-dir results/backtests/tls_log_price_full97_L2D_U4H_E3p0_X0p5_freeze
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

COLUMNS = ["symbol_left", "symbol_right", "tradfi", "oi_left", "oi_right", "年化收益率"]


def load_pairs(run_dir: Path):
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    setup = config["setups"][config["active_setup"]]
    return setup["pairs"]


def load_pair_metrics(run_dir: Path) -> dict[str, dict]:
    path = run_dir / "curve_charts" / "pairs" / "pair_chart_metrics.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["pair_id"]: row for row in csv.DictReader(handle)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path,
                        default=PROJECT_ROOT / "documents" / "pair_oi_return_table.csv")
    parser.add_argument("--enriched-out", type=Path,
                        default=PROJECT_ROOT / "research" / "outputs" / "pair_sizing_table"
                        / "pair_sizing_table_enriched.csv")
    parser.add_argument("--compare-runs", type=Path, nargs="*", default=[],
                        help="extra run dirs whose per-pair CAGR is added to the enriched table")
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    pairs = load_pairs(run_dir)
    metrics = load_pair_metrics(run_dir)
    extra = {}
    for extra_run in args.compare_runs:
        extra_run = extra_run if extra_run.is_absolute() else PROJECT_ROOT / extra_run
        extra[extra_run.name] = load_pair_metrics(extra_run)

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.enriched_out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for pair in pairs:
        row = metrics.get(pair["pair_id"], {})
        tradfi = "True" if "tradfi" in (pair.get("tags") or []) else "False"
        slot_annualized = ""
        if row.get("net_pnl") and row.get("window_start") and row.get("window_end"):
            try:
                net = float(row["net_pnl"])
                days = (int(float(row["window_end"])) - int(float(row["window_start"]))) / 86_400_000.0
                if days > 1 and net > -10_000:
                    slot_annualized = (1.0 + net / 10_000.0) ** (365.0 / days) - 1.0
            except (TypeError, ValueError):
                slot_annualized = ""
        rows.append({
            "pair_id": pair["pair_id"],
            "symbol_left": pair["x_symbol"],
            "symbol_right": pair["y_symbol"],
            "tradfi": tradfi,
            "cagr": row.get("cagr") or "",
            "slot_cagr": "" if slot_annualized == "" else f"{slot_annualized:.6f}",
            "net_pnl": row.get("net_pnl") or "",
            "sharpe": row.get("sharpe") or "",
            "max_drawdown": row.get("max_drawdown") or "",
            "trades": row.get("closed_positions") or "",
            "turnover": row.get("turnover") or "",
        })

    with args.csv_out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([row["symbol_left"], row["symbol_right"], row["tradfi"],
                             "", "", row["cagr"]])

    header = ["pair_id", *COLUMNS, "每万美金槽位年化", "净盈亏_usdt", "sharpe", "max_drawdown",
              "交易数", "换手名义", "traded"] + [f"年化收益率_{name[:28]}" for name in extra]
    with args.enriched_out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([
                row["pair_id"], row["symbol_left"], row["symbol_right"], row["tradfi"],
                "", "", row["cagr"], row["slot_cagr"], row["net_pnl"], row["sharpe"],
                row["max_drawdown"], row["trades"], row["turnover"],
                "yes" if row["cagr"] != "" else "no",
                *[extra[name].get(row["pair_id"], {}).get("cagr", "") for name in extra],
            ])

    traded = [row for row in rows if row["cagr"] != ""]
    tradfi_rows = [row for row in rows if row["tradfi"] == "True"]
    print(f"run            : {run_dir.name}")
    print(f"pairs          : {len(rows)}  (tradfi {len(tradfi_rows)}, 有收益数据 {len(traded)})")
    print(f"wrote          : {args.csv_out}")
    print(f"wrote          : {args.enriched_out}")
    print("\n前 15 行预览：")
    print("{:<16} {:<14} {:<8} {:>10}".format("pair_id", "symbol_left", "tradfi", "年化"))
    for row in rows[:15]:
        cagr = f"{float(row['cagr']):.2%}" if row["cagr"] else "-"
        print("{:<16} {:<14} {:<8} {:>10}".format(row["pair_id"], row["symbol_left"],
                                                  row["tradfi"], cagr))
    ordered = sorted(traded, key=lambda row: -float(row["cagr"]))
    print("\n年化收益率前 10：", [(row["pair_id"], f"{float(row['cagr']):.0%}") for row in ordered[:10]])
    print("年化收益率后 10：", [(row["pair_id"], f"{float(row['cagr']):.0%}") for row in ordered[-10:]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
