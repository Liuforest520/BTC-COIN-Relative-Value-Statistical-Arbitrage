"""Build the pair sizing table: symbols, TradFi flag, open interest (USDT), annualised return.

Columns (exactly as requested) written to documents/pair_oi_return_table.csv:
    symbol_left, symbol_right, tradfi, oi_left, oi_right, 年化收益率

* tradfi        -- the pair carries the ``tradfi`` tag in the run config
* oi_left/right -- that symbol's contract open interest converted to USDT
                   (Binance ``/fapi/v1/openInterest`` x ``/fapi/v1/premiumIndex``
                   mark price; falls back to ``openInterestHist`` USDT value)
* 年化收益率     -- the pair's CAGR from the run's per-pair contribution curve
                   (curve_charts/pairs/pair_chart_metrics.csv); empty when the
                   pair never traded in that window

An enriched copy with liquidity caps lands in
research/outputs/pair_sizing_table/pair_sizing_table_enriched.csv

Usage:
    python research/scripts/build_pair_oi_return_table.py \
        --run-dir results/backtests/tls_log_price_full97_L2D_U4H_E3p0_X0p5_freeze
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

FAPI = "https://fapi.binance.com"
REQUEST_PAUSE = 0.12
COLUMNS = ["symbol_left", "symbol_right", "tradfi", "oi_left", "oi_right", "年化收益率"]


def http_json(url: str, timeout: float = 20.0, retries: int = 3):
    last_error = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            if exc.code == 400:  # unknown symbol: no point retrying
                break
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"{url} -> {last_error}")


def open_interest_usdt(symbol: str) -> tuple[float | None, str]:
    """Return (open interest in USDT, source).

    The historical endpoint already reports the USDT notional, so it needs a
    single small request per symbol; the live endpoint is the fallback and needs
    the mark price as well.  The bulk premiumIndex call is avoided on purpose:
    it is large enough to time out on a slow connection.
    """
    try:
        payload = http_json(
            f"{FAPI}/futures/data/openInterestHist?symbol={symbol}&period=5m&limit=1"
        )
        if payload:
            return float(payload[-1]["sumOpenInterestValue"]), "openInterestHist USDT"
    except Exception:
        pass
    try:
        contracts = float(http_json(f"{FAPI}/fapi/v1/openInterest?symbol={symbol}")["openInterest"])
        price = float(http_json(f"{FAPI}/fapi/v1/premiumIndex?symbol={symbol}")["markPrice"])
        return contracts * price, "openInterest x markPrice"
    except Exception:
        pass
    return None, "unavailable"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path,
                        default=PROJECT_ROOT / "documents" / "pair_oi_return_table.csv")
    parser.add_argument("--enriched-out", type=Path,
                        default=PROJECT_ROOT / "research" / "outputs" / "pair_sizing_table"
                        / "pair_sizing_table_enriched.csv")
    parser.add_argument("--leg-oi-limits", type=float, nargs="+", default=[0.002, 0.005, 0.01],
                        help="per-leg notional caps expressed as a share of that leg's OI")
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    setup = config["setups"][config["active_setup"]]
    pairs = setup["pairs"]

    per_pair: dict[str, dict] = {}
    metrics_path = run_dir / "curve_charts" / "pairs" / "pair_chart_metrics.csv"
    if metrics_path.exists():
        with metrics_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                per_pair[row["pair_id"]] = row

    symbols = sorted({pair["x_symbol"] for pair in pairs} | {pair["y_symbol"] for pair in pairs})
    print(f"run         : {run_dir.name}")
    print(f"pairs       : {len(pairs)}   distinct symbols: {len(symbols)}")

    oi: dict[str, float | None] = {}
    sources: dict[str, str] = {}
    for index, symbol in enumerate(symbols, start=1):
        value, source = open_interest_usdt(symbol)
        oi[symbol] = value
        sources[symbol] = source
        if index % 20 == 0 or index == len(symbols):
            print(f"  open interest {index}/{len(symbols)}")
        time.sleep(REQUEST_PAUSE)

    found = [value for value in oi.values() if value is not None]
    print(f"open interest: {len(found)}/{len(symbols)} symbols resolved")
    if found:
        ordered = sorted(found)
        print("  min {:,.0f}   median {:,.0f}   max {:,.0f} USDT".format(
            ordered[0], ordered[len(ordered) // 2], ordered[-1]))
    missing = [symbol for symbol, value in oi.items() if value is None]
    if missing:
        print(f"  no OI data: {len(missing)} -> {missing}")

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.enriched_out.parent.mkdir(parents=True, exist_ok=True)

    limit_columns = [f"leg_cap_{str(limit).replace('.', 'p')}pct_usdt" for limit in args.leg_oi_limits]
    with args.csv_out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for pair in pairs:
            row = per_pair.get(pair["pair_id"], {})
            cagr = row.get("cagr") or ""
            writer.writerow([
                pair["x_symbol"], pair["y_symbol"],
                "True" if "tradfi" in (pair.get("tags") or []) else "False",
                "" if oi.get(pair["x_symbol"]) is None else f"{oi[pair['x_symbol']]:.2f}",
                "" if oi.get(pair["y_symbol"]) is None else f"{oi[pair['y_symbol']]:.2f}",
                cagr,
            ])

    with args.enriched_out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "pair_id", *COLUMNS, "tags", "traded", "sharpe", "max_drawdown", "trades",
            "oi_min_usdt", *limit_columns,
        ])
        for pair in pairs:
            row = per_pair.get(pair["pair_id"], {})
            left, right = oi.get(pair["x_symbol"]), oi.get(pair["y_symbol"])
            known = [value for value in (left, right) if value is not None]
            minimum = min(known) if known else None
            writer.writerow([
                pair["pair_id"], pair["x_symbol"], pair["y_symbol"],
                "True" if "tradfi" in (pair.get("tags") or []) else "False",
                "" if left is None else f"{left:.2f}",
                "" if right is None else f"{right:.2f}",
                row.get("cagr") or "",
                "|".join(pair.get("tags") or []),
                "yes" if row else "no",
                row.get("sharpe") or "",
                row.get("max_drawdown") or "",
                row.get("closed_positions") or "",
                "" if minimum is None else f"{minimum:.2f}",
                *["" if minimum is None else f"{minimum * limit:.2f}" for limit in args.leg_oi_limits],
            ])

    print(f"\nwrote {args.csv_out}")
    print(f"wrote {args.enriched_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
