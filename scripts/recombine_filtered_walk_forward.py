"""Recombine an existing portfolio result after removing selected pairs.

This is a diagnostic tool. It does not re-run the strategy or re-optimize
parameters; it removes the selected pairs' reconstructed mark-to-market PnL
from an existing portfolio curve and writes a filtered diagnostic result.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import polars as pl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.modules.metrics import calculate_metrics
from core.modules.reporting.pair_summary import (
    _reconstructed_pair_curves,
    pair_defs_from_config,
)


DEFAULT_REMOVED = (
    "soxl_soxs,amd_mu,mu_sndk,sqqq_tqqq,nvda_vrt,nbis_vrt,mu_muu,"
    "nvda_nvdl,iwm_tza,dell_vrt,tbt_tmf,tsla_tsll"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--remove-pairs",
        default=DEFAULT_REMOVED,
        help="Comma-separated pair ids to remove.",
    )
    return parser.parse_args()


def read_frame(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, infer_schema_length=1000, try_parse_dates=False)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def filter_frame(frame: pl.DataFrame, removed: set[str]) -> pl.DataFrame:
    if "pair_id" not in frame.columns:
        return frame
    return frame.filter(
        pl.col("pair_id").is_null() | ~pl.col("pair_id").is_in(sorted(removed))
    )


def main() -> None:
    args = parse_args()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    removed = {item.strip() for item in args.remove_pairs.split(",") if item.strip()}
    if not removed:
        raise ValueError("--remove-pairs must contain at least one pair id")

    equity = read_frame(source / "portfolio_equity_curve.csv").select("ts", "equity")
    trades = read_frame(source / "trades.csv")
    trades = trades.with_columns(
        pl.when(pl.col("side") == "sell")
        .then(pl.col("notional") - pl.col("fee"))
        .otherwise(-pl.col("notional") - pl.col("fee"))
        .alias("cashflow")
    )
    orders = read_frame(source / "orders.csv")
    funding = read_frame(source / "funding_payments.csv")

    config_path = source / "backtest_report" / "config.yaml"
    pair_defs = {
        item["pair_id"]: (item.get("x_symbol"), item.get("y_symbol"))
        for item in pair_defs_from_config(config_path)
    }
    missing = sorted(pair_id for pair_id in removed if pair_id not in pair_defs)
    if missing:
        raise ValueError(f"removed pair ids are not present in the source config: {missing}")

    funding_events = funding.to_dicts()
    removed_pnl = equity.select("ts").with_columns(pl.lit(0.0).alias("removed_pair_pnl"))
    removed_rows = []

    for pair_id in sorted(removed):
        curves = _reconstructed_pair_curves(
            trades=trades,
            position_curve=pl.DataFrame(),
            pair_symbols={pair_id: pair_defs[pair_id]},
            funding_events=funding_events,
            price_source=config_path,
            timeline=equity,
            final_position_valuation=None,
        )
        curve = curves.get(pair_id)
        if curve is None or curve.is_empty():
            raise ValueError(f"could not reconstruct PnL curve for {pair_id}")
        contribution = curve.select(
            "ts", pl.col("pair_pnl").fill_null(0.0).alias("pair_pnl")
        )
        removed_pnl = removed_pnl.join(contribution, on="ts", how="left").with_columns(
            (pl.col("removed_pair_pnl") + pl.col("pair_pnl").fill_null(0.0)).alias(
                "removed_pair_pnl"
            )
        ).drop("pair_pnl")

        pair_row = (
            read_frame(source / "backtest_report" / "pair_metrics.csv")
            .filter(pl.col("交易对") == pair_id)
        )
        if not pair_row.is_empty():
            removed_rows.extend(pair_row.to_dicts())

    filtered_curve = (
        equity.join(removed_pnl, on="ts", how="left")
        .with_columns(
            (pl.col("equity") - pl.col("removed_pair_pnl")).alias("equity")
        )
        .select("ts", "equity")
    )
    filtered_trades = filter_frame(trades, removed)
    filtered_orders = filter_frame(orders, removed)
    filtered_funding = filter_frame(funding, removed)
    metrics = calculate_metrics(
        filtered_curve,
        orders=filtered_orders,
        trades=filtered_trades,
        funding_payments=filtered_funding,
    )

    filtered_curve.write_csv(output / "portfolio_equity_curve.csv")
    filtered_curve.filter(pl.col("ts") >= 1704067200000).write_csv(
        output / "oos_equity_curve.csv"
    )
    filtered_trades.write_csv(output / "trades.csv")
    filtered_orders.write_csv(output / "orders.csv")
    filtered_funding.write_csv(output / "funding_payments.csv")
    removed_pnl.write_csv(output / "removed_pair_pnl_curve.csv")
    pl.DataFrame(removed_rows).write_csv(output / "removed_pair_metrics.csv")
    write_json(output / "metrics.json", metrics)
    pl.DataFrame(
        [{"metric": key, "value": value} for key, value in metrics.items()]
    ).write_csv(output / "metrics.csv")

    source_final = json.loads((source / "final_position_valuation.json").read_text(encoding="utf-8"))
    write_json(output / "final_position_valuation.json", source_final)
    schedule_path = source / "parameter_schedule.json"
    if schedule_path.exists():
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        write_json(
            output / "parameter_schedule.json",
            {pair_id: value for pair_id, value in schedule.items() if pair_id not in removed},
        )
    if config_path.exists():
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        for setup in (raw_config.get("setups") or {}).values():
            pairs = setup.get("pairs") or []
            for pair in pairs:
                if pair.get("pair_id") in removed:
                    pair["enabled"] = False
        (output / "filtered_config.yaml").write_text(
            yaml.safe_dump(raw_config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    (output / "source_result.txt").write_text(str(source) + "\n", encoding="utf-8")
    (output / "removed_pairs.txt").write_text(
        "\n".join(sorted(removed)) + "\n", encoding="utf-8"
    )
    (output / "diagnostic_note.md").write_text(
        "# Filtered Walk-Forward Diagnostic\n\n"
        "This result subtracts the reconstructed mark-to-market PnL of the "
        "removed pairs from the existing portfolio curve. It does not rerun "
        "parameter optimization or strategy execution, so it is a diagnostic "
        "comparison rather than an independent OOS backtest.\n",
        encoding="utf-8",
    )

    print(json.dumps({"output_dir": str(output), "removed_pairs": sorted(removed), "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
