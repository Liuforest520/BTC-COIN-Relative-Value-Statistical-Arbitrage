"""Rebuild per-pair report metrics from an existing backtest directory.

This intentionally does not rerun the strategy.  It is useful after a reporting
fix, especially for compact runs whose position_curve.csv has no per-symbol
position-value columns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import polars as pl

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.modules.reporting.pair_summary import (
    build_pair_summary,
    compact_pair_summary_markdown,
    pair_defs_from_config,
    write_compact_pair_summary_csv,
)
from core.modules.reporting.trade_review import export_trade_review_html


ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _read_csv(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame()
    return pl.read_csv(path, infer_schema_length=10_000, ignore_errors=True)


def reexport(run_dir: Path, include_trade_review: bool = False) -> None:
    run_dir = run_dir.resolve()
    config_path = run_dir / "config.yaml"
    metrics_path = run_dir / "metrics.json"
    if not config_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"not a complete backtest report: {run_dir}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    final_valuation_path = run_dir / "final_position_valuation.json"
    final_valuation = (
        json.loads(final_valuation_path.read_text(encoding="utf-8"))
        if final_valuation_path.exists()
        else {}
    )
    pair_summary = build_pair_summary(
        trades=_read_csv(run_dir / "trades.csv"),
        position_curve=_read_csv(run_dir / "position_curve.csv"),
        funding_payments=_read_csv(run_dir / "funding_payments.csv"),
        pair_defs=pair_defs_from_config(config_path),
        initial_equity=metrics.get("initial_equity"),
        price_source=config_path,
        timeline=_read_csv(run_dir / "equity_curve.csv"),
        final_position_valuation=final_valuation,
    )
    write_compact_pair_summary_csv(run_dir / "pair_metrics.csv", pair_summary)
    (run_dir / "pair_metrics_summary.md").write_text(
        compact_pair_summary_markdown(pair_summary), encoding="utf-8"
    )
    if include_trade_review:
        export_trade_review_html(run_dir)
    print(f"re-exported pair metrics: {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", help="existing results/backtests run directories")
    parser.add_argument("--trade-review", action="store_true", help="also regenerate Trade Review HTML")
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        reexport(_resolve(run_dir), include_trade_review=args.trade_review)


if __name__ == "__main__":
    main()
