from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.modules.metrics import performance_metrics


PERFORMANCE_KEYS = (
    "initial_equity",
    "final_equity",
    "total_return",
    "annualized_return",
    "annualized_mean_return",
    "annualized_volatility",
    "sharpe",
    "calmar",
    "max_drawdown",
)


def _atomic_text(path: Path, text: str) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    os.replace(temporary_path, path)


def _write_metrics_csv(path: Path, metrics: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows({"metric": key, "value": value} for key, value in metrics.items())
    os.replace(temporary_path, path)


def _update_key_metrics(path: Path, metrics: dict) -> None:
    if not path.exists():
        return
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    for row in rows:
        key = row.get("key")
        if key in metrics:
            row["value"] = metrics[key]
        if key == "annualized_return":
            row["metric"] = "复合年化收益（CAGR）"
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def _update_summary(path: Path, metrics: dict) -> None:
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    replacements = {
        "| 业绩 | 年化收益 |": (
            "| 业绩 | 复合年化收益（CAGR） |",
            metrics["annualized_return"],
        ),
        "| 业绩 | 复合年化收益（CAGR） |": (
            "| 业绩 | 复合年化收益（CAGR） |",
            metrics["annualized_return"],
        ),
        "| 业绩 | 卡玛 |": ("| 业绩 | 卡玛 |", metrics["calmar"]),
    }
    for index, line in enumerate(lines):
        for prefix, (new_prefix, value) in replacements.items():
            if line.startswith(prefix):
                lines[index] = f"{new_prefix} {value} |"
                break
    _atomic_text(path, "\n".join(lines) + "\n")


def migrate_report(report_dir: Path) -> None:
    equity_path = report_dir / "equity_curve.csv"
    metrics_json_path = report_dir / "metrics.json"
    if not equity_path.exists() or not metrics_json_path.exists():
        raise FileNotFoundError(f"missing equity_curve.csv or metrics.json in {report_dir}")

    equity = pl.read_csv(equity_path, columns=["ts", "equity"])
    corrected = performance_metrics(equity)
    metrics = json.loads(metrics_json_path.read_text(encoding="utf-8"))
    for key in PERFORMANCE_KEYS:
        metrics[key] = corrected[key]

    _atomic_text(metrics_json_path, json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    _write_metrics_csv(report_dir / "metrics.csv", metrics)
    _update_key_metrics(report_dir / "key_metrics.csv", metrics)
    _update_summary(report_dir / "summary.md", metrics)
    print(
        f"path={report_dir} total_return={metrics['total_return']:.9f} "
        f"cagr={metrics['annualized_return']:.9f} calmar={metrics['calmar']:.9f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recalculate CAGR and dependent metrics in legacy detailed backtest reports."
    )
    parser.add_argument("report_root", type=Path)
    args = parser.parse_args()

    report_dirs = sorted(
        path.parent for path in args.report_root.glob("*/equity_curve.csv")
    )
    if not report_dirs:
        raise FileNotFoundError(f"no report directories found below {args.report_root}")
    for report_dir in report_dirs:
        migrate_report(report_dir)


if __name__ == "__main__":
    main()
