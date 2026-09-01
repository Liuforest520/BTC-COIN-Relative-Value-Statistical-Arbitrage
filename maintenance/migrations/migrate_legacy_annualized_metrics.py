from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path


MILLISECONDS_PER_YEAR = 365 * 24 * 60 * 60 * 1000


def _finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _equity_time_range(path: Path) -> tuple[int, int]:
    first_ts = None
    last_ts = None
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            timestamp = _finite_float(row.get("ts"))
            equity = _finite_float(row.get("equity"))
            if timestamp is None or equity is None:
                continue
            timestamp = int(timestamp)
            first_ts = timestamp if first_ts is None else min(first_ts, timestamp)
            last_ts = timestamp if last_ts is None else max(last_ts, timestamp)
    if first_ts is None or last_ts is None or last_ts <= first_ts:
        raise ValueError(f"cannot determine a positive equity time range from {path}")
    return first_ts, last_ts


def _cagr(total_return: float | None, elapsed_years: float) -> float | None:
    if total_return is None or total_return < -1.0:
        return None
    if total_return == -1.0:
        return -1.0
    value = (1.0 + total_return) ** (1.0 / elapsed_years) - 1.0
    return value if math.isfinite(value) else None


def _calmar(cagr: float | None, max_drawdown: float | None) -> float | None:
    if cagr is None or max_drawdown is None or max_drawdown > 0:
        return None
    if max_drawdown < 0:
        return cagr / abs(max_drawdown)
    if cagr > 0:
        return math.inf
    if cagr < 0:
        return -math.inf
    return 0.0


def migrate(results_path: Path, equity_curve_path: Path) -> int:
    start_ts, end_ts = _equity_time_range(equity_curve_path)
    elapsed_years = (end_ts - start_ts) / MILLISECONDS_PER_YEAR

    with results_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    required = {"total_return", "annualized_return", "max_drawdown", "calmar"}
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(f"{results_path} is missing columns: {sorted(missing)}")

    if "annualized_mean_return" not in fieldnames:
        fieldnames.insert(fieldnames.index("annualized_return") + 1, "annualized_mean_return")

    changed = 0
    for row in rows:
        if row.get("error"):
            continue
        if not row.get("annualized_mean_return"):
            row["annualized_mean_return"] = row.get("annualized_return", "")
        annualized_return = _cagr(_finite_float(row.get("total_return")), elapsed_years)
        calmar = _calmar(annualized_return, _finite_float(row.get("max_drawdown")))
        row["annualized_return"] = "" if annualized_return is None else repr(annualized_return)
        row["calmar"] = "" if calmar is None else repr(calmar)
        changed += 1

    temporary_path = results_path.with_suffix(results_path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, results_path)

    print(
        f"updated={changed} elapsed_years={elapsed_years:.9f} "
        f"start_ts={start_ts} end_ts={end_ts} path={results_path}"
    )
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace legacy arithmetic annualized returns with CAGR in a sweep CSV."
    )
    parser.add_argument("results_csv", type=Path)
    parser.add_argument(
        "equity_curve_csv",
        type=Path,
        help="Representative equity curve with the same backtest time range as every CSV row.",
    )
    args = parser.parse_args()
    migrate(args.results_csv, args.equity_curve_csv)


if __name__ == "__main__":
    main()
