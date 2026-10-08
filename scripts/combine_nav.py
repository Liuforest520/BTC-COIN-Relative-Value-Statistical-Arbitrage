"""Combine train/OOS backtest equity curves into a daily NAV and summary metrics."""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path

DEFAULT_TRAIN = "results/backtests/single10_02_L2D_U4H_MA60_E3p0_ADF0p4_drop12"
DEFAULT_OOS = "results/backtests/single10_02_L2D_U4H_MA60_E3p0_ADF0p4_drop12_oos_tradfi"
DEFAULT_OUTPUT = "results/backtests/single10_02_L2D_U4H_MA60_E3p0_ADF0p4_drop12_combined"


def read_equity_curve(path: Path) -> list[tuple[datetime, float]]:
    rows: list[tuple[datetime, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            ts = int(float(row["ts"]))
            equity = float(row["equity"])
            if math.isfinite(equity):
                rows.append((datetime.fromtimestamp(ts / 1000, tz=timezone.utc), equity))
    if not rows:
        raise ValueError(f"empty equity curve: {path}")
    rows.sort(key=lambda item: item[0])
    return rows


def daily_last(rows: list[tuple[datetime, float]]) -> list[tuple[date, float]]:
    result: dict[date, float] = {}
    for timestamp, equity in rows:
        result[timestamp.date()] = equity
    return sorted(result.items())


def combine_nav(train: list[tuple[datetime, float]], oos: list[tuple[datetime, float]]) -> list[tuple[date, float]]:
    train_initial = train[0][1]
    oos_initial = oos[0][1]
    if train_initial <= 0 or oos_initial <= 0:
        raise ValueError("initial equity must be positive")
    train_daily = daily_last(train)
    oos_daily = daily_last(oos)
    train_last_nav = train_daily[-1][1] / train_initial
    combined: dict[date, float] = {day: equity / train_initial for day, equity in train_daily}
    for day, equity in oos_daily:
        combined[day] = train_last_nav * (equity / oos_initial)
    return sorted(combined.items())


def daily_returns(nav: list[tuple[date, float]]) -> list[float]:
    return [current / previous - 1.0 for (_, previous), (_, current) in zip(nav, nav[1:])]


def max_drawdown(nav: list[tuple[date, float]]) -> tuple[float, int]:
    peak = nav[0][1]
    peak_day = nav[0][0]
    worst = 0.0
    longest_days = 0
    underwater_start: date | None = None
    for day, value in nav:
        if value >= peak:
            if underwater_start is not None:
                longest_days = max(longest_days, (day - underwater_start).days)
                underwater_start = None
            peak = value
            peak_day = day
            continue
        if underwater_start is None:
            underwater_start = peak_day
        worst = min(worst, value / peak - 1.0)
    if underwater_start is not None:
        longest_days = max(longest_days, (nav[-1][0] - underwater_start).days)
    return worst, longest_days


def cagr(nav: list[tuple[date, float]]) -> float | None:
    elapsed_days = (nav[-1][0] - nav[0][0]).days
    if elapsed_days <= 0 or nav[0][1] <= 0 or nav[-1][1] < 0:
        return None
    if nav[-1][1] == 0:
        return -1.0
    return (nav[-1][1] / nav[0][1]) ** (365.0 / elapsed_days) - 1.0


def annualized_return_for_year(nav: list[tuple[date, float]], year: int) -> float | None:
    points = [(day, value) for day, value in nav if day.year == year]
    if len(points) < 2 or points[0][1] <= 0 or points[-1][1] < 0:
        return None
    elapsed_days = (points[-1][0] - points[0][0]).days
    if elapsed_days <= 0:
        return points[-1][1] / points[0][1] - 1.0
    if points[-1][1] == 0:
        return -1.0
    return (points[-1][1] / points[0][1]) ** (365.0 / elapsed_days) - 1.0


def calculate_metrics(nav: list[tuple[date, float]]) -> dict:
    returns = daily_returns(nav)
    mean_return = sum(returns) / len(returns) if returns else None
    variance = (sum((value - mean_return) ** 2 for value in returns) / (len(returns) - 1) if returns and len(returns) > 1 else None)
    volatility = math.sqrt(variance) if variance is not None else None
    sharpe = mean_return / volatility * math.sqrt(365.0) if volatility else None
    downside = [min(value, 0.0) for value in returns]
    downside_variance = (sum(value * value for value in downside) / (len(downside) - 1) if len(downside) > 1 else None)
    downside_deviation = math.sqrt(downside_variance) if downside_variance else None
    sortino = mean_return / downside_deviation * math.sqrt(365.0) if mean_return is not None and downside_deviation else None
    drawdown, longest_dd_days = max_drawdown(nav)
    annualized = cagr(nav)
    return {
        "start_date": nav[0][0].isoformat(),
        "end_date": nav[-1][0].isoformat(),
        "nav_start": nav[0][1],
        "nav_end": nav[-1][1],
        "cumulative_return": nav[-1][1] / nav[0][1] - 1.0,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": drawdown,
        "longest_dd_days": longest_dd_days,
        "calmar": annualized / abs(drawdown) if annualized is not None and drawdown < 0 else None,
        "arr_2024": annualized_return_for_year(nav, 2024),
        "arr_2025": annualized_return_for_year(nav, 2025),
        "arr_2026": annualized_return_for_year(nav, 2026),
        "annualized_return_full_period": annualized,
        "daily_observations": len(nav),
    }


def write_outputs(nav, metrics, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "combined_nav_daily.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "nav"])
        for day, value in nav:
            writer.writerow([day.isoformat(), f"{value:.12f}"])
    with (output_dir / "combined_nav_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    labels = {
        "cumulative_return": "Cumulative Return", "sharpe": "Sharpe", "sortino": "Sortino",
        "max_drawdown": "Max Drawdown", "longest_dd_days": "Longest DD Days", "calmar": "Calmar",
        "arr_2024": "ARR in 2024", "arr_2025": "ARR in 2025", "arr_2026": "ARR in 2026",
    }
    with (output_dir / "combined_nav_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# Combined NAV Summary\n\n")
        handle.write(f"Period: {metrics['start_date']} to {metrics['end_date']}\n\n")
        handle.write("| Metric | Value |\n|---|---:|\n")
        for key, label in labels.items():
            value = metrics[key]
            if value is None: rendered = "N/A"
            elif key == "longest_dd_days": rendered = str(value)
            elif key in {"calmar", "sharpe", "sortino"}: rendered = f"{value:.4f}"
            else: rendered = f"{value:.2%}"
            handle.write(f"| {label} | {rendered} |\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=DEFAULT_TRAIN, type=Path)
    parser.add_argument("--oos", default=DEFAULT_OOS, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT, type=Path)
    args = parser.parse_args()
    train = read_equity_curve(args.train / "equity_curve.csv")
    oos = read_equity_curve(args.oos / "equity_curve.csv")
    if oos[0][0] <= train[-1][0]:
        raise ValueError("OOS equity curve must start after the training equity curve")
    nav = combine_nav(train, oos)
    metrics = calculate_metrics(nav)
    write_outputs(nav, metrics, args.output_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"output_dir={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
