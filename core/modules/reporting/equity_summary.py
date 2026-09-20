from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import is_dataclass
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np


MAX_PLOT_POINTS = 8000


def export_equity_summary_image(result, output_path: Path) -> Path:
    """Write the portfolio metrics and equity curve to one static PNG."""
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".png":
        raise ValueError("equity summary output must be a .png file")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamps, equities = _equity_arrays(getattr(result, "equity_curve", []))
    plot_ts, plot_equity = _downsample_extrema(timestamps, equities, MAX_PLOT_POINTS)
    metrics = getattr(result, "metrics", {}) or {}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    figure = plt.figure(figsize=(16, 9), dpi=160, facecolor="#f6f7f9")
    axis = figure.add_axes((0.07, 0.09, 0.90, 0.62), facecolor="#ffffff")
    figure.text(
        0.04,
        0.955,
        "Portfolio Summary",
        fontsize=22,
        fontweight="bold",
        color="#17202a",
        va="top",
    )

    if len(timestamps):
        period = (
            f"{_format_date(timestamps[0])}  -  {_format_date(timestamps[-1])}"
        )
        figure.text(0.96, 0.948, period, fontsize=10, color="#667085", ha="right")

    _draw_metric_cards(figure, metrics)

    if len(plot_ts):
        dates = [
            datetime.fromtimestamp(float(ts) / 1000.0, tz=timezone.utc).replace(tzinfo=None)
            for ts in plot_ts
        ]
        axis.plot(dates, plot_equity, color="#2563eb", linewidth=1.45, label="Equity")

        initial_equity = _safe_float(metrics.get("initial_equity"))
        if initial_equity is not None:
            axis.axhline(
                initial_equity,
                color="#98a2b3",
                linewidth=0.9,
                linestyle=(0, (4, 4)),
                label="Initial equity",
            )

        peak = np.maximum.accumulate(equities)
        drawdowns = np.divide(
            equities,
            peak,
            out=np.ones_like(equities),
            where=np.abs(peak) > 1e-12,
        ) - 1.0
        trough_index = int(np.argmin(drawdowns))
        trough_date = datetime.fromtimestamp(
            float(timestamps[trough_index]) / 1000.0,
            tz=timezone.utc,
        ).replace(tzinfo=None)
        axis.scatter(
            [trough_date],
            [equities[trough_index]],
            color="#dc2626",
            s=28,
            zorder=4,
            label=f"Max drawdown {drawdowns[trough_index]:.2%}",
        )

        axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        axis.margins(x=0.005)
    else:
        axis.text(
            0.5,
            0.5,
            "No equity data",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="#667085",
            fontsize=16,
        )

    axis.set_title("Equity Curve", loc="left", fontsize=14, fontweight="bold", pad=14)
    axis.set_ylabel("Equity")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axis.grid(axis="both", color="#e7ebf2", linewidth=0.8)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color("#d7dde8")
    axis.tick_params(colors="#667085", labelsize=9)
    if len(plot_ts):
        axis.legend(loc="upper left", frameon=False, ncol=3, fontsize=9)

    temporary_path = output_path.with_name(f".{output_path.stem}.tmp.png")
    try:
        figure.savefig(temporary_path, format="png", facecolor=figure.get_facecolor())
        temporary_path.replace(output_path)
    finally:
        plt.close(figure)
        temporary_path.unlink(missing_ok=True)

    # Stable run names can leave the legacy HTML behind after a report rerun.
    if output_path.name == "portfolio_summary.png":
        output_path.with_suffix(".html").unlink(missing_ok=True)
    return output_path


def export_equity_summary_image_from_run_dir(
    run_dir: Path,
    output_path: Path | None = None,
) -> Path:
    """Regenerate the static portfolio image from an existing backtest."""
    import polars as pl

    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    equity_path = run_dir / "equity_curve.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics file not found: {metrics_path}")
    if not equity_path.is_file():
        raise FileNotFoundError(f"equity curve not found: {equity_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    equity_curve = pl.read_csv(equity_path, columns=["ts", "equity"])
    result = SimpleNamespace(metrics=metrics, equity_curve=equity_curve)
    output = export_equity_summary_image(
        result,
        output_path or run_dir / "portfolio_summary.png",
    )
    _replace_legacy_summary_reference(run_dir)
    return output


def _draw_metric_cards(figure, metrics: dict) -> None:
    cards = [
        ("Final Equity", metrics.get("final_equity"), "money"),
        ("Total Return", metrics.get("total_return"), "percent"),
        ("Sharpe", metrics.get("sharpe"), "number"),
        ("Max Drawdown", metrics.get("max_drawdown"), "percent"),
        ("Trades", metrics.get("trade_count"), "integer"),
        ("Win Rate", metrics.get("win_rate"), "percent"),
        ("Total Fee", metrics.get("total_fee"), "money"),
        ("Total Slippage", metrics.get("total_slippage"), "money"),
    ]
    left, right = 0.04, 0.96
    gap = 0.012
    width = (right - left - 3 * gap) / 4
    for index, (label, value, kind) in enumerate(cards):
        row, column = divmod(index, 4)
        x = left + column * (width + gap)
        y = 0.875 - row * 0.088
        figure.text(
            x,
            y,
            f"{label}\n{_format_metric(value, kind)}",
            fontsize=10,
            color="#667085",
            va="top",
            linespacing=1.7,
            bbox={
                "boxstyle": "round,pad=0.65",
                "facecolor": "#ffffff",
                "edgecolor": "#d7dde8",
                "linewidth": 0.8,
            },
        )


def _equity_arrays(rows) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(rows, dict):
        timestamp_values = rows.get("ts", [])
        equity_values = rows.get("equity", [])
        if len(timestamp_values) != len(equity_values):
            return np.array([], dtype=np.int64), np.array([], dtype=float)
        try:
            timestamps = np.asarray(timestamp_values, dtype=np.int64)
            equities = np.asarray(equity_values, dtype=float)
        except (TypeError, ValueError):
            return np.array([], dtype=np.int64), np.array([], dtype=float)
    elif hasattr(rows, "columns") and hasattr(rows, "get_column"):
        if "ts" not in rows.columns or "equity" not in rows.columns:
            return np.array([], dtype=np.int64), np.array([], dtype=float)
        timestamps = np.asarray(rows.get_column("ts").to_numpy(), dtype=np.int64)
        equities = np.asarray(rows.get_column("equity").to_numpy(), dtype=float)
    else:
        timestamp_values = []
        equity_values = []
        for row in rows or []:
            if is_dataclass(row):
                ts = getattr(row, "ts", None)
                equity = getattr(row, "equity", None)
            elif isinstance(row, dict):
                ts = row.get("ts")
                equity = row.get("equity")
            else:
                ts = getattr(row, "ts", None)
                equity = getattr(row, "equity", None)
            safe_ts = _safe_int(ts)
            safe_equity = _safe_float(equity)
            if safe_ts is None or safe_equity is None:
                continue
            timestamp_values.append(safe_ts)
            equity_values.append(safe_equity)
        timestamps = np.asarray(timestamp_values, dtype=np.int64)
        equities = np.asarray(equity_values, dtype=float)

    valid = np.isfinite(equities)
    timestamps = timestamps[valid]
    equities = equities[valid]
    if len(timestamps) > 1 and np.any(timestamps[1:] < timestamps[:-1]):
        order = np.argsort(timestamps, kind="stable")
        timestamps = timestamps[order]
        equities = equities[order]
    return timestamps, equities


def _downsample_extrema(
    timestamps: np.ndarray,
    equities: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep chronological bucket minima/maxima so drawdowns remain visible."""
    count = len(timestamps)
    if count <= max_points or max_points < 4:
        return timestamps, equities

    bucket_count = max(1, (max_points - 2) // 2)
    edges = np.linspace(1, count - 1, bucket_count + 1, dtype=np.int64)
    selected = [0]
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        values = equities[start:end]
        minimum = start + int(np.argmin(values))
        maximum = start + int(np.argmax(values))
        selected.extend(sorted({minimum, maximum}))
    selected.append(count - 1)
    indices = np.asarray(sorted(set(selected)), dtype=np.int64)
    return timestamps[indices], equities[indices]


def _format_metric(value, kind: str) -> str:
    number = _safe_float(value)
    if number is None:
        return "-"
    if kind == "percent":
        return f"{number:.2%}"
    if kind == "integer":
        return f"{int(round(number)):,}"
    if kind == "money":
        return f"{number:,.2f}"
    return f"{number:.3f}"


def _format_date(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(
        float(timestamp_ms) / 1000.0,
        tz=timezone.utc,
    ).strftime("%Y-%m-%d")


def _replace_legacy_summary_reference(run_dir: Path) -> None:
    summary_path = Path(run_dir) / "summary.md"
    if not summary_path.is_file():
        return
    content = summary_path.read_text(encoding="utf-8")
    updated = content.replace(
        "- `portfolio_summary.html`: 总资金曲线交互网页",
        "- `portfolio_summary.png`: 总资金曲线与核心指标图片",
    )
    if updated != content:
        summary_path.write_text(updated, encoding="utf-8")


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


def main() -> None:
    parser = ArgumentParser(description="Generate a portfolio summary PNG from a backtest result")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = export_equity_summary_image_from_run_dir(args.run_dir, args.output)
    print(output)


if __name__ == "__main__":
    main()
