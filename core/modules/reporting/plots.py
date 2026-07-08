from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl

from core.modules.reporting.utils import normalize_rows


def plot_backtest_summary(result, output_path="results/plots/backtest_summary.png"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    equity = equity_frame(result.equity_curve)
    positions = position_frame(result.position_curve)
    setup_chinese_font()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    fig.patch.set_facecolor("#f7f8fa")

    plot_equity_and_drawdown(axes[0], equity)
    plot_position_ratios(axes[1], positions)
    format_time_axis(axes[1])

    fig.suptitle("回测资金曲线与持仓比例", fontsize=18, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_drawdown(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    equity = equity_frame(result.equity_curve)
    setup_chinese_font()

    fig, ax = plt.subplots(figsize=(15, 4.8))
    dates = equity["datetime"].to_list()
    drawdown = equity["drawdown"].to_list()
    ax.fill_between(dates, drawdown, 0, color="#d62728", alpha=0.28)
    ax.plot(dates, drawdown, color="#b91c1c", linewidth=1.0)
    ax.set_title("回撤曲线", loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel("回撤")
    ax.grid(True, color="#d9dee7", linewidth=0.8, alpha=0.8)
    format_time_axis(ax)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_signal(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signals = signal_frame(result.signal_curve)
    setup_chinese_font()

    fig, ax = plt.subplots(figsize=(15, 5))
    if signals.is_empty() or "zscore" not in signals.columns:
        ax.text(0.5, 0.5, "没有可用信号数据", ha="center", va="center", transform=ax.transAxes)
    else:
        dates = signals["datetime"].to_list()
        ax.plot(dates, signals["zscore"].to_list(), color="#1f77b4", linewidth=1.0, label="z-score")
        entry_z = _first_numeric(signals, "entry_z", 2.0)
        exit_z = _first_numeric(signals, "exit_z", 0.5)
        for level, color, label in [(entry_z, "#dc2626", "开仓阈值"), (-entry_z, "#dc2626", None), (exit_z, "#16a34a", "平仓阈值"), (-exit_z, "#16a34a", None)]:
            ax.axhline(level, color=color, linestyle="--", linewidth=0.9, alpha=0.8, label=label)
        ax.legend(loc="upper left", frameon=False)

    ax.set_title("信号可视化", loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel("信号值")
    ax.grid(True, color="#d9dee7", linewidth=0.8, alpha=0.8)
    format_time_axis(ax)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_neutrality(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    positions = position_frame(result.position_curve)
    setup_chinese_font()

    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    if positions.is_empty():
        axes[0].text(0.5, 0.5, "没有持仓数据", ha="center", va="center", transform=axes[0].transAxes)
    else:
        dates = positions["datetime"].to_list()
        if "net_exposure_ratio" in positions.columns:
            axes[0].plot(dates, positions["net_exposure_ratio"].to_list(), color="#7c3aed", linewidth=1.0, label="净敞口比例")
        if "gross_exposure_ratio" in positions.columns:
            axes[1].plot(dates, positions["gross_exposure_ratio"].to_list(), color="#4b5563", linewidth=1.0, label="总敞口比例")
        for ax in axes:
            ax.axhline(0, color="#111827", linewidth=0.8)
            ax.legend(loc="upper left", frameon=False)

    axes[0].set_title("市场中性验证", loc="left", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("净敞口 / 权益")
    axes[1].set_ylabel("总敞口 / 权益")
    axes[1].set_xlabel("时间")
    for ax in axes:
        ax.grid(True, color="#d9dee7", linewidth=0.8, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    format_time_axis(axes[1])
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_attribution(return_attribution, risk_attribution, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    setup_chinese_font()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _bar_plot(axes[0], return_attribution, "收益归因")
    _bar_plot(axes[1], risk_attribution, "风险归因")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def equity_frame(equity_curve):
    frame = pl.DataFrame(normalize_rows(equity_curve), infer_schema_length=None)
    if frame.is_empty():
        raise ValueError("equity_curve is empty")
    frame = frame.select(["ts", "equity"]).with_columns(
        pl.from_epoch(pl.col("ts"), time_unit="ms").alias("datetime"),
        pl.col("equity").cast(pl.Float64, strict=False),
    )
    frame = frame.with_columns(pl.col("equity").cum_max().alias("peak"))
    frame = frame.with_columns((pl.col("equity") / pl.col("peak") - 1).alias("drawdown"))
    return frame


def position_frame(position_curve):
    frame = pl.DataFrame(normalize_rows(position_curve), infer_schema_length=None)
    if frame.is_empty():
        return frame
    frame = frame.with_columns(pl.col("ts").cast(pl.Int64, strict=False))
    frame = frame.drop_nulls("ts")
    if frame.is_empty():
        return frame
    frame = frame.with_columns(pl.from_epoch(pl.col("ts"), time_unit="ms").alias("datetime"))
    for column in frame.columns:
        if column.endswith("_position_ratio") or column in ["gross_exposure_ratio", "net_exposure_ratio"]:
            frame = frame.with_columns(pl.col(column).cast(pl.Float64, strict=False).fill_null(0.0))
    return frame


def signal_frame(signal_curve):
    frame = pl.DataFrame(normalize_rows(signal_curve), infer_schema_length=None)
    if frame.is_empty():
        return frame
    frame = frame.with_columns(pl.col("ts").cast(pl.Int64, strict=False))
    frame = frame.drop_nulls("ts")
    if frame.is_empty():
        return frame
    frame = frame.with_columns(pl.from_epoch(pl.col("ts"), time_unit="ms").alias("datetime"))
    for column in ["spread", "zscore", "hedge_ratio", "alpha", "beta", "long_vol", "short_vol", "entry_z", "exit_z"]:
        if column in frame.columns:
            frame = frame.with_columns(pl.col(column).cast(pl.Float64, strict=False))
    return frame


def plot_equity_and_drawdown(ax, equity):
    dates = equity["datetime"].to_list()
    values = equity["equity"].to_list()
    peaks = equity["peak"].to_list()
    ax.plot(dates, values, color="#1f77b4", linewidth=1.8, label="资金曲线")
    ax.fill_between(
        dates,
        peaks,
        values,
        where=[peak > value for peak, value in zip(peaks, values)],
        color="#d62728",
        alpha=0.18,
        linewidth=0,
        label="回撤区域",
    )
    ax.set_title("资金走势与回撤", loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel("账户权益")
    ax.grid(True, color="#d9dee7", linewidth=0.8, alpha=0.8)
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)


def plot_position_ratios(ax, positions):
    if positions.is_empty():
        ax.text(0.5, 0.5, "没有持仓数据", ha="center", va="center", transform=ax.transAxes)
        return
    dates = positions["datetime"].to_list()
    ratio_columns = [column for column in positions.columns if column.endswith("_position_ratio")]
    colors = ["#2ca02c", "#ff7f0e", "#9467bd", "#17becf"]
    for index, column in enumerate(ratio_columns):
        label = column.replace("_position_ratio", "")
        ax.plot(dates, positions[column].to_list(), linewidth=1.2, color=colors[index % len(colors)], label=label)
    if "gross_exposure_ratio" in positions.columns:
        ax.plot(dates, positions["gross_exposure_ratio"].to_list(), color="#4b5563", linewidth=1.0, alpha=0.8, linestyle="--", label="总敞口比例")
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_title("每个时点的持仓比例", loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel("持仓价值 / 账户权益")
    ax.set_xlabel("时间")
    ax.grid(True, color="#d9dee7", linewidth=0.8, alpha=0.8)
    ax.legend(loc="upper left", ncols=3, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)


def format_time_axis(ax):
    locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def setup_chinese_font():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _bar_plot(ax, rows, title):
    labels = [row["item"] for row in rows]
    values = [row["value"] for row in rows]
    colors = ["#1f77b4" if value >= 0 else "#d62728" for value in values]
    ax.bar(labels, values, color=colors, alpha=0.85)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", color="#d9dee7", linewidth=0.8, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def _first_numeric(frame, column, default):
    if column not in frame.columns:
        return default
    values = frame.select(pl.col(column).cast(pl.Float64, strict=False)).drop_nulls(column)
    if values.is_empty():
        return default
    return float(values[column][0])
