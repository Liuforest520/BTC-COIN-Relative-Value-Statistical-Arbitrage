"""Plot the real price history preceding the selected log-price entries."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "documents" / "02_价差与收益分析" / "X价格波动收益分析"
ENTRY_FILE = ANALYSIS_DIR / "开仓示例明细.csv"
RUN_DIR = ROOT / "results" / "sweep_batch" / "top_backtests" / "top_03_000044_tls_3d_update"


def _last_update_times(entries: pd.DataFrame) -> dict[str, int]:
    states = pd.read_csv(
        RUN_DIR / "pair_curve.csv",
        usecols=["ts", "pair_id", "beta"],
        dtype={"pair_id": "string", "beta": "string"},
    )
    states["beta"] = pd.to_numeric(states["beta"], errors="coerce")
    updates = {}
    for row in entries.itertuples(index=False):
        pair = states[(states["pair_id"] == row.pair_id) & (states["ts"] <= row.entry_ts)].dropna(subset=["beta"])
        if pair.empty:
            raise RuntimeError(f"missing beta history for {row.pair_id}")
        same_beta = np.isclose(pair["beta"].to_numpy(), row.beta, rtol=1e-10, atol=1e-12)
        matching = pair.loc[same_beta]
        if matching.empty:
            raise RuntimeError(f"entry beta not found in state history for {row.pair_id}")
        updates[row.pair_id] = int(matching["ts"].iloc[0])
    return updates


def _load_prices(symbol: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    path = ROOT / "data" / symbol / f"{symbol}-1m.csv"
    data = (
        pl.scan_csv(path)
        .select(
            pl.col("open_time").cast(pl.Int64).alias("ts"),
            pl.col("close").cast(pl.Float64).alias("close"),
        )
        .filter(pl.col("ts").is_between(start_ts, end_ts, closed="both"))
        .collect()
        .to_pandas()
    )
    if data.empty:
        raise RuntimeError(f"no prices for {symbol} in {start_ts}..{end_ts}")
    data["time_bj"] = pd.to_datetime(data["ts"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
    return data.set_index("ts")


def _pair_history(row, update_ts: int):
    fit_start_ts = update_ts - (int(row.model_lookback_bars) - 1) * 60_000
    x = _load_prices(row.x_symbol, fit_start_ts, int(row.entry_ts)).rename(columns={"close": "x_close"})
    y = _load_prices(row.y_symbol, fit_start_ts, int(row.entry_ts)).rename(columns={"close": "y_close"})
    data = x[["time_bj", "x_close"]].join(y[["y_close"]], how="inner")
    data = data[(data["x_close"] > 0) & (data["y_close"] > 0)].copy()
    data["fit_sample"] = data.index <= update_ts
    return fit_start_ts, data


def _hourly(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.set_index("time_bj")[["x_close", "y_close"]].resample("1h").last().dropna()
    frame["x_log_price"] = np.log(frame["x_close"])
    frame["y_log_price"] = np.log(frame["y_close"])
    return frame


def _configure_time_axis(ax):
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz="Asia/Shanghai"))


def main():
    entries = pd.read_csv(ENTRY_FILE)
    updates = _last_update_times(entries)
    histories = {}
    for row in entries.itertuples(index=False):
        fit_start_ts, data = _pair_history(row, updates[row.pair_id])
        histories[row.pair_id] = (fit_start_ts, data)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    summary_rows = []
    generated_files = []
    for row in entries.itertuples(index=False):
        fit_start_ts, data = histories[row.pair_id]
        hourly = _hourly(data)
        update_time = pd.to_datetime(updates[row.pair_id], unit="ms", utc=True).tz_convert("Asia/Shanghai")
        entry_time = pd.to_datetime(row.entry_ts, unit="ms", utc=True).tz_convert("Asia/Shanghai")

        fig, (ax_raw, ax_log) = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True)
        ax_raw_y = ax_raw.twinx()
        raw_x = ax_raw.plot(
            hourly.index, hourly["x_close"], color="#1f77b4", lw=1.7,
            label=f"X: {row.x_symbol}（左轴）",
        )
        raw_y = ax_raw_y.plot(
            hourly.index, hourly["y_close"], color="#ff7f0e", lw=1.7,
            label=f"Y: {row.y_symbol}（右轴）",
        )
        update_line = ax_raw.axvline(
            update_time, color="#9467bd", lw=1.2, ls="--", label="Beta估计时点"
        )
        entry_line = ax_raw.axvline(
            entry_time, color="#2ca02c", lw=1.2, ls=":", label="实际开仓"
        )
        ax_raw.set_ylabel(f"{row.x_symbol} 原始价格")
        ax_raw_y.set_ylabel(f"{row.y_symbol} 原始价格")
        ax_raw.set_title("原始价格")
        ax_raw.grid(alpha=0.22)
        raw_handles = raw_x + raw_y + [update_line, entry_line]
        ax_raw.legend(raw_handles, [line.get_label() for line in raw_handles], loc="best", ncol=2)

        ax_log.plot(
            hourly.index, hourly["x_log_price"], color="#1f77b4", lw=1.7,
            label=f"log({row.x_symbol})",
        )
        ax_log.plot(
            hourly.index, hourly["y_log_price"], color="#ff7f0e", lw=1.7,
            label=f"log({row.y_symbol})",
        )
        ax_log.axvline(update_time, color="#9467bd", lw=1.2, ls="--")
        ax_log.axvline(entry_time, color="#2ca02c", lw=1.2, ls=":")
        ax_log.set_ylabel("Log price（自然对数）")
        ax_log.set_xlabel("北京时间")
        ax_log.set_title("Log price")
        ax_log.grid(alpha=0.22)
        ax_log.legend(loc="best", ncol=2)
        _configure_time_axis(ax_log)

        fig.suptitle(
            f"{row.pair_id} | 开仓Beta={row.beta:.4f} | 开仓Z={row.entry_zscore:.2f} | "
            "30天模型窗口至实际开仓",
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        output_path = ANALYSIS_DIR / f"{row.pair_id}_开仓前原始价格与LogPrice.png"
        fig.savefig(output_path, dpi=170)
        plt.close(fig)
        generated_files.append(output_path.name)

        fit = data[data["fit_sample"]]
        x_move = (fit["x_close"].iloc[-1] / fit["x_close"].iloc[0] - 1.0) * 100.0
        y_move = (fit["y_close"].iloc[-1] / fit["y_close"].iloc[0] - 1.0) * 100.0
        summary_rows.append(
            {
                "pair_id": row.pair_id,
                "x_symbol": row.x_symbol,
                "y_symbol": row.y_symbol,
                "entry_beta": row.beta,
                "fit_start_time_bj": pd.to_datetime(fit_start_ts, unit="ms", utc=True).tz_convert("Asia/Shanghai").isoformat(),
                "beta_update_time_bj": update_time.isoformat(),
                "entry_time_bj": entry_time.isoformat(),
                "fit_window_x_change_pct": x_move,
                "fit_window_y_change_pct": y_move,
                "minutes_from_beta_update_to_entry": (int(row.entry_ts) - updates[row.pair_id]) / 60_000.0,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(ANALYSIS_DIR / "开仓前价格窗口摘要.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"pair_id": entries["pair_id"], "chart_file": generated_files}).to_csv(
        ANALYSIS_DIR / "开仓前价格图表索引.csv", index=False, encoding="utf-8-sig"
    )

    print(f"wrote {len(generated_files)} pair charts to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
