"""Generate X-price sensitivity plots from the five log-price top backtests.

The examples use real entry fills and model state from the backtest.  For each
entry, X is varied by -50%..+50%; Y is the model-implied price at the configured
exit residual.  The resulting pair PnL is normalized by entry gross notional.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
TOP_ROOT = ROOT / "results" / "sweep_batch" / "top_backtests"
OUT_ROOT = ROOT / "documents" / "02_价差与收益分析" / "X价格波动收益分析"


def _num(value):
    try:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return np.nan
        text = str(value).strip()
        if not text or text.lower() in {"none", "nan", "null"}:
            return np.nan
        return float(text)
    except (TypeError, ValueError):
        return np.nan


def _parse_config(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    setup = config["setups"][config["active_setup"]]
    estimator = setup["pipeline"]["estimator"]
    signal = setup["pipeline"]["signal"]
    pairs = {
        item["pair_id"]: (item["x_symbol"], item["y_symbol"])
        for item in setup["pairs"]
    }
    return config, estimator, signal, pairs


def _read_entries(run_dir: Path, pair_symbols: dict[str, tuple[str, str]]):
    fills = pd.read_csv(
        run_dir / "trades.csv",
        usecols=[
            "group_id", "symbol", "action", "side", "quantity", "price",
            "notional", "fee", "slippage", "ts", "position_id", "pair_id",
            "target_hedge_ratio",
        ],
    )
    fills["ts"] = pd.to_numeric(fills["ts"], errors="coerce")
    fills["quantity"] = pd.to_numeric(fills["quantity"], errors="coerce")
    fills["price"] = pd.to_numeric(fills["price"], errors="coerce")
    fills["fee"] = pd.to_numeric(fills["fee"], errors="coerce").fillna(0.0)
    fills["slippage"] = pd.to_numeric(fills["slippage"], errors="coerce").fillna(0.0)

    opens = fills[fills["action"].eq("open")].copy()
    entries = []
    for position_id, group in opens.groupby("position_id", sort=False):
        if len(group) < 2:
            continue
        pair_id = str(group["pair_id"].iloc[0])
        if pair_id not in pair_symbols:
            continue
        x_symbol, y_symbol = pair_symbols[pair_id]
        x_row = group[group["symbol"].eq(x_symbol)]
        y_row = group[group["symbol"].eq(y_symbol)]
        if x_row.empty or y_row.empty:
            continue
        x_row = x_row.iloc[0]
        y_row = y_row.iloc[0]
        x_sign = 1.0 if str(x_row["side"]).lower() == "buy" else -1.0
        y_sign = 1.0 if str(y_row["side"]).lower() == "buy" else -1.0
        entries.append(
            {
                "position_id": position_id,
                "pair_id": pair_id,
                "entry_ts": int(x_row["ts"]),
                "x_symbol": x_symbol,
                "y_symbol": y_symbol,
                "entry_x_price": float(x_row["price"]),
                "entry_y_price": float(y_row["price"]),
                "qty_x": x_sign * float(x_row["quantity"]),
                "qty_y": y_sign * float(y_row["quantity"]),
                "entry_x_notional": float(x_row["notional"]),
                "entry_y_notional": float(y_row["notional"]),
                "entry_fee": float(x_row["fee"] + y_row["fee"]),
                "entry_slippage": float(x_row["slippage"] + y_row["slippage"]),
                "target_hedge_ratio": float(x_row["target_hedge_ratio"]),
                "x_side": str(x_row["side"]),
                "y_side": str(y_row["side"]),
                "fills": fills,
            }
        )
    return entries


def _attach_model_state(entries, run_dir: Path):
    state = pd.read_csv(
        run_dir / "pair_curve.csv",
        usecols=["ts", "pair_id", "alpha", "beta", "hedge_beta", "spread_mean", "spread_std"],
        dtype={"ts": "int64", "pair_id": "string"},
    )
    for col in ["alpha", "beta", "hedge_beta", "spread_mean", "spread_std"]:
        state[col] = pd.to_numeric(state[col], errors="coerce")
    lookup = state.set_index(["ts", "pair_id"])
    result = []
    for entry in entries:
        key = (entry["entry_ts"], entry["pair_id"])
        try:
            row = lookup.loc[key]
        except KeyError:
            continue
        values = {col: _num(row[col]) for col in ["alpha", "beta", "hedge_beta", "spread_mean", "spread_std"]}
        if not all(np.isfinite(values[col]) for col in ["alpha", "beta", "spread_mean", "spread_std"]):
            continue
        spread = math.log(entry["entry_y_price"]) - values["alpha"] - values["beta"] * math.log(entry["entry_x_price"])
        zscore = (spread - values["spread_mean"]) / values["spread_std"] if values["spread_std"] > 0 else np.nan
        result.append({**entry, **values, "entry_spread": spread, "entry_zscore": zscore})
    return result


def _attach_actual_exit(entry):
    fills = entry.pop("fills")
    closes = fills[(fills["position_id"] == entry["position_id"]) & fills["action"].eq("close")]
    x_close = closes[closes["symbol"].eq(entry["x_symbol"])]
    y_close = closes[closes["symbol"].eq(entry["y_symbol"])]
    if x_close.empty or y_close.empty:
        return entry
    x_close = x_close.iloc[0]
    y_close = y_close.iloc[0]
    exit_x = float(x_close["price"])
    exit_y = float(y_close["price"])
    pnl = entry["qty_x"] * (exit_x - entry["entry_x_price"]) + entry["qty_y"] * (exit_y - entry["entry_y_price"])
    gross = abs(entry["qty_x"] * entry["entry_x_price"]) + abs(entry["qty_y"] * entry["entry_y_price"])
    costs = float(x_close["fee"] + y_close["fee"] + x_close["slippage"] + y_close["slippage"])
    entry.update(
        {
            "exit_ts": int(x_close["ts"]),
            "exit_x_price": exit_x,
            "exit_y_price": exit_y,
            "actual_x_move_pct": (exit_x / entry["entry_x_price"] - 1.0) * 100.0,
            "actual_gross_pnl": pnl,
            "actual_net_pnl": pnl - entry["entry_fee"] - entry["entry_slippage"] - costs,
            "actual_gross_return_pct": pnl / gross * 100.0 if gross else np.nan,
            "actual_net_return_pct": (pnl - entry["entry_fee"] - entry["entry_slippage"] - costs) / gross * 100.0 if gross else np.nan,
        }
    )
    return entry


def _scenario(entry, entry_z: float, exit_z: float, fee_rate: float, slippage_bps: float):
    sign = 1.0 if entry["qty_x"] > 0 else -1.0
    target_spread = entry["spread_mean"] + sign * exit_z * entry["spread_std"]
    gross = abs(entry["qty_x"] * entry["entry_x_price"]) + abs(entry["qty_y"] * entry["entry_y_price"])
    rows = []
    for move in np.linspace(-0.50, 0.50, 101):
        x_price = entry["entry_x_price"] * (1.0 + move)
        y_price = math.exp(entry["alpha"] + entry["beta"] * math.log(x_price) + target_spread)
        pnl = entry["qty_x"] * (x_price - entry["entry_x_price"]) + entry["qty_y"] * (y_price - entry["entry_y_price"])
        # Approximate four fills (two legs in and two legs out) at the configured rates.
        turnover = gross + abs(entry["qty_x"] * x_price) + abs(entry["qty_y"] * y_price)
        costs = turnover * (fee_rate + slippage_bps / 10000.0)
        rows.append(
            {
                "position_id": entry["position_id"],
                "pair_id": entry["pair_id"],
                "entry_zscore": entry["entry_zscore"],
                "x_move_pct": move * 100.0,
                "scenario_x_price": x_price,
                "scenario_y_target_price": y_price,
                "gross_pnl": pnl,
                "gross_return_pct": pnl / gross * 100.0 if gross else np.nan,
                "estimated_cost": costs,
                "net_pnl": pnl - costs,
                "net_return_pct": (pnl - costs) / gross * 100.0 if gross else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    top_dirs = sorted(TOP_ROOT.glob("top_*"))[:5]
    if len(top_dirs) < 5:
        raise RuntimeError(f"expected five top backtests, found {len(top_dirs)}")

    target_entry_z = 2.5
    selected_run = None
    for rank, run_dir in enumerate(top_dirs, start=1):
        config, estimator, signal, pairs = _parse_config(run_dir / "config.yaml")
        if estimator.get("regression_method") == "log_price" and math.isclose(
            float(signal.get("entry_z", 0.0)), target_entry_z
        ):
            selected_run = (rank, run_dir, config, estimator, signal, pairs)
            break
    if selected_run is None:
        raise RuntimeError("no top-five log-price backtest with entry_z=2.5")

    rank, run_dir, config, estimator, signal, pairs = selected_run
    entries = _attach_model_state(_read_entries(run_dir, pairs), run_dir)
    entries = [_attach_actual_exit(entry) for entry in entries]
    entries = [
        entry for entry in entries
        if np.isfinite(entry.get("entry_zscore", np.nan))
        and np.isfinite(entry.get("actual_x_move_pct", np.nan))
        and abs(entry["actual_x_move_pct"]) <= 50.0
    ]
    if not entries:
        raise RuntimeError("no closed entry/model-state examples found")

    # Select one threshold-nearest event from each of the five most frequently
    # traded pairs.  This avoids choosing examples based on their profitability.
    pair_counts = pd.Series([entry["pair_id"] for entry in entries]).value_counts()
    selected = []
    for pair_id in pair_counts.head(5).index:
        candidates = [entry for entry in entries if entry["pair_id"] == pair_id]
        candidates.sort(key=lambda e: (abs(abs(e["entry_zscore"]) - target_entry_z), e["entry_ts"]))
        selected.append(candidates[0])

    all_scenarios = []
    for entry in selected:
        entry.update(
            {
                "rank": rank,
                "config_id": run_dir.name.split("_")[2],
                "experiment_name": run_dir.name,
                "estimator_method": estimator.get("method"),
                "regression_method": estimator.get("regression_method"),
                "model_lookback_bars": estimator.get("model_lookback_bars"),
                "model_update_interval_bars": estimator.get("model_update_interval_bars"),
                "entry_z_threshold": target_entry_z,
                "exit_z_threshold": float(signal.get("exit_z", 0.5)),
                "fee_rate": float(config.get("cost", {}).get("fee_rate", 0.0)),
                "slippage_bps": float(config.get("cost", {}).get("slippage_bps", 0.0)),
            }
        )
        scenario = _scenario(entry, target_entry_z, entry["exit_z_threshold"], entry["fee_rate"], entry["slippage_bps"])
        scenario["rank"] = rank
        scenario["model"] = entry["estimator_method"]
        scenario["regression_method"] = entry["regression_method"]
        all_scenarios.append(scenario)

    for entry in selected:
        entry["entry_time_bj"] = pd.to_datetime(entry["entry_ts"], unit="ms", utc=True).tz_convert("Asia/Shanghai").isoformat()
        if np.isfinite(entry.get("exit_ts", np.nan)):
            entry["exit_time_bj"] = pd.to_datetime(entry["exit_ts"], unit="ms", utc=True).tz_convert("Asia/Shanghai").isoformat()

    entry_columns = [
        "rank", "config_id", "experiment_name", "estimator_method", "regression_method", "pair_id",
        "x_symbol", "y_symbol", "position_id", "entry_ts", "entry_time_bj", "entry_z_threshold", "exit_z_threshold",
        "entry_zscore", "entry_spread", "alpha", "beta", "hedge_beta", "spread_mean", "spread_std",
        "entry_x_price", "entry_y_price", "qty_x", "qty_y", "entry_x_notional", "entry_y_notional",
        "target_hedge_ratio", "entry_fee", "entry_slippage", "exit_ts", "exit_time_bj", "exit_x_price", "exit_y_price",
        "actual_x_move_pct", "actual_gross_return_pct", "actual_net_return_pct", "model_lookback_bars",
        "model_update_interval_bars", "fee_rate", "slippage_bps",
    ]
    entries_df = pd.DataFrame(selected)
    for col in entry_columns:
        if col not in entries_df:
            entries_df[col] = np.nan
    entries_df[entry_columns].to_csv(OUT_ROOT / "开仓示例明细.csv", index=False, encoding="utf-8-sig")
    scenarios_df = pd.concat(all_scenarios, ignore_index=True)
    scenarios_df.to_csv(OUT_ROOT / "X价格敏感性曲线数据.csv", index=False, encoding="utf-8-sig")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(len(selected), 1, figsize=(11, 3.1 * len(selected)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, entry in zip(axes, selected):
        curve = scenarios_df[scenarios_df["position_id"].eq(entry["position_id"])]
        ax.plot(curve["x_move_pct"], curve["gross_return_pct"], color="#1f77b4", lw=1.8, label="毛收益率")
        ax.plot(curve["x_move_pct"], curve["net_return_pct"], color="#d62728", lw=1.5, label="扣成本后收益率")
        if "actual_x_move_pct" in entry and np.isfinite(entry.get("actual_x_move_pct", np.nan)):
            actual_y = entry.get("actual_net_return_pct", np.nan)
            if np.isfinite(actual_y):
                ax.scatter([entry["actual_x_move_pct"]], [actual_y], color="#2ca02c", s=45, zorder=5, label="真实平仓结果")
        ax.axvline(0.0, color="#777777", lw=0.8)
        ax.axhline(0.0, color="#777777", lw=0.8)
        ax.set_title(
            f"{entry['estimator_method'].upper()} | {entry['pair_id']} | 开仓Z={entry['entry_zscore']:.2f} | "
            f"开仓Beta={entry['beta']:.4f} | "
            f"X={entry['entry_x_price']:.6g}, Y={entry['entry_y_price']:.6g}"
        )
        ax.set_ylabel("收益率 (%)")
        ax.grid(alpha=0.22)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("X价格相对开仓价的变化 (%)")
    axes[-1].set_xlim(-50, 50)
    fig.suptitle("Log-price 模型：开仓实例的 X 价格敏感性收益曲线", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUT_ROOT / "X价格波动与收益率.png", dpi=160)
    plt.close(fig)

    lines = [
        "# X价格波动与收益率分析",
        "",
        "本分析从 `results/sweep_batch/top_backtests` 的 log-price 前五名结果中，选择开仓阈值为 2.5、平仓阈值为 0.5 的回测配置。再从交易次数最多的五个 Pair 中各取一笔最接近开仓阈值的真实开仓事件；例子不是按收益高低挑选。每笔事件记录开仓成交价格、数量、Alpha、Beta、残差均值、残差标准差和实际开仓 Z-score。",
        "",
        "对每个开仓事件，将 X 价格相对开仓价从 -50% 扫描到 +50%。每个 X 价格对应的 Y 价格按开仓时模型关系和 `exit_z` 对应的目标残差计算。收益率按开仓时两腿总名义金额归一化；净收益率额外估算双腿开平仓手续费和滑点。绿色点（若存在）为该笔交易在真实回测中的实际平仓结果。",
        "",
        "## 文件",
        "",
        "- `X价格波动与收益率.png`：收益率敏感性曲线。",
        "- `开仓示例明细.csv`：每个例子的开仓指标、价格、仓位及实际平仓结果。",
        "- `X价格敏感性曲线数据.csv`：绘图使用的逐点数据。",
        "",
        "## 开仓示例",
        "",
        "| Pair | 开仓时间（北京时间） | 两腿方向 | 开仓Z | Alpha | Beta | 残差均值 | 残差标准差 | X开仓价 | Y开仓价 | 实际净收益率 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in selected:
        lines.append(
            f"| {entry['pair_id']} | {entry['entry_time_bj']} | "
            f"X {entry['x_side']} / Y {entry['y_side']} | {entry['entry_zscore']:.4f} | "
            f"{entry['alpha']:.6f} | {entry['beta']:.6f} | {entry['spread_mean']:.6f} | "
            f"{entry['spread_std']:.6f} | {entry['entry_x_price']:.8g} | "
            f"{entry['entry_y_price']:.8g} | {entry['actual_net_return_pct']:.4f}% |"
        )
    lines.extend(
        [
            "",
            "## 曲线口径",
            "",
            "每条曲线都以真实开仓状态为起点，但曲线上的其他点是条件情景：假设 X 变动到横轴位置，同时残差恰好回到平仓阈值，模型由此确定 Y 的目标价格。绿色点才是历史回测中实际发生的平仓结果。",
        ]
    )
    (OUT_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(selected)} examples to {OUT_ROOT}")


if __name__ == "__main__":
    main()
