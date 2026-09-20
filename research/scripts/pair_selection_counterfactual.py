"""Counterfactual: what if only the best-performing pairs had been traded?

Reads the per-pair contribution curves of one finished run from its review data
(``trade_review_data/pairs/<pair_id>/<YYYY-MM>.json``, field ``pnl``) and
stitches subsets of them into portfolio curves:

* every pair with data (the reference, must reproduce metrics.json)
* the best 75% and best 50% of those pairs, ranked by realised PnL
* (extra) worst 50%, and a walk-forward variant that ranks on the first half
  and holds through the second half, i.e. no look-ahead

Each pair curve is already net of fees, slippage and funding:
``pnl = cumulative fill cashflow + qx*Px + qy*Py - cumulative funding``.

LOOK-AHEAD WARNING: the "best 75% / 50%" numbers pick pairs with the whole
backtest's outcome known.  They are an upper bound, not a tradable strategy.
Use the walk-forward line to judge whether pair selection actually helps.

Usage:
    python research/scripts/pair_selection_counterfactual.py \
        --run-dir results/backtests/tls_log_price_full97_L2D_U4H_E3p0_X0p5_freeze
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

MILLISECONDS_PER_YEAR = 365 * 24 * 60 * 60 * 1000
MILLISECONDS_PER_DAY = 24 * 60 * 60 * 1000


def load_config(run_dir: Path) -> dict:
    return yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))


def pair_tags(config: dict) -> dict[str, list[str]]:
    setup = config["setups"][config["active_setup"]]
    return {p["pair_id"]: list(p.get("tags") or []) for p in setup["pairs"]}


def equity_axis(run_dir: Path) -> list[int]:
    axis: list[int] = []
    with (run_dir / "equity_curve.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = row.get("ts")
            if raw:
                axis.append(int(float(raw)))
    return axis


def load_pair_curve(run_dir: Path, pair_id: str) -> list[tuple[int, float]]:
    """Return [(ts, pnl), ...] for one pair, in time order."""
    folder = run_dir / "trade_review_data" / "pairs" / pair_id
    if not folder.is_dir():
        return []
    points: list[tuple[int, float]] = []
    for path in sorted(folder.glob("*.json")):
        if path.name == "summary.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        offsets = payload.get("t") or []
        values = payload.get("pnl")
        if values is None or not offsets:
            continue
        ts = int(payload["t0"])
        for index, delta in enumerate(offsets):
            ts += int(delta)
            value = values[index] if index < len(values) else None
            points.append((ts, float(value) if value is not None else 0.0))
    points.sort(key=lambda item: item[0])
    return points


def metrics(ts: list[int], equity: list[float]) -> dict:
    if len(equity) < 2:
        return {}
    initial, final = equity[0], equity[-1]
    spacing = sorted(
        ts[index] - ts[index - 1] for index in range(1, len(ts)) if ts[index] > ts[index - 1]
    )
    median_spacing = spacing[len(spacing) // 2] if spacing else 60_000
    factor = MILLISECONDS_PER_YEAR / median_spacing if median_spacing else 0.0

    returns = [
        equity[index] / equity[index - 1] - 1.0
        for index in range(1, len(equity))
        if equity[index - 1] > 0
    ]
    sharpe = None
    if len(returns) > 2 and factor:
        mean = sum(returns) / len(returns)
        variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
        std = math.sqrt(variance)
        if std > 0:
            sharpe = mean * factor / (std * math.sqrt(factor))

    peak = equity[0]
    max_drawdown = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            max_drawdown = min(max_drawdown, value / peak - 1.0)

    elapsed_days = (ts[-1] - ts[0]) / MILLISECONDS_PER_DAY
    cagr = None
    if elapsed_days > 0 and initial > 0 and final > 0:
        cagr = (final / initial) ** (365.0 / elapsed_days) - 1.0
    calmar = cagr / abs(max_drawdown) if cagr is not None and max_drawdown < 0 else None
    return {
        "initial_equity": initial,
        "final_equity": final,
        "total_return": final / initial - 1.0 if initial else None,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "days": elapsed_days,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "research" / "outputs" / "pair_selection_counterfactual",
    )
    parser.add_argument("--split-fraction", type=float, default=0.5,
                        help="walk-forward split: rank on this fraction of the window")
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    config = load_config(run_dir)
    tags = pair_tags(config)
    official = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    axis = equity_axis(run_dir)
    if not axis:
        raise SystemExit(f"no equity curve in {run_dir}")

    # pairs that actually traded, and pairs without data (no fills at all)
    traded: set[str] = set()
    with (run_dir / "trades.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("pair_id"):
                traded.add(row["pair_id"])

    # universe = pairs whose symbols have data in the window.  The 44 TradFi
    # pairs have no bars at all, so they never trade; keep them out.
    no_data = {
        pair_id for pair_id, pair_tags_list in tags.items()
        if "tradfi" in pair_tags_list and pair_id not in traded
    }
    universe = [pair_id for pair_id in tags if pair_id not in no_data]

    index_of = {ts: index for index, ts in enumerate(axis)}
    curves: dict[str, list[float]] = {}
    for pair_id in universe:
        points = load_pair_curve(run_dir, pair_id)
        if not points:
            continue
        values = [0.0] * len(axis)
        last = 0.0
        cursor = 0
        for ts, pnl in points:
            position = index_of.get(ts)
            if position is None:
                continue
            for index in range(cursor, position):
                values[index] = last
            values[position] = pnl
            cursor = position + 1
            last = pnl
        for index in range(cursor, len(axis)):
            values[index] = last
        curves[pair_id] = values

    print(f"run          : {run_dir.name}")
    print(f"axis points  : {len(axis)}")
    print(f"pairs in cfg : {len(tags)}   no-data (TradFi) excluded: {len(no_data)}"
          f"   universe: {len(universe)}   curves loaded: {len(curves)}")
    print(f"pairs with fills: {len(traded & set(universe))}")

    initial = float(official.get("initial_equity") or 100_000.0)
    ranking = sorted(curves, key=lambda pair_id: -curves[pair_id][-1])
    count = len(ranking)
    top75 = ranking[: max(1, round(count * 0.75))]
    top50 = ranking[: max(1, round(count * 0.50))]
    bottom50 = ranking[-(count - len(top50)):]

    split_index = int(len(axis) * args.split_fraction)
    by_first_half = sorted(curves, key=lambda pair_id: -curves[pair_id][max(0, split_index - 1)])
    walk_forward = by_first_half[: max(1, round(count * 0.50))]

    selections = {
        "all_pairs": ranking,
        "top75_lookahead": top75,
        "top50_lookahead": top50,
        "bottom50_lookahead": bottom50,
        f"top50_walkforward_rank{args.split_fraction:.0%}": walk_forward,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    curves_out: dict[str, list[float]] = {}
    for label, pair_ids in selections.items():
        pnl = [sum(curves[pair_id][index] for pair_id in pair_ids) for index in range(len(axis))]
        equity = [initial + value for value in pnl]
        curves_out[label] = equity
        stats = metrics(axis, equity)
        stats.update({"label": label, "pairs": len(pair_ids)})
        summary_rows.append(stats)
        print(f"{label:<34} pairs={len(pair_ids):>3}  ret={stats['total_return']:>8.2%}"
              f"  cagr={stats['cagr']:>8.2%}  sharpe={stats['sharpe']:>6.2f}"
              f"  maxDD={stats['max_drawdown']:>8.2%}  calmar={stats['calmar']:>6.2f}")

    print(f"\nofficial metrics.json: ret={official['total_return']:.2%}"
          f"  sharpe={official['sharpe']:.3f}  maxDD={official['max_drawdown']:.2%}")

    # ---- write artifacts ----
    with (args.output_dir / "pair_ranking.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "pair_id", "tags", "final_pnl", "pnl_first_half",
                         "pnl_second_half", "in_top75", "in_top50", "in_walkforward"])
        for position, pair_id in enumerate(ranking, start=1):
            values = curves[pair_id]
            writer.writerow([
                position, pair_id, "|".join(tags.get(pair_id) or []),
                f"{values[-1]:.6f}", f"{values[max(0, split_index - 1)]:.6f}",
                f"{values[-1] - values[max(0, split_index - 1)]:.6f}",
                int(pair_id in top75), int(pair_id in top50), int(pair_id in walk_forward),
            ])

    with (args.output_dir / "portfolio_curves.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        labels = list(curves_out)
        writer.writerow(["ts", *[f"equity_{label}" for label in labels],
                         *[f"pnl_{label}" for label in labels]])
        for index, ts in enumerate(axis):
            writer.writerow([ts, *[f"{curves_out[label][index]:.6f}" for label in labels],
                             *[f"{curves_out[label][index] - initial:.6f}" for label in labels]])

    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "pairs", "initial_equity", "final_equity", "total_return",
                         "cagr", "sharpe", "max_drawdown", "calmar", "days"])
        for row in summary_rows:
            writer.writerow([
                row["label"], row["pairs"], f"{row['initial_equity']:.2f}",
                f"{row['final_equity']:.2f}", f"{row['total_return']:.6f}",
                "" if row["cagr"] is None else f"{row['cagr']:.6f}",
                "" if row["sharpe"] is None else f"{row['sharpe']:.6f}",
                f"{row['max_drawdown']:.6f}",
                "" if row["calmar"] is None else f"{row['calmar']:.6f}",
                f"{row['days']:.2f}",
            ])

    for label, pair_ids in (("top75", top75), ("top50", top50), ("walkforward", walk_forward)):
        (args.output_dir / f"selected_{label}.txt").write_text(
            "\n".join(pair_ids) + "\n", encoding="utf-8"
        )

    _write_markdown(args.output_dir / "README.md", run_dir, official, summary_rows,
                    len(universe), len(traded & set(universe)), len(no_data), ranking,
                    curves, split_index, args.split_fraction)
    _plot(args.output_dir / "equity_curves.png", axis, curves_out, initial)

    print(f"\noutputs -> {args.output_dir}")
    return 0


def _write_markdown(path: Path, run_dir: Path, official: dict, rows: list[dict],
                    universe: int, traded: int, no_data: int, ranking: list[str],
                    curves: dict[str, list[float]], split_index: int,
                    split_fraction: float) -> None:
    lines = [
        "# Pair 选择的反事实回测（用单对曲线拼合）",
        "",
        f"- 来源 run：`{run_dir.name}`",
        f"- 数据来源：该 run 的 `trade_review_data/pairs/<pair_id>/<YYYY-MM>.json`（字段 `pnl`，已含手续费/滑点/资金费）",
        f"- 组合初始资金：{official.get('initial_equity'):,.2f}（沿用该 run）",
        f"- 有数据的币对：{universe}（另有 {no_data} 个 TradFi 币对在窗口内没有 K 线，已排除；其中有成交的 {traded} 个）",
        "",
        "> **重要：前 75% / 前 50% 是用整段回测的最终盈亏排名挑出来的，属于事后选择（look-ahead），不是可实现的策略，只能当上限看。**",
        "> `top50_walkforward` 行才是没有未来信息的版本：用前 "
        f"{split_fraction:.0%} 的窗口排名，持有到结束。",
        "",
        "| 方案 | 币对数 | 总收益 | CAGR | Sharpe | 最大回撤 | Calmar |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| {} | {} | {:.2%} | {} | {} | {:.2%} | {} |".format(
            row["label"], row["pairs"], row["total_return"],
            "-" if row["cagr"] is None else f"{row['cagr']:.2%}",
            "-" if row["sharpe"] is None else f"{row['sharpe']:.2f}",
            row["max_drawdown"],
            "-" if row["calmar"] is None else f"{row['calmar']:.2f}",
        ))
    lines += [
        "",
        f"该 run 官方 `metrics.json` 作为对照：总收益 {official['total_return']:.2%}、"
        f"Sharpe {official['sharpe']:.3f}、最大回撤 {official['max_drawdown']:.2%}"
        "（`all_pairs` 行应与它基本一致，差一点是因为只按分钟曲线求指标、且未重算复利路径）。",
        "",
        "## 排名前 15 的币对",
        "",
        "| 名次 | pair | 期末盈亏 | 前半段 | 后半段 |",
        "|---:|---|---:|---:|---:|",
    ]
    for position, pair_id in enumerate(ranking[:15], start=1):
        values = curves[pair_id]
        lines.append("| {} | {} | {:,.0f} | {:,.0f} | {:,.0f} |".format(
            position, pair_id, values[-1], values[max(0, split_index - 1)],
            values[-1] - values[max(0, split_index - 1)]))
    lines += [
        "",
        "## 已知偏差",
        "",
        "1. 只把已发生的成交按子集相加，没有重算组合层：实际少做币对会释放并发名额、改变权益路径与复利，因此结果偏乐观/偏差方向不确定。",
        "2. 单对仓位是开仓当时权益的 10%（该历史 run 的旧版固定槽位资金逻辑），这里沿用原成交，不做再缩放。",
        "3. 期末未平仓的浮盈亏已包含在单对曲线里（按窗口末价 mark）。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot(path: Path, axis: list[int], curves: dict[str, list[float]], initial: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["axes.unicode_minus"] = False
    figure, (ax, ax_dd) = plt.subplots(
        2, 1, figsize=(15, 9), dpi=150, sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    colors = ["#334155", "#2563eb", "#16a34a", "#f97316", "#dc2626"]
    for (label, equity), color in zip(curves.items(), colors):
        ax.plot(axis, equity, linewidth=1.2, label=label, color=color)
        peak = equity[0]
        drawdown = []
        for value in equity:
            peak = max(peak, value)
            drawdown.append(value / peak - 1.0 if peak > 0 else 0.0)
        ax_dd.plot(axis, [value * 100 for value in drawdown], linewidth=0.9,
                   color=color, alpha=0.8)
    ax.axhline(initial, color="#94a3b8", linestyle="--", linewidth=0.9)
    ax.set_ylabel("equity")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="upper left", fontsize=9)
    ax_dd.set_ylabel("drawdown %")
    ax_dd.grid(alpha=0.25, linewidth=0.6)
    ax.set_title("Pair selection counterfactual (top 75% / 50% are look-ahead)", fontsize=12)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
