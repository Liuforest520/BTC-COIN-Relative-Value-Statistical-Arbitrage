"""Counterfactual: trade only the pairs that pass a top-10 holder concentration filter.

No re-backtest: the per-pair contribution curves are read from the finished run's
review data (``trade_review_data/pairs/<pair_id>/<YYYY-MM>.json``, field ``pnl``,
already net of fees, slippage and funding) and stitched into portfolio curves.

One chart per selection is written, plus a reference chart for every pair with
data.  Selection rule: a pair passes when every leg *with* concentration data in
``documents/df_top10_bn.csv`` has ``top10_filter <= threshold``; legs without data
are exempt (TradFi-mapped contracts have no holder distribution at all).

LOOK-AHEAD WARNING: the filter is static (concentration is measured today), so the
result is not tradable as-is; it answers "how would this deployment have looked".

Usage:
    python research/scripts/top10_filter_counterfactual.py \
        --run-dir results/backtests/tls_log_price_full97_L2D_U4H_E3p0_X0p5_freeze
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for candidate in (PROJECT_ROOT, Path(__file__).resolve().parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import pair_selection_counterfactual as cf  # noqa: E402

DEFAULT_CSV = PROJECT_ROOT / "documents" / "df_top10_bn.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "research" / "outputs" / "top10_filter_counterfactual"

CJK_FONTS = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC")


def load_concentration(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        reader.fieldnames = [name.strip() for name in reader.fieldnames or []]
        for row in reader:
            symbol = str(row.get("symbol") or "").strip()
            raw = str(row.get("top10_filter") or "").strip()
            if symbol and raw:
                try:
                    values[symbol] = float(raw)
                except ValueError:
                    pass
    return values


def setup_font(plt) -> bool:
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in CJK_FONTS:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    plt.rcParams["axes.unicode_minus"] = False
    return False


def plot_one(path: Path, axis: list[int], equity: list[float], initial: float,
             title: str, stats: dict, cjk: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    peak = equity[0]
    drawdown = []
    for value in equity:
        peak = max(peak, value)
        drawdown.append(value / peak - 1.0 if peak > 0 else 0.0)

    figure, (ax, ax_dd) = plt.subplots(
        2, 1, figsize=(15, 8.5), dpi=150, sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.1, "top": 0.80, "bottom": 0.07},
    )
    ax.plot(axis, equity, color="#0f766e", linewidth=1.3)
    ax.axhline(initial, color="#94a3b8", linestyle="--", linewidth=0.9)
    trough = min(range(len(drawdown)), key=lambda index: drawdown[index])
    ax.scatter([axis[trough]], [equity[trough]], color="#dc2626", s=30, zorder=6)
    ax.annotate(f"{drawdown[trough]:.2%}", xy=(axis[trough], equity[trough]),
                xytext=(8, -18), textcoords="offset points", color="#dc2626", fontsize=9)
    ax.set_ylabel("Equity / 资金曲线" if cjk else "Equity")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.margins(x=0.01)

    ax_dd.fill_between(axis, [value * 100 for value in drawdown], 0, color="#f87171", alpha=0.45)
    ax_dd.set_ylabel("Drawdown % / 回撤%" if cjk else "Drawdown %")
    ax_dd.grid(alpha=0.25, linewidth=0.6)
    ax_dd.margins(x=0.01)

    figure.suptitle(title, fontsize=14, x=0.06, ha="left", y=0.965)
    lines = [
        "币对 pairs        {}  (有数据 {} / 全部 {})".format(
            stats["pairs"], stats["universe"], stats["configured"]) if cjk else
        "pairs             {}  (with data {} / configured {})".format(
            stats["pairs"], stats["universe"], stats["configured"]),
        "期末权益 final    {:,.0f}    总收益 return {:.2%}".format(
            stats["final_equity"], stats["total_return"]) if cjk else
        "final equity      {:,.0f}    total return {:.2%}".format(
            stats["final_equity"], stats["total_return"]),
        "CAGR {:.2%}   Sharpe {:.2f}   Calmar {:.2f}   最大回撤 maxDD {:.2%}".format(
            stats["cagr"], stats["sharpe"], stats["calmar"], stats["max_drawdown"]),
        "区间 window       {}  ..  {}".format(stats["start"], stats["end"]),
    ]
    figure.text(0.06, 0.90, "\n".join(lines), ha="left", va="top", fontsize=10,
                family="monospace" if not cjk else "sans-serif",
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8fafc", edgecolor="#cbd5e1"))
    figure.savefig(path)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    parser.add_argument(
        "--sets",
        choices=("both", "kept", "excluded"),
        default="both",
        help="which baskets to export: the pairs that pass the filter, the ones "
             "the filter rejects, or both (default)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    concentration = load_concentration(args.csv)
    config = cf.load_config(run_dir)
    tags = cf.pair_tags(config)
    official = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    axis = cf.equity_axis(run_dir)
    initial = float(official.get("initial_equity") or 100_000.0)

    traded = set()
    with (run_dir / "trades.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("pair_id"):
                traded.add(row["pair_id"])

    no_data = {pair_id for pair_id, pair_tags in tags.items()
               if "tradfi" in pair_tags and pair_id not in traded}
    universe = [pair_id for pair_id in tags if pair_id not in no_data]

    curves: dict[str, list[float]] = {}
    legs: dict[str, tuple[str, str]] = {}
    for pair_id in universe:
        points = cf.load_pair_curve(run_dir, pair_id)
        if not points:
            continue
        values = [0.0] * len(axis)
        last = 0.0
        cursor = 0
        index_of = {ts: index for index, ts in enumerate(axis)} if False else None
        curves[pair_id] = values
        legs[pair_id] = _pair_symbols(config, pair_id)

        # fill the aligned array from the (ts, pnl) points
        cursor = 0
        for ts, pnl in points:
            position = cf_bisect(axis, ts)
            if position is None:
                continue
            for index in range(cursor, position):
                values[index] = last
            values[position] = pnl
            cursor = position + 1
            last = pnl
        for index in range(cursor, len(axis)):
            values[index] = last

    print(f"run            : {run_dir.name}")
    print(f"configured     : {len(tags)}  (no data / TradFi excluded: {len(no_data)})"
          f"   universe with curves: {len(curves)}")
    print(f"concentration  : {args.csv.name}  symbols={len(concentration)}")

    passthrough: dict[str, list[str]] = {}
    for threshold in args.thresholds:
        selected, excluded = [], []
        for pair_id, (x_symbol, y_symbol) in legs.items():
            leg_values = [concentration.get(x_symbol), concentration.get(y_symbol)]
            worst = max((value for value in leg_values if value is not None), default=None)
            if worst is None or worst <= threshold:
                selected.append(pair_id)
            else:
                excluded.append(pair_id)
        passthrough[f"le{threshold:.1f}".replace(".", "p")] = selected
        passthrough[f"le{threshold:.1f}_excluded".replace(".", "p")] = excluded

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels: dict[str, list[str]] = {}
    if args.sets in ("both", "kept"):
        if args.sets == "both":
            labels["all_pairs"] = sorted(curves)
        for key, pair_ids in passthrough.items():
            if not key.endswith("_excluded"):
                labels[key] = pair_ids
    if args.sets in ("both", "excluded"):
        for key, pair_ids in passthrough.items():
            if key.endswith("_excluded"):
                labels[key] = pair_ids

    # drop artifacts from the previous run that this selection no longer produces
    for stale in list(args.output_dir.glob("equity_*.png")) + list(args.output_dir.glob("pairs_*.txt")):
        stem = stale.stem
        if stem.startswith("equity_"):
            name = stem[len("equity_"):]
        else:
            name = stem[len("pairs_"):]
        if name not in labels:
            stale.unlink()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cjk = setup_font(plt)

    rows = []
    for label, pair_ids in labels.items():
        pnl = [sum(curves[pair_id][index] for pair_id in pair_ids) for index in range(len(axis))]
        equity = [initial + value for value in pnl]
        stats = cf.metrics(axis, equity)
        stats.update({
            "label": label, "pairs": len(pair_ids), "universe": len(curves),
            "configured": len(tags),
            "start": _date(axis[0]), "end": _date(axis[-1]),
        })
        rows.append(stats)
        if label.startswith("all_pairs"):
            title = "All pairs with data / 全部有数据的币对"
        elif label.endswith("_excluded"):
            title = (f"Excluded by top10_filter <= {label[2:5].replace('p', '.')} "
                     f"/ 被该阈值剔除的币对")
        else:
            title = f"top10_filter <= {label[2:].replace('p', '.')} / 集中度筛选后的币对"
        plot_one(args.output_dir / f"equity_{label}.png", axis, equity, initial,
                 f"{title}  ({len(pair_ids)} pairs)", stats, cjk)
        print(f"{label:<22} pairs={len(pair_ids):>3}  ret={stats['total_return']:>8.2%}"
              f"  cagr={stats['cagr']:>8.2%}  sharpe={stats['sharpe']:>6.2f}"
              f"  maxDD={stats['max_drawdown']:>8.2%}  calmar={stats['calmar']:>6.2f}")

    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "pairs", "universe", "final_equity", "total_return",
                         "cagr", "sharpe", "max_drawdown", "calmar"])
        for row in rows:
            writer.writerow([
                row["label"], row["pairs"], row["universe"], f"{row['final_equity']:.2f}",
                f"{row['total_return']:.6f}", f"{row['cagr']:.6f}", f"{row['sharpe']:.6f}",
                f"{row['max_drawdown']:.6f}", f"{row['calmar']:.6f}",
            ])

    for label, pair_ids in labels.items():
        if label.startswith("all_pairs"):
            continue
        (args.output_dir / f"pairs_{label}.txt").write_text("\n".join(sorted(pair_ids)) + "\n",
                                                            encoding="utf-8")

    _write_readme(args.output_dir / "README.md", run_dir, official, rows, len(tags),
                  len(no_data), len(curves), args.thresholds, concentration, legs)
    print(f"\noutputs -> {args.output_dir}")
    return 0


def _pair_symbols(config: dict, pair_id: str) -> tuple[str, str]:
    setup = config["setups"][config["active_setup"]]
    for pair in setup["pairs"]:
        if pair["pair_id"] == pair_id:
            return pair["x_symbol"], pair["y_symbol"]
    return "", ""


def cf_bisect(axis: list[int], ts: int):
    from bisect import bisect_left

    index = bisect_left(axis, ts)
    if index >= len(axis):
        return None
    return index if axis[index] == ts else None


def _date(ts: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _write_readme(path: Path, run_dir: Path, official: dict, rows: list[dict],
                  configured: int, no_data: int, universe: int, thresholds,
                  concentration: dict[str, float], legs: dict[str, tuple[str, str]]) -> None:
    lines = [
        "# top10 持仓集中度筛选的反事实回测（按单对曲线捏合）",
        "",
        f"- 来源 run：`{run_dir.name}`（**没有重新回测**）",
        "- 数据来源：该 run 的 `trade_review_data/pairs/<pair_id>/<YYYY-MM>.json` 里的 `pnl` 字段",
        "  （= 累计成交现金流 + qx·Px + qy·Py − 累计资金费，已含手续费/滑点/资金费）",
        f"- 集中度表：`documents/df_top10_bn.csv`（{len(concentration)} 个 symbol 有值）",
        f"- 配置里的币对：{configured}；其中 {no_data} 个 TradFi 币对在窗口内没有 K 线，已排除；"
        f"参与捏合的币对：{universe}",
        f"- 初始资金：{official.get('initial_equity'):,.2f}（沿用该 run）",
        "",
        "> 筛选规则：一条腿有数据且 `top10_filter > 阈值` → 整个币对剔除；"
        "没有数据的腿豁免（TradFi 合约没有链上持币分布）。",
        "> **注意**：集中度是今天的静态快照，属于事后信息，不是可实盘复现的策略。",
        "",
        "| 方案 | 币对数 | 总收益 | CAGR | Sharpe | 最大回撤 | Calmar | 图 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        chart = f"`equity_{row['label']}.png`" if "_excluded" not in row["label"] else "-"
        lines.append("| {} | {} | {:.2%} | {:.2%} | {:.2f} | {:.2%} | {:.2f} | {} |".format(
            row["label"], row["pairs"], row["total_return"], row["cagr"], row["sharpe"],
            row["max_drawdown"], row["calmar"], chart,
        ))
    lines += [
        "",
        f"对照：该 run 官方 `metrics.json` = 总收益 {official['total_return']:.2%}、"
        f"Sharpe {official['sharpe']:.3f}、最大回撤 {official['max_drawdown']:.2%}"
        "（`all_pairs` 行应与之一致）。",
        "",
        "## 已知偏差",
        "",
        "1. 只把已发生的成交按子集相加，未重算组合层：实际少做币对会释放并发名额、改变权益路径与复利。",
        "2. 单对仓位仍是开仓当时权益的 10%（该历史 run 的旧版固定槽位资金逻辑），未按筛选后的币对数重新分配。",
        "3. 期末未平仓浮盈亏已按窗口末价计入单对曲线。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
