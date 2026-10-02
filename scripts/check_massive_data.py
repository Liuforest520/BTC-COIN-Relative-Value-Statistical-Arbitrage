"""Massive 美股分钟数据质量检查。

检查项：
  1) 当天缺口分钟：每个标的在当日交易时段内缺了多少分钟（正常应有 390 分钟）
  2) 整天缺失：某个交易日文件行数/标的数明显异常，或日期整段缺失
  3) 标的消失：昨天有数据、之后再也不出现的标的；以及中途出现空洞的标的

交易时段不写死时区：先取当日全体标的的时间跨度作为"实际时段"（美股 9:30-16:00，
对应 UTC 13:30-20:00 夏令时 / 14:30-21:00 冬令时），再以该时段为基准逐标的比对。

用法：
    python scripts/check_massive_data.py                     # 全量
    python scripts/check_massive_data.py --limit 2           # 只查前 2 天（试跑）
    python scripts/check_massive_data.py --start 2024-01-01 --end 2024-03-31
报告输出到 reports/massive_data_check/
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from collections import defaultdict
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "stocks" / "flat_files" / "minute_aggregates"
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports" / "massive_data_check"
EXPECTED_MINUTES = 390
LOGGER = logging.getLogger("check_massive_data")


def _fmt_dur(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    return f"{minutes}m{sec:02d}s" if minutes < 60 else f"{minutes // 60}h{minutes % 60:02d}m"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Massive 美股分钟数据质量检查")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--start", help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", help="结束日期 YYYY-MM-DD")
    parser.add_argument("--limit", type=int, help="只检查前 N 天")
    parser.add_argument("--top-volume", type=int, default=500,
                        help="只对当日成交额前 N 的标的做缺口检查（默认 500；0=全部）")
    parser.add_argument("--min-missing", type=int, default=1,
                        help="缺多少分钟以上才记入 missing 报告（默认 30；小盘股无成交属正常，"
                             "阈值过小会产生大量噪音）")
    parser.add_argument("--top", type=int, default=15, help="日志里展示的最差样例数量")
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.rglob("*.csv"))
    if args.start:
        files = [f for f in files if f.parent.name >= args.start]
    if args.end:
        files = [f for f in files if f.parent.name <= args.end]
    if args.limit:
        files = files[: args.limit]
    if not files:
        LOGGER.error("没有找到可检查的 .csv（目录 %s）", input_dir.resolve())
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("=" * 96)
    LOGGER.info("Massive 美股分钟数据质量检查")
    LOGGER.info("  数据目录 : %s", input_dir.resolve())
    LOGGER.info("  检查天数 : %d 天（%s → %s）", len(files), files[0].parent.name, files[-1].parent.name)
    LOGGER.info("  报告目录 : %s", out_dir.resolve())
    LOGGER.info("=" * 96)

    daily_rows: list[dict] = []
    missing_rows: list[dict] = []
    coverage: dict[str, dict] = {}
    day_tickers: dict[str, set[str]] = {}
    started = time.perf_counter()
    total_missing_minutes = 0
    worst: list[tuple[int, str, str]] = []

    for index, path in enumerate(files, start=1):
        day = path.parent.name
        df = pl.read_csv(path, columns=["ticker", "window_start", "volume"],
                         schema_overrides={"window_start": pl.Int64, "volume": pl.Float64})
        if df.height == 0:
            daily_rows.append({"date": day, "rows": 0, "tickers": 0, "session_start_utc": "",
                               "session_end_utc": "", "session_minutes": 0,
                               "tickers_with_gaps": 0, "missing_minutes": 0, "empty_rows": 0})
            LOGGER.warning("[%d/%d] %s 文件为空！", index, len(files), day)
            continue

        minutes_all = (df["window_start"] // 60_000_000_000)
        # 常规交易时段 = 成交量最集中的 390 分钟连续窗口（盘前盘后量远小于盘中）
        hist = df.with_columns(minutes_all.alias("minute")).group_by("minute").agg(
            pl.col("volume").sum().alias("vol")
        ).sort("minute")
        mins = hist["minute"].to_list()
        vols = hist["vol"].to_list()
        best_start, best_vol = mins[0], -1.0
        left = running = 0
        for right in range(len(mins)):
            running += vols[right]
            while mins[right] - mins[left] + 1 > EXPECTED_MINUTES:
                running -= vols[left]
                left += 1
            if running > best_vol:
                best_vol, best_start = running, mins[left]
        lo, hi = best_start, best_start + EXPECTED_MINUTES - 1
        session_minutes = EXPECTED_MINUTES
        session = df.with_columns(minutes_all.alias("minute"))
        session = session.filter((pl.col("minute") >= lo) & (pl.col("minute") <= hi))
        per_ticker = session.group_by("ticker").agg(
            pl.col("minute").n_unique().alias("present"),
            pl.col("volume").sum().alias("volume"),
        )
        # 只对当日成交额最大的 N 个标的做"缺口分钟"检查：小盘股本来就不会每分钟成交，
        # 把它们算进来会产生大量噪音（实测 1 万个标的里 9 千多个都会"有缺口"）。
        if args.top_volume and per_ticker.height > args.top_volume:
            checked = per_ticker.sort("volume", descending=True).head(args.top_volume)
        else:
            checked = per_ticker
        gaps = checked.filter(pl.col("present") < EXPECTED_MINUTES)
        session_total = per_ticker["present"].sum()
        day_missing = int(session_minutes * per_ticker.height - session_total)
        total_missing_minutes += day_missing
        day_tickers[day] = set(per_ticker["ticker"].to_list())

        for r in gaps.sort("present").head(args.top).iter_rows(named=True):
            miss = session_minutes - int(r["present"])
            if miss >= args.min_missing:
                worst.append((miss, day, r["ticker"]))
        if gaps.height:
            for r in gaps.filter((session_minutes - pl.col("present")) >= args.min_missing).iter_rows(named=True):
                missing_rows.append({"date": day, "ticker": r["ticker"],
                                     "expected_minutes": session_minutes,
                                     "present_minutes": int(r["present"]),
                                     "missing_minutes": session_minutes - int(r["present"])})

        for r in per_ticker.iter_rows(named=True):
            info = coverage.setdefault(r["ticker"], {"first": day, "last": day, "days": 0, "gaps": 0})
            info["last"] = day
            info["days"] += 1
            if session_minutes - int(r["present"]) >= args.min_missing:
                info["gaps"] += 1

        daily_rows.append({
            "date": day, "rows": df.height, "tickers": per_ticker.height,
            "session_start_utc": str(lo * 60), "session_end_utc": str(hi * 60),
            "session_minutes": session_minutes,
            "tickers_with_gaps": gaps.height, "missing_minutes": day_missing,
            "empty_rows": int(df.filter(pl.col("volume") <= 0).height),
        })
        if index % 25 == 0 or index == len(files):
            LOGGER.info("进度 %d/%d | 已用 %s | 累计缺口 %d 分钟",
                        index, len(files), _fmt_dur(time.perf_counter() - started), total_missing_minutes)

    # --- 报告 1：逐日汇总 ---
    daily_path = out_dir / "daily_summary.csv"
    with daily_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(daily_rows[0].keys()))
        w.writeheader()
        w.writerows(daily_rows)

    # --- 报告 2：缺口分钟明细 ---
    missing_path = out_dir / "missing_minutes.csv"
    with missing_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["date", "ticker", "expected_minutes", "present_minutes", "missing_minutes"])
        w.writeheader()
        w.writerows(missing_rows)

    # --- 报告 3：标的覆盖与消失 ---
    ordered_days = [r["date"] for r in daily_rows]
    last_day = ordered_days[-1]
    cov_rows = []
    vanished = []
    for ticker, info in sorted(coverage.items()):
        expected_days = ordered_days.index(info["last"]) - ordered_days.index(info["first"]) + 1
        cov_rows.append({"ticker": ticker, "first_day": info["first"], "last_day": info["last"],
                         "days_present": info["days"], "days_in_range": expected_days,
                         "absent_inside_range": expected_days - info["days"], "days_with_gaps": info["gaps"]})
        if info["last"] != last_day:
            vanished.append({"ticker": ticker, "last_day": info["last"],
                             "missing_days_since": len(ordered_days) - ordered_days.index(info["last"]) - 1,
                             "first_day": info["first"], "days_present": info["days"]})
    cov_path = out_dir / "ticker_coverage.csv"
    with cov_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["ticker", "first_day", "last_day", "days_present",
                                           "days_in_range", "absent_inside_range", "days_with_gaps"])
        w.writeheader()
        w.writerows(cov_rows)
    van_path = out_dir / "vanished_tickers.csv"
    vanished.sort(key=lambda r: -int(r["missing_days_since"]))
    with van_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["ticker", "last_day", "missing_days_since", "first_day", "days_present"])
        w.writeheader()
        w.writerows(vanished)

    # --- 日志汇总 ---
    empty_days = [r["date"] for r in daily_rows if r["rows"] == 0]
    thin_days = sorted(daily_rows, key=lambda r: r["tickers"])[:5]
    LOGGER.info("-" * 96)
    LOGGER.info("检查完成，用时 %s", _fmt_dur(time.perf_counter() - started))
    LOGGER.info("  交易日文件        : %d", len(daily_rows))
    LOGGER.info("  空文件            : %d %s", len(empty_days), empty_days[:5])
    LOGGER.info("  标的总数          : %d", len(coverage))
    LOGGER.info("  缺口分钟总计      : %d（%d 条明细）", total_missing_minutes, len(missing_rows))
    LOGGER.info("  有缺口的标的日次  : %d", len({(r['date'], r['ticker']) for r in missing_rows}))
    LOGGER.info("  消失的标的        : %d（最后一天 %s）", len(vanished), last_day)
    LOGGER.info("  标的数最少的 5 天 : %s",
                [(r["date"], r["tickers"]) for r in thin_days])
    LOGGER.info("  缺口最严重的 10 条：")
    for miss, day, ticker in sorted(worst, reverse=True)[:10]:
        LOGGER.info("     %s %-8s 缺 %d 分钟", day, ticker, miss)
    LOGGER.info("  报告：%s / %s / %s / %s",
                daily_path.name, missing_path.name, cov_path.name, van_path.name)
    LOGGER.info("-" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
