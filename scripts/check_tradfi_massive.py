"""检查抽取出来的 tradfi 分钟数据质量（data/stocks/tradfi/<SYMBOL>/<SYMBOL>-1m-massive.csv）。

检查项：
  1) 常规交易时段内缺分钟：每天应为 390 分钟（美东 09:30-15:59），逐标的比对，
     并输出具体缺失的分钟区间
  2) 整天缺失：该标的在某个交易日完全没有常规时段数据
  3) 重复分钟、时段外数据（盘前盘后，仅统计不报错）

时区处理：
  美东夏令时(EDT)=UTC-4，冬令时(EST)=UTC-5，两者切换会让交易时段的 UTC 窗口差 1 小时。
  优先用 zoneinfo 的 America/New_York；若系统缺 tzdata，则退回内置的美国 DST 规则
  （2007 年起：3 月第二个周日 02:00 开始，11 月第一个周日 02:00 结束）。

用法：
    python scripts/check_tradfi_massive.py                 # 全部标的
    python scripts/check_tradfi_massive.py --limit 3       # 只查前 3 个标的
    python scripts/check_tradfi_massive.py --symbols NVDAUSDT,TSLAUSDT
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import time
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "stocks" / "tradfi"
OUT_DIR = PROJECT_ROOT / "reports" / "tradfi_massive_check"
LOGGER = logging.getLogger("check_tradfi")
SESSION_OPEN = (9, 30)
SESSION_MINUTES = 390          # 常规交易日：09:30-15:59
EARLY_CLOSE_MINUTES = 210      # 半日市：09:30-12:59（13:00 提前收盘）
NS_PER_MIN = 60_000_000_000


def early_close_days(year: int) -> set[dt.date]:
    """美股半日市（13:00 ET 提前收盘）：
      - 独立日前一个交易日（7/4 若为周六则提前到 7/3 休市，提前收盘再往前一天）
      - 感恩节次日（黑色星期五）
      - 圣诞前夜（12/24，若逢周末则取之前的周五）
    """
    def observed(day: dt.date) -> dt.date:
        if day.weekday() == 5:      # 周六 -> 前一天休市
            return day - dt.timedelta(days=1)
        if day.weekday() == 6:      # 周日 -> 后一天休市
            return day + dt.timedelta(days=1)
        return day

    def prev_trading_day(day: dt.date) -> dt.date:
        cur = day - dt.timedelta(days=1)
        while cur.weekday() >= 5:
            cur -= dt.timedelta(days=1)
        return cur

    def nth_weekday(y: int, month: int, weekday: int, n: int) -> dt.date:
        first = dt.date(y, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + dt.timedelta(days=offset + 7 * (n - 1))

    independence = prev_trading_day(observed(dt.date(year, 7, 4)))
    thanksgiving = nth_weekday(year, 11, 3, 4)              # 11 月第 4 个周四
    black_friday = thanksgiving + dt.timedelta(days=1)
    christmas_eve = dt.date(year, 12, 24)
    if christmas_eve.weekday() >= 5:
        christmas_eve = prev_trading_day(observed(dt.date(year, 12, 25)))
    return {independence, black_friday, christmas_eve}


def _tz():
    try:
        return ZoneInfo("America/New_York")
    except Exception:  # noqa: BLE001 - Windows 常常没有 tzdata
        return None


TZ = _tz()


def _us_dst_active(day: dt.date) -> bool:
    """2007 年起的美国规则：3 月第二个周日 ~ 11 月第一个周日为夏令时。"""
    def nth_sunday(year: int, month: int, n: int) -> dt.date:
        first = dt.date(year, month, 1)
        offset = (6 - first.weekday()) % 7
        return first + dt.timedelta(days=offset + 7 * (n - 1))

    start = nth_sunday(day.year, 3, 2)
    end = nth_sunday(day.year, 11, 1)
    return start <= day < end


def session_bounds_utc(day: dt.date, minutes: int = SESSION_MINUTES) -> tuple[int, int]:
    """返回该交易日常规时段在 UTC 下的 [起, 止] 分钟时间戳（含端点）。"""
    if TZ is not None:
        local_open = dt.datetime(day.year, day.month, day.day, *SESSION_OPEN, tzinfo=TZ)
        offset_hours = local_open.utcoffset().total_seconds() / 3600
    else:
        offset_hours = -4 if _us_dst_active(day) else -5
    start_utc = dt.datetime(day.year, day.month, day.day, *SESSION_OPEN,
                            tzinfo=dt.timezone.utc) - dt.timedelta(hours=offset_hours)
    start_min = int(start_utc.timestamp() // 60)
    return start_min, start_min + minutes - 1


def _fmt_dur(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m < 60 else f"{m // 60}h{m % 60:02d}m"


def _fmt_et(minute_ts: int) -> str:
    epoch = minute_ts * 60
    if TZ is not None:
        return dt.datetime.fromtimestamp(epoch, TZ).strftime("%Y-%m-%d %H:%M")
    day = dt.datetime.fromtimestamp(epoch, dt.timezone.utc).date()
    off = -4 if _us_dst_active(day) else -5
    return dt.datetime.fromtimestamp(epoch + off * 3600, dt.timezone.utc).strftime("%Y-%m-%d %H:%M")


def _ranges(minutes: list[int]) -> list[str]:
    """把缺失的分钟列表压缩成可读区间（按 UTC 时间）。"""
    out, i = [], 0
    while i < len(minutes):
        j = i
        while j + 1 < len(minutes) and minutes[j + 1] == minutes[j] + 1:
            j += 1
        fmt = lambda m: dt.datetime.fromtimestamp(m * 60, dt.timezone.utc).strftime("%H:%M")
        out.append(fmt(minutes[i]) if i == j else f"{fmt(minutes[i])}-{fmt(minutes[j])}")
        i = j + 1
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="检查 tradfi Massive 分钟数据质量")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--symbols", help="逗号分隔的标的文件夹名，默认全部")
    p.add_argument("--limit", type=int, help="只查前 N 个标的")
    args = p.parse_args(argv)

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*/*-1m-massive.csv"))
    if args.symbols:
        wanted = {s.strip() for s in args.symbols.split(",") if s.strip()}
        files = [f for f in files if f.parent.name in wanted]
    if args.limit:
        files = files[: args.limit]
    if not files:
        LOGGER.error("没有找到 *-1m-massive.csv（先跑 extract_tradfi_from_massive.py）")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("=" * 96)
    LOGGER.info("tradfi 分钟数据质量检查（常规时段 %d 分钟/天）", SESSION_MINUTES)
    LOGGER.info("  时区来源 : %s", "zoneinfo America/New_York" if TZ else "内置美国 DST 规则（无 tzdata）")
    LOGGER.info("  检查标的 : %d 个", len(files))
    for probe in (dt.date(2024, 1, 2), dt.date(2024, 7, 1)):
        lo, _ = session_bounds_utc(probe)
        LOGGER.info("  时段校验 : %s -> %s UTC（%s）", probe,
                    dt.datetime.fromtimestamp(lo * 60, dt.timezone.utc).strftime("%H:%M"),
                    "冬令时 EST" if not _us_dst_active(probe) else "夏令时 EDT")
    LOGGER.info("=" * 96)

    started = time.perf_counter()
    daily_rows, gap_rows, day_rows, symbols_summary = [], [], [], []
    all_days: set[str] = set()
    symbol_days: dict[str, set[str]] = defaultdict(set)
    per_symbol_missing: dict[str, list[dict]] = defaultdict(list)
    EARLY_CLOSE = set().union(*(early_close_days(y) for y in range(2024, 2027)))
    LOGGER.info("  半日市日历（按 %d 分钟校验）: %s", EARLY_CLOSE_MINUTES,
                ", ".join(sorted(d.isoformat() for d in EARLY_CLOSE)))

    for index, path in enumerate(files, start=1):
        symbol = path.parent.name
        df = pl.read_csv(path, has_header=False, columns=[0, 6],
                         new_columns=["ticker", "window_start"],
                         schema_overrides={"ticker": pl.Utf8, "window_start": pl.Utf8})
        df = df.filter(pl.col("ticker") != "ticker")          # 跳过可能存在的表头行
        df = df.with_columns(pl.col("window_start").cast(pl.Int64, strict=False))
        df = df.drop_nulls("window_start")
        if df.height == 0:
            LOGGER.warning("[%d/%d] %s 文件为空或格式异常", index, len(files), symbol)
            continue
        minutes = sorted(set((df["window_start"] // NS_PER_MIN).to_list()))
        by_day: dict[str, list[int]] = defaultdict(list)
        for m in minutes:
            by_day[dt.datetime.fromtimestamp(m * 60, dt.timezone.utc).strftime("%Y-%m-%d")].append(m)

        total_missing = in_session_days = dup = outside = 0
        for day_str, mins in sorted(by_day.items()):
            day = dt.date.fromisoformat(day_str)
            early = day in EARLY_CLOSE
            expected = EARLY_CLOSE_MINUTES if early else SESSION_MINUTES
            lo, hi = session_bounds_utc(day, expected)
            inside = [m for m in mins if lo <= m <= hi]
            present = len(set(inside))
            dup += len(inside) - present
            outside += len(mins) - len(inside)
            all_days.add(day_str)
            if present:
                symbol_days[symbol].add(day_str)
            if present == 0:
                day_rows.append({"symbol": symbol, "date": day_str,
                                 "kind": "常规时段无数据（文件里有该日但时段内为空）",
                                 "expected_minutes": expected, "early_close": early,
                                 "market_symbols_with_data": -1, "verdict": "待交叉判定"})
                continue
            in_session_days += 1
            missing = expected - present
            if missing:
                total_missing += missing
                absent = [m for m in range(lo, hi + 1) if m not in set(inside)]
                for m in absent:
                    per_symbol_missing[symbol].append({
                        "symbol": symbol, "date": day_str, "kind": "缺失分钟",
                        "time_utc": dt.datetime.fromtimestamp(m * 60, dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
                        "time_et": _fmt_et(m),
                        "time_utc_plus8": dt.datetime.fromtimestamp(m * 60 + 8 * 3600, dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
                        "note": "半日市(13:00收盘)" if early else "",
                    })
                gap_rows.append({"symbol": symbol, "date": day_str, "present_minutes": present,
                                 "expected_minutes": expected, "missing_minutes": missing,
                                 "early_close": early,
                                 "missing_ranges_utc": ";".join(_ranges(absent)),
                                 "coverage_pct": round(present / expected * 100, 2)})
                daily_rows.append({"symbol": symbol, "date": day_str, "present": present,
                                   "missing": missing, "early_close": early,
                                   "out_of_session": len(mins) - len(inside)})
        symbols_summary.append({
            "symbol": symbol, "first_day": min(by_day) if by_day else "",
            "last_day": max(by_day) if by_day else "", "days_with_data": len(by_day),
            "days_in_session": in_session_days, "missing_minutes_total": total_missing,
            "duplicate_minutes": dup, "out_of_session_rows": outside,
        })
        LOGGER.info("[%d/%d] %-14s 天数 %4d 覆盖不足的天 %4d 缺分钟 %6d 时段外(盘前盘后) %6d 重复 %d",
                    index, len(files), symbol, len(by_day), len(by_day) - in_session_days,
                    total_missing, outside, dup)

    # 整天缺失（以全体标的出现过的日期为基准）
    window = sorted(all_days)
    if window:
        w_lo, w_hi = window[0], window[-1]
        for row in symbols_summary:
            have = {r["date"] for r in day_rows if r["symbol"] == row["symbol"]}
            have |= set()
            present_days = set()
            for r in gap_rows:
                pass
            # 重新计算该标的实际有数据的日期集合
            present_days = {r["date"] for r in daily_rows if r["symbol"] == row["symbol"]}
            present_days |= {r["date"] for r in day_rows if r["symbol"] == row["symbol"] and False}
            row["missing_whole_days"] = None

    # 整天缺失（含"文件里完全没有该日"）+ 全市场交叉判定
    market_days = sorted(all_days)
    presence: dict[str, set[str]] = defaultdict(set)
    for r in gap_rows:
        presence[r["date"]].add(r["symbol"])
    for row in symbols_summary:
        have = symbol_days[row["symbol"]]
        presence.setdefault
        for d in sorted(have):
            presence[d].add(row["symbol"])
    # 有数据的日期（不论是否覆盖全）
    for d in sorted(all_days):
        presence.setdefault(d, set())
    full_day_rows = []
    for row in symbols_summary:
        sym = row["symbol"]
        have = symbol_days[sym]
        if not have:
            continue
        lo, hi = min(have), max(have)
        expected_days = [d for d in market_days if lo <= d <= hi]
        for d in expected_days:
            if d in have:
                continue
            early = dt.date.fromisoformat(d) in EARLY_CLOSE
            others = len(presence.get(d, set()) - {sym})
            total = len(symbols_summary) - 1
            verdict = ("数据源问题（当天多数标的都缺）" if others < total * 0.5
                       else "标的自身无交易（同一天其他标的正常）")
            full_day_rows.append({"symbol": sym, "date": d, "kind": "文件里完全没有该日",
                                  "expected_minutes": EARLY_CLOSE_MINUTES if early else SESSION_MINUTES,
                                  "early_close": early, "market_symbols_with_data": others,
                                  "verdict": verdict})
    day_rows.extend(full_day_rows)

    # 交易日判定 + 交叉判定：
    #   Massive 的日文件在周末/假期也会带少量行（隔夜/盘后），若不剔除会产生大量假警报。
    #   只有"多数标的当天有常规时段数据"的日期才算交易日；其余整行标记为非交易日并忽略。
    n_sym = len(symbols_summary)
    trading_days = {d for d, syms in presence.items() if len(syms) >= max(3, 0.2 * n_sym)}
    cleaned_rows, non_trading = [], 0
    for r in day_rows:
        d = r["date"]
        on_market = len(presence.get(d, set()))
        if d not in trading_days:
            r["kind"] = "非交易日（周末/假期），忽略"
            r["market_symbols_with_data"] = on_market
            r["verdict"] = "非交易日"
            non_trading += 1
            continue
        others = len(presence.get(d, set()) - {r["symbol"]})
        r["market_symbols_with_data"] = others
        r["verdict"] = ("数据源问题（当天多数标的都缺）" if others < (n_sym - 1) * 0.5
                        else "标的自身无交易（同日其他标的正常）")
        cleaned_rows.append(r)
    day_rows = cleaned_rows
    LOGGER.info("  交易日 %d 天（非交易日整行已剔除 %d 条）", len(trading_days), non_trading)
    LOGGER.info("  真缺失：标的自身无交易 %d 条 / 疑似数据源问题 %d 条",
                sum(1 for r in day_rows if r["verdict"].startswith("标的自身")),
                sum(1 for r in day_rows if r["verdict"].startswith("数据源")))

    for r in day_rows:
        per_symbol_missing[r["symbol"]].append({
            "symbol": r["symbol"], "date": r["date"], "kind": "整天缺失",
            "time_utc": r["date"], "time_et": r["date"], "time_utc_plus8": r["date"],
            "note": f"{r['verdict']}；当天有数据的其他标的数={r['market_symbols_with_data']}",
        })
    written_files = 0
    for symbol, rows in sorted(per_symbol_missing.items()):
        if not rows:
            continue
        target = data_dir / symbol / f"{symbol}-1m-massive-missing.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        rows.sort(key=lambda r: (r["time_utc"], r["kind"]))
        with target.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["symbol", "date", "kind", "time_utc",
                                               "time_et", "time_utc_plus8", "note"])
            w.writeheader()
            w.writerows(rows)
        written_files += 1
        LOGGER.info("  缺失清单 %-14s %6d 行 -> %s", symbol, len(rows), target.resolve())
    LOGGER.info("  已写出 %d 个标的的缺失清单（data/stocks/tradfi/<SYMBOL>/<SYMBOL>-1m-massive-missing.csv）", written_files)

    gap_path = out_dir / "missing_minutes.csv"
    with gap_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["symbol", "date", "present_minutes", "expected_minutes",
                                           "missing_minutes", "early_close",
                                           "missing_ranges_utc", "coverage_pct"])
        w.writeheader()
        w.writerows(gap_rows)
    day_path = out_dir / "missing_days.csv"
    with day_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["symbol", "date", "kind", "expected_minutes",
                                           "early_close", "market_symbols_with_data", "verdict"])
        w.writeheader()
        w.writerows(day_rows)
    sym_path = out_dir / "symbol_summary.csv"
    with sym_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(symbols_summary[0].keys()))
        w.writeheader()
        w.writerows(symbols_summary)

    LOGGER.info("-" * 96)
    LOGGER.info("检查完成，用时 %s", _fmt_dur(time.perf_counter() - started))
    LOGGER.info("  标的数            : %d", len(files))
    LOGGER.info("  日期范围          : %s → %s", window[0] if window else "-", window[-1] if window else "-")
    LOGGER.info("  覆盖不足的标的日次: %d（缺分钟合计 %d）", len(gap_rows), sum(r["missing_minutes"] for r in gap_rows))
    LOGGER.info("  常规时段全空的日次: %d", len(day_rows))
    worst = sorted(gap_rows, key=lambda r: -r["missing_minutes"])[:10]
    for r in worst:
        LOGGER.info("     %-14s %s 缺 %3d 分钟 (%s)", r["symbol"], r["date"], r["missing_minutes"],
                    r["missing_ranges_utc"][:60])
    LOGGER.info("  报告：%s / %s / %s", gap_path.name, day_path.name, sym_path.name)
    LOGGER.info("-" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
