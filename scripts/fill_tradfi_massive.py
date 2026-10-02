"""把 tradfi 分钟数据补全：常规交易时段（9:30-15:59 ET，半日市至 12:59）逐分钟补齐。

规则：
  - 缺失分钟用「上一分钟收盘价」填充（O=H=L=C=前收，volume=0，transactions=0）
  - 当天开盘前就缺的分钟，用「当天第一根可用 bar」回填
  - 盘前盘后（时段外）默认丢弃，可用 --keep-extended 保留
  - 输出新增列 filled：1=填充值，0=原始成交

输出：data/stocks/tradfi/<SYMBOL>/<SYMBOL>-1m-massive-filled.csv（原始文件不动）

用法：
    python scripts/fill_tradfi_massive.py --limit 2          # 试跑 2 个标的
    python scripts/fill_tradfi_massive.py                    # 全部
    python scripts/fill_tradfi_massive.py --symbols KORUUSDT,BTCUSDT
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "stocks" / "tradfi"
SESSION_OPEN = (9, 30)
SESSION_MINUTES = 390
EARLY_CLOSE_MINUTES = 210
NS_PER_MIN = 60_000_000_000
HEADER = ["ticker", "window_start", "open", "high", "low", "close", "volume",
          "transactions", "time_utc", "time_et", "time_utc_plus8", "filled"]
LOGGER = logging.getLogger("fill_tradfi")

try:
    TZ = ZoneInfo("America/New_York")
except Exception:  # noqa: BLE001
    TZ = None


def _dst_active(day: dt.date) -> bool:
    def nth_sunday(y: int, m: int, n: int) -> dt.date:
        first = dt.date(y, m, 1)
        return first + dt.timedelta(days=(6 - first.weekday()) % 7 + 7 * (n - 1))
    return nth_sunday(day.year, 3, 2) <= day < nth_sunday(day.year, 11, 1)


def early_close(day: dt.date) -> bool:
    def prev_td(d: dt.date) -> dt.date:
        cur = d - dt.timedelta(days=1)
        while cur.weekday() >= 5:
            cur -= dt.timedelta(days=1)
        return cur
    def observed(d: dt.date) -> dt.date:
        return d - dt.timedelta(days=1) if d.weekday() == 5 else (d + dt.timedelta(days=1) if d.weekday() == 6 else d)
    first = dt.date(day.year, 11, 1)
    thanksgiving = first + dt.timedelta(days=(3 - first.weekday()) % 7 + 21)
    eve = dt.date(day.year, 12, 24)
    if eve.weekday() >= 5:
        eve = prev_td(observed(dt.date(day.year, 12, 25)))
    return day in {prev_td(observed(dt.date(day.year, 7, 4))), thanksgiving + dt.timedelta(days=1), eve}


def session_minutes(day: dt.date) -> tuple[int, int]:
    expected = EARLY_CLOSE_MINUTES if early_close(day) else SESSION_MINUTES
    if TZ is not None:
        off = dt.datetime(day.year, day.month, day.day, *SESSION_OPEN, tzinfo=TZ).utcoffset().total_seconds() / 3600
    else:
        off = -4 if _dst_active(day) else -5
    start = dt.datetime(day.year, day.month, day.day, *SESSION_OPEN, tzinfo=dt.timezone.utc) - dt.timedelta(hours=off)
    lo = int(start.timestamp() // 60)
    return lo, lo + expected - 1


def _stamp(minute: int) -> tuple[str, str, str]:
    utc = dt.datetime.fromtimestamp(minute * 60, dt.timezone.utc)
    et = utc.astimezone(TZ) if TZ else utc + dt.timedelta(hours=-4 if _dst_active(utc.date()) else -5)
    bj = dt.datetime.fromtimestamp(minute * 60 + 8 * 3600, dt.timezone.utc)
    return utc.strftime("%Y-%m-%d %H:%M"), et.strftime("%Y-%m-%d %H:%M"), bj.strftime("%Y-%m-%d %H:%M")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="补齐 tradfi 分钟数据的常规交易时段")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--symbols", help="逗号分隔，默认全部")
    p.add_argument("--limit", type=int)
    p.add_argument("--keep-extended", action="store_true", help="保留盘前盘后（默认丢弃）")
    p.add_argument("--overwrite", action="store_true", help="覆盖已存在的 *-filled.csv")
    args = p.parse_args(argv)

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*/*-1m-massive.csv"))
    if args.symbols:
        wanted = {s.strip() for s in args.symbols.split(",") if s.strip()}
        files = [f for f in files if f.parent.name in wanted]
    if args.limit:
        files = files[: args.limit]
    if not files:
        LOGGER.error("没有找到 *-1m-massive.csv")
        return 1

    LOGGER.info("=" * 96)
    LOGGER.info("补齐 tradfi 常规交易时段（%d 分钟/天，半日市 %d 分钟）", SESSION_MINUTES, EARLY_CLOSE_MINUTES)
    LOGGER.info("  标的数 : %d", len(files))
    LOGGER.info("  盘前盘后: %s", "保留" if args.keep_extended else "丢弃")
    LOGGER.info("=" * 96)

    started = time.perf_counter()
    for index, path in enumerate(files, start=1):
        symbol = path.parent.name
        target = path.parent / f"{symbol}-1m-massive-filled.csv"
        if target.exists() and not args.overwrite:
            LOGGER.info("[%d/%d] %-14s 已存在，跳过（--overwrite 可覆盖）", index, len(files), symbol)
            continue
        df = pl.read_csv(path, has_header=False, columns=[0, 1, 2, 3, 4, 5, 6, 7],
                         new_columns=["ticker", "volume", "open", "close", "high", "low",
                                      "window_start", "transactions"],
                         schema_overrides={"ticker": pl.Utf8, "window_start": pl.Utf8})
        df = df.filter(pl.col("ticker") != "ticker")
        df = df.with_columns([
            pl.col("window_start").cast(pl.Int64, strict=False),
            pl.col("open").cast(pl.Float64, strict=False),
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("high").cast(pl.Float64, strict=False),
            pl.col("low").cast(pl.Float64, strict=False),
            pl.col("volume").cast(pl.Float64, strict=False),
            pl.col("transactions").cast(pl.Int64, strict=False),
        ]).drop_nulls("window_start").sort("window_start")

        bars: dict[int, dict] = {}
        for r in df.iter_rows(named=True):
            bars[int(r["window_start"]) // NS_PER_MIN] = r
        days = sorted({dt.datetime.fromtimestamp(m * 60, dt.timezone.utc).date() for m in bars})

        out_rows, filled_count, orig_count = [], 0, 0
        last_close = None
        for day in days:
            lo, hi = session_minutes(day)
            day_bars = {m: bars[m] for m in bars if lo <= m <= hi}
            if not day_bars:
                continue
            first_avail = min(day_bars)
            last_close = None
            for m in range(lo, hi + 1):
                utc_s, et_s, bj_s = _stamp(m)
                bar = day_bars.get(m)
                if bar is not None:
                    last_close = float(bar["close"])
                    out_rows.append([symbol, m * NS_PER_MIN, bar["open"], bar["high"], bar["low"],
                                     bar["close"], bar["volume"], bar["transactions"],
                                     utc_s, et_s, bj_s, 0])
                    orig_count += 1
                else:
                    src = last_close if last_close is not None else float(day_bars[first_avail]["open"])
                    out_rows.append([symbol, m * NS_PER_MIN, src, src, src, src, 0.0, 0,
                                     utc_s, et_s, bj_s, 1])
                    last_close = src
                    filled_count += 1
            if args.keep_extended:
                for m in sorted(bars):
                    if bars[m] is not None and lo <= m <= hi:
                        continue
                    if dt.datetime.fromtimestamp(m * 60, dt.timezone.utc).date() != day:
                        continue
                    b = bars[m]
                    utc_s, et_s, bj_s = _stamp(m)
                    out_rows.append([symbol, m * NS_PER_MIN, b["open"], b["high"], b["low"],
                                     b["close"], b["volume"], b["transactions"], utc_s, et_s, bj_s, 0])

        with target.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(HEADER)
            w.writerows(out_rows)
        LOGGER.info("[%d/%d] %-14s 交易日 %3d 原始 %6d 行 + 填充 %6d 行 = %6d 行 -> %s",
                    index, len(files), symbol, len(days), orig_count, filled_count,
                    len(out_rows), target.name)

    LOGGER.info("完成，用时 %.1fs", time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
