"""把 Massive 美股分钟数据按 tradfi 标的抽取到各自的 data/<SYMBOL>/ 文件夹。

输出：data/<SYMBOL>/<SYMBOL>-1m-massive.csv
  列：ticker,volume,open,close,high,low,window_start,transactions,time_utc,time_utc_plus8
  （保留 Massive 原始字段，额外附两列可读时间；与币安数据分开存放，不做合并）

tradfi 标的从配置的 pair 定义里读取（带 tradfi 标签的 pair 的两条腿），
符号映射规则：去掉 "USDT" 后缀 -> 美股 ticker（例：NVDAUSDT -> NVDA）。
Massive 里找不到的 ticker 会单独列出来。

用法：
    python scripts/extract_tradfi_from_massive.py --limit 2          # 试跑 2 天
    python scripts/extract_tradfi_from_massive.py --overwrite         # 全量覆盖重写
    python scripts/extract_tradfi_from_massive.py                     # 全量追加
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import time
from pathlib import Path

import polars as pl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "massive_stocks"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
LOGGER = logging.getLogger("extract_tradfi")
HEADER = ["ticker", "volume", "open", "close", "high", "low", "window_start",
          "transactions", "time_utc", "time_utc_plus8"]


def _fmt_dur(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m < 60 else f"{m // 60}h{m % 60:02d}m"


def _tradfi_symbols(config_path: Path) -> tuple[list[str], dict[str, str]]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    setup = cfg["setups"][cfg["active_setup"]]
    symbols: list[str] = []
    for pair in setup["pairs"]:
        if "tradfi" not in (pair.get("tags") or []):
            continue
        for key in ("x_symbol", "y_symbol"):
            sym = pair.get(key)
            if sym and sym not in symbols:
                symbols.append(sym)
    return symbols, {s: s[:-4] if s.endswith("USDT") else s for s in symbols}


def _readable(ts_ns: int, offset_hours: int = 0) -> str:
    return dt.datetime.fromtimestamp(
        ts_ns // 10**9 + offset_hours * 3600, dt.timezone.utc
    ).strftime("%Y-%m-%d %H:%M")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="按 tradfi 标的抽取 Massive 分钟数据")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--start", help="起始日期 YYYY-MM-DD")
    p.add_argument("--end", help="结束日期 YYYY-MM-DD")
    p.add_argument("--limit", type=int, help="只处理前 N 天")
    p.add_argument("--overwrite", action="store_true", help="覆盖已有输出（默认追加）")
    p.add_argument("--flush-every", type=int, default=50, help="每 N 天落盘一次")
    args = p.parse_args(argv)

    symbols, mapping = _tradfi_symbols(Path(args.config))
    tickers = sorted(set(mapping.values()))
    LOGGER.info("=" * 96)
    LOGGER.info("Massive -> tradfi 标的抽取")
    LOGGER.info("  配置      : %s", Path(args.config).resolve())
    LOGGER.info("  tradfi 标的: %d 个（币安符号 -> 美股 ticker）", len(symbols))
    LOGGER.info("  映射示例  : %s", ", ".join(f"{k}->{v}" for k, v in list(mapping.items())[:8]))
    LOGGER.info("  数据源    : %s", Path(args.input_dir).resolve())
    LOGGER.info("  输出      : <data>/<SYMBOL>/<SYMBOL>-1m-massive.csv")
    LOGGER.info("  写入模式  : %s", "覆盖" if args.overwrite else "追加")
    LOGGER.info("=" * 96)

    files = sorted(Path(args.input_dir).glob("*/*.csv"))
    if args.start:
        files = [f for f in files if f.parent.name >= args.start]
    if args.end:
        files = [f for f in files if f.parent.name <= args.end]
    if args.limit:
        files = files[: args.limit]
    if not files:
        LOGGER.error("没有可处理的数据文件")
        return 1

    data_dir = Path(args.data_dir)
    ticker_to_symbol = {v: k for k, v in mapping.items()}
    buffers: dict[str, list[list]] = {t: [] for t in tickers}
    written: dict[str, int] = {t: 0 for t in tickers}
    found: set[str] = set()
    started_at = time.perf_counter()
    first_write = not args.overwrite

    def flush() -> None:
        nonlocal first_write
        for ticker, rows in buffers.items():
            if not rows:
                continue
            symbol = ticker_to_symbol[ticker]
            target = data_dir / symbol / f"{symbol}-1m-massive.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            # 表头只在"文件不存在"或"覆盖模式的第一批"时写；
            # 否则新出现的标的（首日无数据、后期才上市）会以追加模式创建出无表头文件。
            need_header = (not target.exists()) or (args.overwrite and first_write)
            with target.open("w" if need_header else "a", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                if need_header:
                    w.writerow(HEADER)
                w.writerows(rows)
            written[ticker] += len(rows)
            LOGGER.info("  落盘 %-12s +%d 行 -> %s", ticker, len(rows), target.resolve())
            rows.clear()
        first_write = False

    for index, path in enumerate(files, start=1):
        day = path.parent.name
        df = pl.read_csv(
            path,
            columns=["ticker", "volume", "open", "close", "high", "low",
                     "window_start", "transactions"],
            schema_overrides={"volume": pl.Float64, "open": pl.Float64, "close": pl.Float64,
                              "high": pl.Float64, "low": pl.Float64,
                              "window_start": pl.Int64, "transactions": pl.Int64},
        )
        df = df.filter(pl.col("ticker").is_in(tickers))
        if df.height:
            kept = 0
            for r in df.iter_rows(named=True):
                ticker = r["ticker"]
                found.add(ticker)
                ts = int(r["window_start"])
                buffers[ticker].append([ticker, r["volume"], r["open"], r["close"], r["high"],
                                        r["low"], ts, r["transactions"],
                                        _readable(ts), _readable(ts, 8)])
                kept += 1
            LOGGER.info("[%d/%d] %s 命中 %d 个标的 / %d 行",
                        index, len(files), day, df["ticker"].n_unique(), kept)
        else:
            LOGGER.warning("[%d/%d] %s 没有命中任何 tradfi 标的", index, len(files), day)
        if index % args.flush_every == 0:
            flush()
    flush()

    missing = [t for t in tickers if t not in found]
    LOGGER.info("-" * 96)
    LOGGER.info("抽取完成，用时 %s", _fmt_dur(time.perf_counter() - started_at))
    LOGGER.info("  处理天数   : %d（%s → %s）", len(files), files[0].parent.name, files[-1].parent.name)
    LOGGER.info("  命中标的   : %d / %d", len(found), len(tickers))
    LOGGER.info("  写入总行数 : %d", sum(written.values()))
    for ticker in sorted(written, key=lambda t: -written[t]):
        if written[ticker]:
            LOGGER.info("     %-12s %8d 行  (data/%s/%s-1m-massive.csv)",
                        ticker, written[ticker], ticker_to_symbol[ticker], ticker_to_symbol[ticker])
    if missing:
        LOGGER.warning("  未找到的标的（%d 个，Massive 中无对应美股 ticker）: %s",
                       len(missing), ", ".join(missing))
    LOGGER.info("-" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
