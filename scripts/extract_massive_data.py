"""Decompress downloaded Massive daily flat files: <date>.csv.gz -> <date>.csv.

Input layout (written by scripts/download_massive_data.py):

    data/stocks/flat_files/minute_aggregates/2021/09/2021-09-28.csv.gz

Usage:

    python scripts/extract_massive_data.py                       # all days, keep .gz
    python scripts/extract_massive_data.py --limit 3 --overwrite # quick test
    python scripts/extract_massive_data.py --delete-gz           # free disk afterwards
"""

from __future__ import annotations

import argparse
import gzip
import logging
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "stocks" / "flat_files" / "minute_aggregates"
LOGGER = logging.getLogger("extract_massive_data")
PROGRESS_EVERY = 25
BUFFER = 8 * 1024 * 1024


def _format_bytes(size: float) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024.0
    return f"{value:.1f}TB"


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _peek_header(archive: Path, lines: int = 3) -> list[str]:
    try:
        with gzip.open(archive, "rt", encoding="utf-8", errors="replace") as handle:
            out = []
            for _ in range(lines):
                line = handle.readline()
                if not line:
                    break
                out.append(line.rstrip("\n"))
            return out
    except OSError as exc:
        return [f"<无法读取: {exc}>"]


def _extract_one(archive: Path, target: Path, overwrite: bool, delete_gz: bool) -> tuple[str, int, int]:
    """Return (status, compressed_bytes, plain_bytes)."""
    if target.exists() and target.stat().st_size > 0 and not overwrite:
        return "skipped", archive.stat().st_size, target.stat().st_size
    compressed = archive.stat().st_size
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".part")
    started = time.perf_counter()
    try:
        with gzip.open(archive, "rb") as src, temp.open("wb") as dst:
            shutil.copyfileobj(src, dst, BUFFER)
        temp.replace(target)
    except (OSError, EOFError) as exc:
        temp.unlink(missing_ok=True)
        LOGGER.error("解压失败 %s -> %s : %s", archive, target, exc)
        raise
    plain = target.stat().st_size
    elapsed = time.perf_counter() - started
    LOGGER.info(
        "解压完成 %s -> %s  %s -> %s  用时 %s  速度 %s/s",
        archive.name, target.name, _format_bytes(compressed), _format_bytes(plain),
        _format_duration(elapsed), _format_bytes(plain / elapsed if elapsed > 0 else 0.0),
    )
    if delete_gz:
        archive.unlink(missing_ok=True)
        LOGGER.info("已删除压缩包 %s", archive)
    return "extracted", compressed, plain


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Massive 日频 flat file 解压（.csv.gz -> .csv）")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help=f"默认 {DEFAULT_INPUT_DIR}")
    parser.add_argument("--out-dir", help="解压到统一目录；默认与压缩包同目录（逐日文件夹）")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 .csv")
    parser.add_argument("--delete-gz", action="store_true", help="解压成功后删除 .csv.gz")
    parser.add_argument("--limit", type=int, help="只处理前 N 个文件（用于试跑）")
    parser.add_argument("--start", help="只解压该日期（含）之后的文件夹，格式 YYYY-MM-DD")
    parser.add_argument("--end", help="只解压该日期（含）之前的文件夹，格式 YYYY-MM-DD")
    parser.add_argument("--clean", action="store_true",
                        help="删除已解压出来的 .csv（及残留 .part），不解压；配合 --start/--end 限定范围")
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)

    if args.clean:
        plains = sorted(list(input_dir.rglob("*.csv")) + list(input_dir.rglob("*.csv.part")))
        if args.start:
            plains = [p for p in plains if p.parent.name >= args.start]
        if args.end:
            plains = [p for p in plains if p.parent.name <= args.end]
        if not plains:
            LOGGER.info("没有需要删除的已解压文件（范围 %s → %s）", args.start or "-", args.end or "-")
            return 0
        LOGGER.info("=" * 96)
        LOGGER.info("清理已解压文件：%d 个，合计 %s", len(plains),
                    _format_bytes(sum(p.stat().st_size for p in plains)))
        LOGGER.info("  范围 : %s → %s", plains[0].parent.name, plains[-1].parent.name)
        LOGGER.info("  注意 : 只删除 .csv 明文，.csv.gz 压缩包保留")
        LOGGER.info("=" * 96)
        freed = removed = failed = 0
        for index, path in enumerate(plains, start=1):
            size = path.stat().st_size
            try:
                path.unlink()
            except OSError as exc:
                failed += 1
                LOGGER.error("删除失败 %s : %s", path.resolve(), exc)
                continue
            removed += 1
            freed += size
            LOGGER.info("[%d/%d] 已删除 %s (%s)", index, len(plains), path.resolve(), _format_bytes(size))
            if index % PROGRESS_EVERY == 0 or index == len(plains):
                LOGGER.info("进度 %d/%d | 已删 %d 失败 %d | 释放 %s",
                            index, len(plains), removed, failed, _format_bytes(freed))
        LOGGER.info("清理结束：删除 %d 个，失败 %d，释放 %s", removed, failed, _format_bytes(freed))
        return 1 if failed else 0

    archives = sorted(input_dir.rglob("*.csv.gz"))
    if args.start:
        archives = [a for a in archives if a.parent.name >= args.start]
    if args.end:
        archives = [a for a in archives if a.parent.name <= args.end]
    if not archives:
        LOGGER.error("在 %s 下没找到符合条件的 *.csv.gz（是不是还没下载？）", input_dir.resolve())
        return 1
    if args.limit:
        archives = archives[: args.limit]

    sample = _peek_header(archives[0], 3)
    LOGGER.info("=" * 96)
    LOGGER.info("Massive flat file 解压开始")
    LOGGER.info("  输入目录 : %s", input_dir.resolve())
    LOGGER.info("  待处理   : %d 个 .csv.gz", len(archives))
    LOGGER.info("  日期范围 : %s → %s%s", archives[0].parent.name, archives[-1].parent.name,
                f"（筛选 --start {args.start} --end {args.end}）" if (args.start or args.end) else "")
    LOGGER.info("  输出位置 : %s", (Path(args.out_dir).resolve() if args.out_dir else "<与压缩包同目录>/<日期>.csv"))
    LOGGER.info("  覆盖已有 : %s    解压后删除压缩包: %s",
                "是" if args.overwrite else "否（已存在则跳过）", "是" if args.delete_gz else "否")
    LOGGER.info("  样例文件 : %s", archives[0].name)
    for line in sample:
        LOGGER.info("     %s", line[:200])
    LOGGER.info("=" * 96)

    started_at = time.perf_counter()
    extracted = skipped = failed = 0
    compressed_total = plain_total = 0
    for index, archive in enumerate(archives, start=1):
        target = (Path(args.out_dir) / f"{archive.parent.name}.csv") if args.out_dir else archive.with_suffix("")
        LOGGER.info("[%d/%d] %s", index, len(archives), archive.resolve())
        try:
            status, cbytes, pbytes = _extract_one(archive, target, args.overwrite, args.delete_gz)
        except (OSError, EOFError):
            failed += 1
            continue
        if status == "extracted":
            extracted += 1
        else:
            skipped += 1
            LOGGER.info("已存在，跳过 %s (%s)", target.resolve(), _format_bytes(pbytes))
        compressed_total += cbytes
        plain_total += pbytes
        if index % PROGRESS_EVERY == 0 or index == len(archives):
            LOGGER.info(
                "进度 %d/%d | 解压 %d 跳过 %d 失败 %d | 压缩 %s -> 明文 %s | 已用 %s",
                index, len(archives), extracted, skipped, failed,
                _format_bytes(compressed_total), _format_bytes(plain_total),
                _format_duration(time.perf_counter() - started_at),
            )

    LOGGER.info("-" * 96)
    LOGGER.info("解压结束：共 %d，解压 %d，跳过 %d，失败 %d", len(archives), extracted, skipped, failed)
    LOGGER.info("  压缩总大小 %s -> 明文总大小 %s（膨胀 %.2f 倍）",
                _format_bytes(compressed_total), _format_bytes(plain_total),
                plain_total / compressed_total if compressed_total else 0.0)
    LOGGER.info("  总用时 %s", _format_duration(time.perf_counter() - started_at))
    LOGGER.info("-" * 96)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
