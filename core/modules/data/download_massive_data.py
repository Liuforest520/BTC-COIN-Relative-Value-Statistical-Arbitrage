"""Download Massive daily US stock minute-aggregate flat files.

Massive publishes the stock minute aggregate dataset as one gzip-compressed
CSV per UTC date. Each daily file already contains all available tickers, so
this downloader intentionally does not accept a symbol filter.

Output layout:

    data/stocks/flat_files/minute_aggregates/
      2025/01/2025-01-01.csv.gz
      2025/01/2025-01-02.csv.gz

Credentials can be supplied through config/massive_key.config, environment
variables, command-line options, or the empty constants below. Never commit
real credentials.
"""

from __future__ import annotations

import argparse
import configparser
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

try:
    import boto3
    from botocore.client import Config as BotoConfig
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError as exc:  # pragma: no cover - depends on local environment
    boto3 = None
    BotoConfig = None
    BotoCoreError = Exception
    ClientError = Exception
    _BOTO3_IMPORT_ERROR = exc
else:
    _BOTO3_IMPORT_ERROR = None


# Fill these locally only if you do not use config/massive_key.config or env vars.
MASSIVE_ACCESS_KEY = ""
MASSIVE_SECRET_KEY = ""

MASSIVE_ENDPOINT = "https://files.massive.com"
MASSIVE_BUCKET = "flatfiles"
DEFAULT_DATASET_PREFIX = "us_stocks_sip/minute_aggs_v1"
DEFAULT_OUTPUT_DIR = Path("data/stocks/flat_files/minute_aggregates")
DEFAULT_CREDENTIALS_FILE = Path("config/massive_key.config")
LOGGER = logging.getLogger("download_massive_data")


@dataclass(frozen=True)
class DownloadedDay:
    day: date
    path: Path
    downloaded: bool


class MassiveDownloadError(RuntimeError):
    """Raised when one or more Massive flat files cannot be downloaded."""


def _load_credentials_file(path: str | Path) -> tuple[str, str]:
    """Load Massive S3 credentials from a local, gitignored INI file."""
    config_path = Path(path)
    if not config_path.exists():
        return "", ""

    parser = configparser.ConfigParser(interpolation=None)
    parser.read(config_path, encoding="utf-8")
    if not parser.has_section("massive"):
        raise ValueError(f"Massive credentials file must contain [massive]: {config_path}")
    section = parser["massive"]
    access_key = section.get("aws_access_key_id", fallback="").strip()
    secret_key = section.get("aws_secret_access_key", fallback="").strip()
    return access_key, secret_key


def _build_client(access_key: str, secret_key: str):
    if boto3 is None:
        raise RuntimeError(
            "boto3 is required for Massive downloads; install project dependencies "
            "or run pip install boto3."
        ) from _BOTO3_IMPORT_ERROR

    # Each worker owns its client so no mutable client/session state is shared.
    return boto3.client(
        "s3",
        endpoint_url=MASSIVE_ENDPOINT,
        region_name="us-east-1",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=BotoConfig(
            signature_version="s3v4",
            retries={"max_attempts": 5, "mode": "standard"},
            connect_timeout=30,
            read_timeout=600,
        ),
    )


def _parse_day(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid date {value!r}; expected YYYY-MM-DD"
        ) from exc


def _date_range(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _object_key(day: date, dataset_prefix: str) -> str:
    prefix = dataset_prefix.strip("/")
    return f"{prefix}/{day:%Y}/{day:%m}/{day:%Y-%m-%d}.csv.gz"


def _is_not_found(exc: BaseException) -> bool:
    if not isinstance(exc, ClientError):
        return False
    code = str(exc.response.get("Error", {}).get("Code", ""))
    return code in {"404", "NoSuchKey", "NotFound"}


PROGRESS_EVERY = 10


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


def _progress_line(
    done: int, total: int, fresh: int, skipped: int, missing: int,
    failed: int, bytes_total: float, elapsed: float, workers: int,
) -> str:
    percent = done / total * 100 if total else 100.0
    speed = bytes_total / elapsed if elapsed > 0 else 0.0
    remaining = (total - done) * (elapsed / done) if done else 0.0
    return (
        f"进度 {done}/{total} ({percent:5.1f}%) | 新增 {fresh} 跳过 {skipped} "
        f"缺失 {missing} 失败 {failed} | 总量 {_format_bytes(bytes_total)} "
        f"({_format_bytes(speed)}/s) | 已用 {_format_duration(elapsed)} "
        f"预计剩余 {_format_duration(remaining)} [workers={workers}]"
    )


def _download_one_day(
    day: date,
    output_dir: Path,
    access_key: str,
    secret_key: str,
    dataset_prefix: str,
    overwrite: bool,
    progress: str = "",
) -> DownloadedDay | None:
    key = _object_key(day, dataset_prefix)
    day_dir = output_dir / f"{day:%Y}" / f"{day:%m}"
    target = day_dir / f"{day:%Y-%m-%d}.csv.gz"

    if target.exists() and target.stat().st_size > 0 and not overwrite:
        size = target.stat().st_size
        LOGGER.info("%s 已存在，跳过 %s (%s)", progress, target.resolve(), _format_bytes(size))
        return DownloadedDay(day=day, path=target, downloaded=False)

    day_dir.mkdir(parents=True, exist_ok=True)
    temp_path = day_dir / f"{target.name}.{uuid.uuid4().hex}.part"
    LOGGER.info(
        "%s 开始下载 day=%s\n        来源 s3://%s/%s\n        落盘 %s",
        progress, day, MASSIVE_BUCKET, key, target.resolve(),
    )
    started = time.perf_counter()
    client = _build_client(access_key, secret_key)
    try:
        client.download_file(MASSIVE_BUCKET, key, str(temp_path))
        temp_path.replace(target)
        elapsed = time.perf_counter() - started
        size = target.stat().st_size
        speed = size / elapsed if elapsed > 0 else 0.0
        LOGGER.info(
            "%s 下载完成 day=%s 大小 %s 用时 %s 速度 %s/s -> %s",
            progress, day, _format_bytes(size), _format_duration(elapsed),
            _format_bytes(speed), target.resolve(),
        )
        return DownloadedDay(day=day, path=target, downloaded=True)
    except (ClientError, BotoCoreError, OSError) as exc:
        temp_path.unlink(missing_ok=True)
        if _is_not_found(exc):
            # Weekends and market holidays normally have no daily object.
            LOGGER.warning("%s 无文件（周末或休市）day=%s key=%s，跳过", progress, day, key)
            return None
        LOGGER.error(
            "%s 下载失败 day=%s key=%s 用时 %s 错误 %s",
            progress, day, key, _format_duration(time.perf_counter() - started), exc,
        )
        raise MassiveDownloadError(f"failed to download {key}: {exc}") from exc


def download_massive_data(
    start: date,
    end: date,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    workers: int = 8,
    access_key: str | None = None,
    secret_key: str | None = None,
    credentials_file: str | Path = DEFAULT_CREDENTIALS_FILE,
    dataset_prefix: str = DEFAULT_DATASET_PREFIX,
    overwrite: bool = False,
) -> dict[str, int]:
    """Download one all-symbol Massive flat file for each UTC date."""
    if end < start:
        raise ValueError("end date must be on or after start date")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    output_path = Path(output_dir)
    credential_source = "命令行参数"
    file_access_key, file_secret_key = _load_credentials_file(credentials_file)
    if not (access_key and secret_key):
        if os.getenv("MASSIVE_ACCESS_KEY") and os.getenv("MASSIVE_SECRET_KEY"):
            credential_source = "环境变量 MASSIVE_ACCESS_KEY / MASSIVE_SECRET_KEY"
        elif file_access_key and file_secret_key:
            credential_source = f"配置文件 {Path(credentials_file).resolve()}"
        elif MASSIVE_ACCESS_KEY and MASSIVE_SECRET_KEY:
            credential_source = "模块内常量"
    access_key = (
        access_key
        or os.getenv("MASSIVE_ACCESS_KEY")
        or file_access_key
        or MASSIVE_ACCESS_KEY
    )
    secret_key = (
        secret_key
        or os.getenv("MASSIVE_SECRET_KEY")
        or file_secret_key
        or MASSIVE_SECRET_KEY
    )
    if not access_key or not secret_key:
        raise ValueError(
            "Massive credentials are missing. Set MASSIVE_ACCESS_KEY and "
            "MASSIVE_SECRET_KEY, or fill config/massive_key.config."
        )

    days = list(_date_range(start, end))
    downloaded_days: list[DownloadedDay] = []
    failures: list[BaseException] = []

    prefix = dataset_prefix.strip("/")
    LOGGER.info("=" * 96)
    LOGGER.info("Massive 行情数据下载开始")
    LOGGER.info("  日期范围   : %s → %s（共 %d 天）", start, end, len(days))
    LOGGER.info("  远端位置   : s3://%s/%s/{YYYY}/{MM}/{YYYY-MM-DD}.csv.gz", MASSIVE_BUCKET, prefix)
    LOGGER.info("  输出根目录 : %s", output_path.resolve())
    LOGGER.info("  每日落盘   : <输出根目录>/<YYYY-MM-DD>/<YYYY-MM-DD>.csv.gz")
    LOGGER.info("  并发线程   : %d   覆盖已有: %s", workers, "是" if overwrite else "否（已存在则跳过）")
    LOGGER.info("  凭据来源   : %s", credential_source)
    LOGGER.info("=" * 96)

    started_at = time.perf_counter()
    fresh = skipped = missing = total_bytes = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="massive") as pool:
        futures = {}
        for index, day in enumerate(days, start=1):
            futures[pool.submit(
                _download_one_day,
                day,
                output_path,
                access_key,
                secret_key,
                dataset_prefix,
                overwrite,
                f"[{index}/{len(days)}]",
            )] = day
        for done, future in enumerate(as_completed(futures), start=1):
            day = futures[future]
            try:
                result = future.result()
            except BaseException as exc:
                failures.append(exc)
                LOGGER.error("下载失败 day=%s: %s", day, exc)
            else:
                if result is None:
                    missing += 1
                elif result.downloaded:
                    fresh += 1
                    downloaded_days.append(result)
                    try:
                        total_bytes += result.path.stat().st_size
                    except OSError:
                        pass
                else:
                    skipped += 1
                    downloaded_days.append(result)
                    try:
                        total_bytes += result.path.stat().st_size
                    except OSError:
                        pass
            if done % PROGRESS_EVERY == 0 or done == len(days):
                LOGGER.info(_progress_line(
                    done, len(days), fresh, skipped, missing, len(failures),
                    total_bytes, time.perf_counter() - started_at, workers,
                ))

    elapsed_total = time.perf_counter() - started_at
    LOGGER.info("-" * 96)
    LOGGER.info("Massive 行情数据下载结束")
    LOGGER.info("  请求天数     : %d", len(days))
    LOGGER.info("  本次新下载   : %d", fresh)
    LOGGER.info("  已存在跳过   : %d", skipped)
    LOGGER.info("  无文件(休市) : %d", missing)
    LOGGER.info("  失败         : %d", len(failures))
    LOGGER.info("  文件总量     : %s", _format_bytes(total_bytes))
    LOGGER.info("  总用时       : %s（平均 %s/天）",
                _format_duration(elapsed_total),
                _format_duration(elapsed_total / len(days)) if days else "-")
    LOGGER.info("  输出目录     : %s", output_path.resolve())
    LOGGER.info("-" * 96)

    if failures:
        raise MassiveDownloadError(
            f"{len(failures)} of {len(days)} daily downloads failed; "
            f"first error: {failures[0]}"
        ) from failures[0]

    summary = {
        "requested_days": len(days),
        "downloaded_days": fresh,
        "skipped_days": skipped,
        "missing_or_closed_days": missing,
    }
    LOGGER.info("Massive download complete: %s", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download daily all-symbol Massive stock minute aggregates."
    )
    parser.add_argument("--start", required=True, type=_parse_day, help="inclusive UTC date, YYYY-MM-DD")
    parser.add_argument("--end", required=True, type=_parse_day, help="inclusive UTC date, YYYY-MM-DD")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"daily output root (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--workers", type=int, default=8, help="parallel daily downloads (default: 8)")
    parser.add_argument("--access-key", help="Massive access key; prefer the local config file or MASSIVE_ACCESS_KEY")
    parser.add_argument("--secret-key", help="Massive secret key; prefer the local config file or MASSIVE_SECRET_KEY")
    parser.add_argument(
        "--credentials-file",
        default=str(DEFAULT_CREDENTIALS_FILE),
        help=f"local INI credentials file (default: {DEFAULT_CREDENTIALS_FILE})",
    )
    parser.add_argument(
        "--dataset-prefix",
        default=DEFAULT_DATASET_PREFIX,
        help=f"S3 object prefix (default: {DEFAULT_DATASET_PREFIX})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="redownload and replace existing daily .csv.gz files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    try:
        download_massive_data(
            args.start,
            args.end,
            output_dir=args.output_dir,
            workers=args.workers,
            access_key=args.access_key,
            secret_key=args.secret_key,
            credentials_file=args.credentials_file,
            dataset_prefix=args.dataset_prefix,
            overwrite=args.overwrite,
        )
    except (ValueError, MassiveDownloadError, RuntimeError, OSError) as exc:
        LOGGER.error("Massive download failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
