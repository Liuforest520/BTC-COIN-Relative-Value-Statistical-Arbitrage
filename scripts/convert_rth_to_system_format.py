"""Convert Massive adjusted RTH 1-minute CSV files to the project schema.

The source files in ``by_stock_adjusted_RTH`` contain columns such as
``dt_utc`` and ``ticker``.  The backtest input convention used by this
project is the Binance-like 1-minute schema:

    date,symbol,open_time,open_time_utc,open_time_date,open,high,low,close,
    volume,close_time,close_time_utc,close_time_date,amount,number_of_trades,
    buy_volume,buy_amount

Conversion is intentionally streaming so a complete source directory does
not need to fit in memory.  Source files are never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "research" / "新增tradfi交易对" / "by_stock_adjusted_RTH"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "tradfi_rth_compatible"
LOGGER = logging.getLogger("convert_rth_to_system_format")

OUTPUT_COLUMNS = [
    "date",
    "symbol",
    "open_time",
    "open_time_utc",
    "open_time_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "close_time_utc",
    "close_time_date",
    "amount",
    "number_of_trades",
    "buy_volume",
    "buy_amount",
]

UTC = timezone.utc
SHANGHAI = timezone(timedelta(hours=8))


def _system_symbol(source_symbol: str) -> str:
    symbol = source_symbol.strip().upper()
    return symbol if symbol.endswith("USDT") else f"{symbol}USDT"


def _source_ticker(system_symbol: str) -> str:
    symbol = system_symbol.strip().upper()
    return symbol[:-4] if symbol.endswith("USDT") else symbol


def _parse_timestamp(value: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty dt_utc")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _format_second(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _numeric(value: str, column: str) -> str:
    text = str(value or "").strip()
    number = float(text)
    if not math.isfinite(number):
        raise ValueError(f"{column} is not finite")
    return text


def _convert_file(source_path: Path, output_path: Path, symbol: str) -> dict:
    required = {"dt_utc", "open", "high", "low", "close", "volume"}
    source_rows = 0
    written_rows = 0
    skipped_rows = 0
    previous_ts = None
    first_ts = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with source_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            fields = {str(value).strip() for value in (reader.fieldnames or [])}
            missing = sorted(required - fields)
            if missing:
                raise ValueError(f"missing columns: {missing}")

            with temporary_path.open("w", encoding="utf-8", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=OUTPUT_COLUMNS)
                writer.writeheader()

                for row in reader:
                    source_rows += 1
                    try:
                        timestamp = _parse_timestamp(row.get("dt_utc"))
                        timestamp_ms = int(timestamp.timestamp() * 1000)
                        values = {
                            column: _numeric(row.get(column), column)
                            for column in ("open", "high", "low", "close", "volume")
                        }
                    except (TypeError, ValueError, OverflowError) as exc:
                        skipped_rows += 1
                        LOGGER.warning("%s row %d skipped: %s", source_path.name, source_rows, exc)
                        continue

                    if previous_ts is not None and timestamp_ms <= previous_ts:
                        raise ValueError(
                            f"timestamps must be strictly increasing at source row {source_rows}: "
                            f"{timestamp_ms} <= {previous_ts}"
                        )
                    previous_ts = timestamp_ms
                    if first_ts is None:
                        first_ts = timestamp_ms

                    close_timestamp = timestamp + timedelta(milliseconds=59_999)
                    writer.writerow({
                        "date": _format_second(timestamp),
                        "symbol": symbol,
                        "open_time": timestamp_ms,
                        "open_time_utc": _format_second(timestamp),
                        "open_time_date": _format_second(timestamp.astimezone(SHANGHAI)),
                        **values,
                        "close_time": timestamp_ms + 59_999,
                        "close_time_utc": _format_second(close_timestamp),
                        "close_time_date": _format_second(close_timestamp.astimezone(SHANGHAI)),
                        "amount": "0.0",
                        "number_of_trades": "0",
                        "buy_volume": "0.0",
                        "buy_amount": "0.0",
                    })
                    written_rows += 1

        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return {
        "source": str(source_path),
        "output": str(output_path),
        "symbol": symbol,
        "source_rows": source_rows,
        "written_rows": written_rows,
        "skipped_rows": skipped_rows,
        "first_open_time": first_ts,
        "last_open_time": previous_ts,
    }


def _is_system_format_file(path: Path) -> bool:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    return header == OUTPUT_COLUMNS


def _pair_file_symbols(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        columns = (
            ("标的A", "标的B")
            if {"标的A", "标的B"}.issubset(fields)
            else ("A", "B")
            if {"A", "B"}.issubset(fields)
            else None
        )
        if columns is None:
            raise ValueError(
                f"{path} must contain either 标的A/标的B or A/B columns"
            )
        values = []
        for row in reader:
            values.extend(row.get(column, "") for column in columns)
    return list(dict.fromkeys(_system_symbol(value) for value in values if str(value).strip()))


def _select_sources(
    source_dir: Path,
    symbols: str | None,
    pairs_file: Path | None,
) -> list[tuple[Path, str]]:
    if symbols and pairs_file:
        raise ValueError("--symbols and --pairs-file cannot be used together")
    if pairs_file:
        requested = _pair_file_symbols(pairs_file)
    elif symbols:
        requested = [_system_symbol(value) for value in symbols.split(",") if value.strip()]
    else:
        requested = []

    if requested:
        result = []
        for symbol in dict.fromkeys(requested):
            source = source_dir / f"{_source_ticker(symbol)}.csv"
            if source.is_file():
                result.append((source, symbol))
            else:
                LOGGER.warning("%s: source file missing; skipped", symbol)
        return result

    result = []
    for source in sorted(source_dir.glob("*.csv")):
        if source.name.startswith("_"):
            continue
        result.append((source, _system_symbol(source.stem)))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--symbols",
        help="optional comma-separated tickers, with or without the USDT suffix",
    )
    parser.add_argument(
        "--pairs-file",
        type=Path,
        help="CSV containing pair columns 标的A/标的B or A/B; converts only its symbols",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="replace each source CSV in its original directory after a successful conversion",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.source_dir.is_dir():
        parser.error(f"source directory does not exist: {args.source_dir}")
    if not args.in_place:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    skipped = []
    try:
        sources = _select_sources(args.source_dir, args.symbols, args.pairs_file)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    for source_path, symbol in sources:
        output_path = (
            source_path
            if args.in_place
            else args.output_dir / symbol / f"{symbol}-1m-extended.csv"
        )
        if args.in_place and _is_system_format_file(source_path):
            LOGGER.info("%s: already uses system schema, skipped", symbol)
            skipped.append({"symbol": symbol, "reason": "already_system_format", "output": str(source_path)})
            continue
        if output_path.exists() and not args.in_place and not args.overwrite:
            LOGGER.info("%s: output exists, skipped: %s", symbol, output_path)
            skipped.append({"symbol": symbol, "reason": "output_exists", "output": str(output_path)})
            continue
        try:
            stats = _convert_file(source_path, output_path, symbol)
            results.append(stats)
            LOGGER.info(
                "%s: converted rows=%d -> %s",
                symbol,
                stats["written_rows"],
                output_path,
            )
        except (OSError, ValueError, csv.Error) as exc:
            LOGGER.error("%s: conversion failed: %s", symbol, exc)
            skipped.append({"symbol": symbol, "reason": str(exc), "source": str(source_path)})

    manifest = {
        "source_dir": str(args.source_dir),
        "output_dir": str(args.source_dir if args.in_place else args.output_dir),
        "in_place": bool(args.in_place),
        "results": results,
        "skipped": skipped,
    }
    manifest_path = (
        args.source_dir / "rth_conversion_manifest.json"
        if args.in_place
        else args.output_dir / "rth_conversion_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("complete: converted=%d skipped=%d", len(results), len(skipped))
    return 1 if skipped and not results else 0


if __name__ == "__main__":
    raise SystemExit(main())
