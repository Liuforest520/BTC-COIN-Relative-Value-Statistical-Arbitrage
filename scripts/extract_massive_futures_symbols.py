"""Extract the selected commodity futures from Massive minute flat files.

The input contains all contracts and spread instruments for an exchange-day.
This script keeps only single-leg dated contracts for the selected roots and
writes one audit-friendly CSV per project symbol. It does not roll contracts,
fill missing minutes, or alter prices.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "futures" / "minute_aggregates"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "research" / "新增tradfi交易对" / "massive_futures_1m_extracted"
LOGGER = logging.getLogger("extract_massive_futures_symbols")

MONTH_CODES = "FGHJKMNQUVXZ"

# Project symbol -> Massive single-leg futures root and expected exchange.
FUTURES = {
    "BZUSDT": ("BZ", "nymex"),
    "CLUSDT": ("CL", "nymex"),
    "COPPERUSDT": ("HG", "comex"),
    "NATGASUSDT": ("NG", "nymex"),
    "XAGUSDT": ("SI", "comex"),
    "XAUUSDT": ("GC", "comex"),
    "XPDUSDT": ("PA", "nymex"),
    "XPTUSDT": ("PL", "nymex"),
}

OUTPUT_COLUMNS = [
    "symbol",
    "contract",
    "exchange",
    "session_end_date",
    "window_start",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "dollar_volume",
    "transactions",
    "source_file",
]


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid date {value!r}; expected YYYY-MM-DD"
        ) from exc


def _date_from_file(path: Path) -> date | None:
    match = re.search(r"(\d{4}-\d{2}-\d{2})\.csv$", path.name)
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def _files(input_dir: Path, start: date | None, end: date | None) -> list[Path]:
    files = []
    for path in input_dir.rglob("*.csv"):
        day = _date_from_file(path)
        if day is None:
            continue
        if start is not None and day < start:
            continue
        if end is not None and day > end:
            continue
        files.append((day, path))
    return [path for _, path in sorted(files, key=lambda item: (item[0], str(item[1])))]


def _symbol_list(value: str | None) -> list[str]:
    if not value:
        return list(FUTURES)
    symbols = [item.strip().upper() for item in value.split(",") if item.strip()]
    unknown = sorted(set(symbols) - set(FUTURES))
    if unknown:
        raise ValueError(f"unsupported symbols: {', '.join(unknown)}")
    return list(dict.fromkeys(symbols))


def _contract_pattern(root: str) -> re.Pattern[str]:
    # Excludes spreads such as GCJ3-GCM3 and synthetic strings such as 0GCK3.
    return re.compile(rf"^{re.escape(root)}[{MONTH_CODES}]\d{{1,2}}$")


def extract(
    input_dir: Path,
    output_dir: Path,
    symbols: Iterable[str],
    start: date | None = None,
    end: date | None = None,
    overwrite: bool = False,
) -> dict:
    selected = list(symbols)
    files = _files(input_dir, start, end)
    if not files:
        raise ValueError(f"no extracted daily CSV files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    handles = {}
    writers = {}
    temp_paths = {}
    final_paths = {}
    stats = {}

    for symbol in selected:
        final = output_dir / f"{symbol}-1m-raw.csv"
        if final.exists() and not overwrite:
            LOGGER.info("%s: output exists, skipped: %s", symbol, final)
            continue
        final.parent.mkdir(parents=True, exist_ok=True)
        temp = final.with_suffix(final.suffix + ".part")
        handle = temp.open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        handles[symbol] = handle
        writers[symbol] = writer
        temp_paths[symbol] = temp
        final_paths[symbol] = final
        root, exchange = FUTURES[symbol]
        stats[symbol] = {
            "symbol": symbol,
            "root": root,
            "exchange": exchange,
            "rows": 0,
            "source_files": 0,
            "contracts": Counter(),
        }

    patterns = {symbol: _contract_pattern(FUTURES[symbol][0]) for symbol in stats}
    expected_exchange = {symbol: FUTURES[symbol][1] for symbol in stats}
    files_with_rows = defaultdict(set)

    try:
        for index, path in enumerate(files, start=1):
            exchange = path.parent.parent.parent.name
            # The expected exchange is checked per row target below. This also
            # makes the script tolerant of a combined input directory.
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                required = {
                    "ticker", "exchange", "session_end_date", "window_start",
                    "open", "high", "low", "close", "volume",
                    "dollar_volume", "transactions",
                }
                missing = sorted(required - set(reader.fieldnames or []))
                if missing:
                    raise ValueError(f"{path} missing columns: {missing}")
                for row in reader:
                    ticker = str(row.get("ticker") or "").strip()
                    for symbol, pattern in patterns.items():
                        if expected_exchange[symbol] != exchange or not pattern.fullmatch(ticker):
                            continue
                        writers[symbol].writerow({
                            "symbol": symbol,
                            "contract": ticker,
                            "exchange": exchange,
                            "session_end_date": row.get("session_end_date", ""),
                            "window_start": row.get("window_start", ""),
                            "open": row.get("open", ""),
                            "high": row.get("high", ""),
                            "low": row.get("low", ""),
                            "close": row.get("close", ""),
                            "volume": row.get("volume", ""),
                            "dollar_volume": row.get("dollar_volume", ""),
                            "transactions": row.get("transactions", ""),
                            "source_file": str(path),
                        })
                        stats[symbol]["rows"] += 1
                        stats[symbol]["contracts"][ticker] += 1
                        files_with_rows[symbol].add(path)
                        break

            if index == 1 or index % 50 == 0 or index == len(files):
                LOGGER.info("progress=%d/%d file=%s", index, len(files), path)
    finally:
        for handle in handles.values():
            handle.close()

    manifest = {
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "price_values_changed": False,
        "timestamp_values_changed": False,
        "excluded": {
            "spread_contracts": True,
            "non_dated_or_anomalous_tickers": True,
        },
        "symbols": {},
    }
    for symbol, item in stats.items():
        temp = temp_paths[symbol]
        final = final_paths[symbol]
        if item["rows"] == 0:
            temp.unlink(missing_ok=True)
            LOGGER.warning("%s: no matching rows", symbol)
            continue
        temp.replace(final)
        item["contracts"] = dict(sorted(item["contracts"].items()))
        item["source_files"] = len(files_with_rows[symbol])
        manifest["symbols"][symbol] = {**item, "output": str(final.resolve())}
        LOGGER.info(
            "%s: rows=%d contracts=%d source_files=%d -> %s",
            symbol,
            item["rows"],
            len(item["contracts"]),
            item["source_files"],
            final.resolve(),
        )

    manifest_path = output_dir / "extraction_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--symbols", help="comma-separated project symbols; default: all 8")
    parser.add_argument("--start", type=_parse_date)
    parser.add_argument("--end", type=_parse_date)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.start and args.end and args.end < args.start:
        parser.error("--end must be on or after --start")
    try:
        extract(
            args.input_dir,
            args.output_dir,
            _symbol_list(args.symbols),
            start=args.start,
            end=args.end,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, csv.Error) as exc:
        LOGGER.error("extraction failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
