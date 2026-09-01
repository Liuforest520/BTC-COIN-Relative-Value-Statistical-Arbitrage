from __future__ import annotations

import csv
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Iterator

import polars as pl

from core.modules.data.loader import COLUMN_ALIASES


ENGLISH_ALIASES = {
    "timestamp": "ts",
    "time": "date",
    "datetime": "date",
    "date": "date",
    "data_time": "date",
    "ts": "ts",
    "open": "open",
    "high": "high",
    "close": "close",
    "low": "low",
    "volume": "volume",
    "vol": "volume",
}


def iter_csv_bars(file_path: str | Path) -> Iterator[list]:
    """Stream normalized [ts, open, high, close, low, volume] bars from CSV."""
    path = Path(file_path)
    encoding = _detect_encoding(path)
    if encoding in {"utf-8-sig", "utf-8"}:
        yield from _iter_polars_csv_bars(path, encoding)
        return

    yield from _iter_python_csv_bars(path, encoding)


def _iter_polars_csv_bars(path: Path, encoding: str) -> Iterator[list]:
    """Use Polars' native parser while retaining bounded-memory batch iteration."""
    with path.open("r", encoding=encoding, newline="") as f:
        fieldnames = next(csv.reader(f), None)
    if not fieldnames:
        return

    column_map = _column_map(fieldnames)
    source_by_target = {target: source for source, target in column_map.items()}
    required = ["open", "high", "close", "low", "volume"]
    missing = [name for name in required if name not in source_by_target]
    if missing:
        raise ValueError(f"CSV missing OHLCV columns: {missing}")

    time_target = "ts" if "ts" in source_by_target else "date" if "date" in source_by_target else None
    if time_target is None:
        raise ValueError("CSV must contain a time column: timestamp, ts, data_time, or date")

    targets = [time_target, *required]
    sources = [source_by_target[target] for target in targets]
    schema_overrides = {
        source_by_target[name]: pl.Float64
        for name in required
    }
    schema_overrides[source_by_target[time_target]] = pl.Int64 if time_target == "ts" else pl.String
    reader = pl.read_csv_batched(
        path,
        columns=sources,
        schema_overrides=schema_overrides,
        batch_size=50_000,
        n_threads=1,
        ignore_errors=True,
        encoding="utf8-lossy" if encoding == "utf-8-sig" else "utf8",
    )

    last_ts = None
    pending_bar = None
    while True:
        batches = reader.next_batches(1)
        if not batches:
            break
        frame = batches[0].select(sources).filter(
            pl.all_horizontal([
                pl.col(source_by_target[name]).is_not_null()
                & pl.col(source_by_target[name]).is_finite()
                for name in required
            ])
        )
        for row in frame.iter_rows():
            try:
                ts = _parse_timestamp(row[0])
                bar = [ts, *row[1:]]
            except (TypeError, ValueError):
                continue

            if last_ts is not None and ts < last_ts:
                raise ValueError(f"CSV timestamps must be sorted ascending: {path}")
            if last_ts is not None and ts != last_ts and pending_bar is not None:
                yield pending_bar
            pending_bar = bar
            last_ts = ts

    if pending_bar is not None:
        yield pending_bar


def _iter_python_csv_bars(path: Path, encoding: str) -> Iterator[list]:
    """Fallback for legacy encodings unsupported by Polars' CSV reader."""
    with path.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return

        column_map = _column_map(reader.fieldnames)
        missing = [name for name in ["open", "high", "close", "low", "volume"] if name not in column_map.values()]
        if missing:
            raise ValueError(f"CSV missing OHLCV columns: {missing}")

        time_targets = {"ts", "date"}
        if not any(target in time_targets for target in column_map.values()):
            raise ValueError("CSV must contain a time column: timestamp, ts, data_time, or date")

        last_ts = None
        pending_bar = None
        for row in reader:
            bar = _parse_row(row, column_map)
            if bar is None:
                continue

            ts = int(bar[0])
            if last_ts is not None and ts < last_ts:
                raise ValueError(f"CSV timestamps must be sorted ascending: {path}")

            if last_ts is not None and ts != last_ts and pending_bar is not None:
                yield pending_bar

            pending_bar = bar
            last_ts = ts

        if pending_bar is not None:
            yield pending_bar


def _detect_encoding(path: Path) -> str:
    with path.open("rb") as f:
        sample = f.read(65536)
    for encoding in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            sample.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "utf-8-sig"


def _column_map(fieldnames: list[str]) -> dict[str, str]:
    result = {}
    used_targets = set()
    for name in fieldnames:
        clean = name.lstrip("\ufeff").strip()
        target = COLUMN_ALIASES.get(clean) or ENGLISH_ALIASES.get(clean.lower()) or clean
        if target == "timestamp":
            target = "ts"
        if target in {"data_time", "datetime", "time"}:
            target = "date"
        if target in used_targets:
            continue
        result[name] = target
        used_targets.add(target)
    return result


def _parse_row(row: dict[str, str], column_map: dict[str, str]) -> list | None:
    values = {}
    for source, target in column_map.items():
        values[target] = row.get(source)

    time_value = values.get("ts", values.get("date"))
    try:
        ts = _parse_timestamp(time_value)
        open_price = float(values["open"])
        high = float(values["high"])
        close = float(values["close"])
        low = float(values["low"])
        volume = float(values["volume"])
    except (TypeError, ValueError):
        return None

    if not all(isfinite(value) for value in (open_price, high, close, low, volume)):
        return None

    return [ts, open_price, high, close, low, volume]


def _parse_timestamp(value) -> int:
    if isinstance(value, (int, float)):
        parsed = int(value)
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("empty timestamp")
        try:
            parsed = int(float(text))
        except ValueError:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)

    if parsed > 10_000_000_000_000_000:
        return parsed // 1000
    if parsed > 10_000_000_000:
        return parsed
    return parsed * 1000
