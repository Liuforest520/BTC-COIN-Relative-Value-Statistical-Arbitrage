from pathlib import Path
from io import StringIO

import polars as pl


COLUMN_ALIASES = {
    "时间戳": "ts",
    "timestamp": "ts",
    "开盘价": "open",
    "最高价": "high",
    "最低价": "low",
    "收盘价": "close",
    "成交量": "volume",
    "vol": "volume",
}


def load_csv_data(file_path: str | Path) -> pl.DataFrame:
    data = _read_csv(file_path)
    data = _rename_columns(data)

    time_column = None
    for name in ["timestamp", "ts", "data_time", "date"]:
        if name in data.columns:
            time_column = name
            break

    if time_column is None:
        raise ValueError("CSV must contain a time column: timestamp, ts, data_time, or date")

    required_columns = ["open", "high", "close", "low", "volume"]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"CSV missing OHLCV columns: {missing_columns}")

    if time_column == "ts":
        time_expr = _timestamp_expr("ts")
    else:
        time_expr = (
            pl.col(time_column)
            .str.to_datetime(strict=False, time_zone="UTC")
            .dt.replace_time_zone(None)
            .dt.epoch(time_unit="ms")
        )

    data = data.with_columns(time_expr.alias("ts"))
    data = data.with_columns(pl.col(required_columns).cast(pl.Float64, strict=False))

    keep_columns = ["ts", *required_columns]
    optional_columns = [column for column in ["symbol", "volCcy"] if column in data.columns]

    return (
        data.select(keep_columns + optional_columns)
        .drop_nulls(subset=keep_columns)
        .sort("ts")
        .unique(subset=["ts"], keep="last", maintain_order=True)
    )


def _read_csv(file_path: str | Path) -> pl.DataFrame:
    raw = Path(file_path).read_bytes()
    for encoding in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            text = raw.decode(encoding)
            return pl.read_csv(StringIO(text))
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    return pl.read_csv(file_path)


def _rename_columns(data: pl.DataFrame) -> pl.DataFrame:
    rename_map = {}
    existing_targets = set(data.columns)

    for column in data.columns:
        clean_name = column.lstrip("\ufeff").strip()
        target = COLUMN_ALIASES.get(clean_name)
        if target is None or target in existing_targets:
            continue
        rename_map[column] = target
        existing_targets.add(target)

    if not rename_map:
        return data
    return data.rename(rename_map)


def _timestamp_expr(column: str):
    return (
        pl.when(pl.col(column).cast(pl.Int64, strict=False) > 10_000_000_000_000_000)
        .then(pl.col(column).cast(pl.Int64, strict=False) // 1000)
        .when(pl.col(column).cast(pl.Int64, strict=False) > 10_000_000_000)
        .then(pl.col(column).cast(pl.Int64, strict=False))
        .otherwise(pl.col(column).cast(pl.Int64, strict=False) * 1000)
    )
