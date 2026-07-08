from pathlib import Path

import polars as pl

from core.modules.data.loader import _read_csv


FUNDING_INTERVAL_MS = 8 * 60 * 60 * 1000


def load_funding_data(file_path: str | Path, symbol: str | None = None) -> pl.DataFrame:
    frame = _read_csv(file_path)
    frame = _rename_funding_columns(frame)

    required_columns = ["funding_time", "funding_rate"]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"funding CSV missing columns: {missing_columns}")

    frame = frame.with_columns(
        pl.col("funding_time").cast(pl.Int64, strict=False),
        pl.col("funding_rate").cast(pl.Float64, strict=False),
    )
    frame = frame.with_columns(
        (pl.col("funding_time") // FUNDING_INTERVAL_MS * FUNDING_INTERVAL_MS).alias("ts")
    )

    if "symbol" not in frame.columns:
        if symbol is None:
            raise ValueError("funding CSV must contain symbol or caller must pass symbol")
        frame = frame.with_columns(pl.lit(symbol).alias("symbol"))

    return (
        frame.select(["symbol", "ts", "funding_rate"])
        .drop_nulls(["symbol", "ts", "funding_rate"])
        .unique(subset=["symbol", "ts"], keep="last", maintain_order=True)
        .sort(["symbol", "ts"])
    )


def build_funding_map(funding_frames: list[pl.DataFrame]) -> dict[int, dict[str, float]]:
    if not funding_frames:
        return {}

    frame = pl.concat(funding_frames, how="vertical")
    funding_map = {}
    for row in frame.iter_rows(named=True):
        ts = int(row["ts"])
        symbol = row["symbol"]
        funding_map.setdefault(ts, {})[symbol] = float(row["funding_rate"])
    return funding_map


def _rename_funding_columns(frame: pl.DataFrame) -> pl.DataFrame:
    rename_map = {}
    aliases = {
        "fundingTime": "funding_time",
        "funding_time": "funding_time",
        "fundingRate": "funding_rate",
        "funding_rate": "funding_rate",
        "symbol": "symbol",
    }

    for column in frame.columns:
        clean_name = column.lstrip("\ufeff").strip()
        target = aliases.get(clean_name)
        if target is not None and target not in frame.columns:
            rename_map[column] = target

    if not rename_map:
        return frame
    return frame.rename(rename_map)
