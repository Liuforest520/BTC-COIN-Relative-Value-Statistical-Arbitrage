"""Compact, pair-local 1-minute data cache for fast walk-forward sweeps."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from core.modules.data import load_csv_data, load_funding_data


class NoAlignedPairDataError(ValueError):
    """The two legs have no common 1-minute bars in the requested interval."""


@dataclass
class PairDataCache:
    """Aligned arrays shared by every candidate in one pair/window."""

    timestamps: np.ndarray
    x_open: np.ndarray
    x_high: np.ndarray
    x_low: np.ndarray
    x_close: np.ndarray
    y_open: np.ndarray
    y_high: np.ndarray
    y_low: np.ndarray
    y_close: np.ndarray
    funding_x: np.ndarray
    funding_y: np.ndarray
    funding_x_event: np.ndarray
    funding_y_event: np.ndarray

    def __post_init__(self):
        n = len(self.timestamps)
        for name in (
            "x_open", "x_high", "x_low", "x_close",
            "y_open", "y_high", "y_low", "y_close",
            "funding_x", "funding_y", "funding_x_event", "funding_y_event",
        ):
            value = np.ascontiguousarray(getattr(self, name))
            if len(value) != n:
                raise ValueError(f"{name} length does not match timestamps")
            setattr(self, name, value)
        self.timestamps = np.ascontiguousarray(self.timestamps, dtype=np.int64)
        if n and np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("pair cache timestamps must be strictly increasing")

    @property
    def size(self) -> int:
        return int(self.timestamps.size)

    def close(self) -> None:
        """Release large arrays explicitly between pair windows."""
        self.timestamps = np.empty(0, dtype=np.int64)
        for field in ("x_open", "x_high", "x_low", "x_close", "y_open", "y_high", "y_low", "y_close", "funding_x", "funding_y"):
            setattr(self, field, np.empty(0, dtype=np.float64))
        self.funding_x_event = np.empty(0, dtype=np.bool_)
        self.funding_y_event = np.empty(0, dtype=np.bool_)


def load_pair_data(
    x_path: str | Path,
    y_path: str | Path,
    x_funding_path: str | Path | None = None,
    y_funding_path: str | Path | None = None,
    start_ts: int | None = None,
    end_ts: int | None = None,
    measurement_start_ts: int | None = None,
    warmup_bars: int = 0,
) -> PairDataCache:
    """Load and inner-align two 1-minute OHLCV files and funding events.

    ``start_ts`` is inclusive and ``end_ts`` is exclusive.  Funding events
    are assigned to the first aligned bar at or after their event timestamp,
    matching the exchange manager's "next available bar" settlement rule.
    """
    if not _is_1m_path(x_path) or not _is_1m_path(y_path):
        raise ValueError("Numba pair cache requires 1m OHLCV inputs")
    use_bar_warmup = measurement_start_ts is not None and int(warmup_bars or 0) > 0
    load_start_ts = None if use_bar_warmup else start_ts
    x = _load_ohlcv(x_path, "x", load_start_ts, end_ts)
    y = _load_ohlcv(y_path, "y", load_start_ts, end_ts)
    frame = x.join(y, on="ts", how="inner", coalesce=True).sort("ts")
    if use_bar_warmup and not frame.is_empty():
        measurement_start_ts = int(measurement_start_ts)
        train_index = int(frame["ts"].search_sorted(measurement_start_ts, side="left"))
        slice_start = max(0, train_index - int(warmup_bars))
        frame = frame.slice(slice_start)
    if frame.is_empty():
        raise NoAlignedPairDataError(
            "X/Y files have no overlapping 1-minute rows in requested range"
        )

    timestamps = np.ascontiguousarray(frame["ts"].to_numpy(), dtype=np.int64)
    values = {
        name: np.ascontiguousarray(frame[name].to_numpy(), dtype=np.float64)
        for name in (
            "x_open", "x_high", "x_low", "x_close",
            "y_open", "y_high", "y_low", "y_close",
        )
    }
    if np.any(~np.isfinite(values["x_close"])) or np.any(~np.isfinite(values["y_close"])):
        raise ValueError("pair cache contains non-finite close values")

    fx, fx_event = _funding_array(x_funding_path, timestamps, "x")
    fy, fy_event = _funding_array(y_funding_path, timestamps, "y")
    return PairDataCache(timestamps=timestamps, funding_x=fx, funding_y=fy,
                         funding_x_event=fx_event, funding_y_event=fy_event,
                         **values)


def _is_1m_path(path: str | Path) -> bool:
    """Reject obvious non-1m files while allowing plain 1m CSV filenames.

    Some adjusted RTH sources are named ``AMAT.csv`` rather than
    ``AMAT-1m.csv``.  Their schema and timestamps are still minute bars, so
    the absence of an ``1m`` token in the filename must not reject them.
    """
    name = Path(path).name.lower()
    if any(token in name for token in ("5m", "15m", "30m", "1h", "4h", "1d")):
        return False
    return True


def _load_ohlcv(path: str | Path, prefix: str, start_ts: int | None, end_ts: int | None) -> pl.DataFrame:
    frame = load_csv_data(Path(path))
    frame = frame.select(["ts", "open", "high", "low", "close"])
    frame = frame.filter(
        (pl.col("ts") >= int(start_ts)) if start_ts is not None else pl.lit(True)
    )
    if end_ts is not None:
        frame = frame.filter(pl.col("ts") < int(end_ts))
    frame = frame.with_columns(
        pl.col("ts").cast(pl.Int64),
        *[pl.col(c).cast(pl.Float64, strict=False) for c in ("open", "high", "low", "close")],
    ).drop_nulls()
    # A source file may contain duplicate timestamps after a merge.  The
    # normal loader keeps the last row, so retain that behavior here.
    return frame.unique(subset=["ts"], keep="last", maintain_order=True).sort("ts").rename({
        column: f"{prefix}_{column}" for column in ("open", "high", "low", "close")
    })


def _funding_array(path: str | Path | None, timestamps: np.ndarray, prefix: str):
    rates = np.zeros(timestamps.size, dtype=np.float64)
    events = np.zeros(timestamps.size, dtype=np.bool_)
    if not path or not Path(path).exists() or timestamps.size == 0:
        return rates, events
    frame = load_funding_data(Path(path), symbol=prefix)
    event_ts = frame["ts"].to_numpy()
    event_rates = frame["funding_rate"].to_numpy()
    for raw_ts, raw_rate in zip(event_ts, event_rates):
        if raw_ts is None or raw_rate is None:
            continue
        index = int(np.searchsorted(timestamps, int(raw_ts), side="left"))
        if index >= timestamps.size:
            continue
        # The complete backtest applies every event at the next bar.  Funding
        # files are unique by symbol/timestamp, but assignment also handles a
        # duplicate defensively by keeping the latest value.
        rates[index] = float(raw_rate)
        events[index] = True
    return rates, events


__all__ = ["PairDataCache", "NoAlignedPairDataError", "load_pair_data"]
