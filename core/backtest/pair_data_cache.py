"""Compact, pair-local 1-minute data cache for fast walk-forward sweeps."""
from __future__ import annotations

from array import array
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.modules.data import iter_csv_bars, load_funding_data


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
    timestamps, values = _stream_aligned_ohlcv(
        x_path,
        y_path,
        start_ts=start_ts,
        end_ts=end_ts,
        measurement_start_ts=measurement_start_ts,
        warmup_bars=warmup_bars,
    )
    if timestamps.size == 0:
        raise NoAlignedPairDataError(
            "X/Y files have no overlapping 1-minute rows in requested range"
        )
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


def _stream_aligned_ohlcv(
    x_path: str | Path,
    y_path: str | Path,
    *,
    start_ts: int | None,
    end_ts: int | None,
    measurement_start_ts: int | None,
    warmup_bars: int,
):
    """Merge two sorted CSV streams while retaining only the requested rows."""
    warmup_bars = max(0, int(warmup_bars or 0))
    measurement_start_ts = (
        int(measurement_start_ts) if measurement_start_ts is not None else None
    )
    start_ts = int(start_ts) if start_ts is not None else None
    end_ts = int(end_ts) if end_ts is not None else None
    x_iter = iter_csv_bars(x_path, end_ts=end_ts, strict_memory=True)
    y_iter = iter_csv_bars(y_path, end_ts=end_ts, strict_memory=True)
    x_bar = next(x_iter, None)
    y_bar = next(y_iter, None)
    output = _CompactAlignedRows()
    warmup = _WarmupRing(warmup_bars)

    while x_bar is not None and y_bar is not None:
        x_ts = int(x_bar[0])
        y_ts = int(y_bar[0])
        if end_ts is not None and min(x_ts, y_ts) >= end_ts:
            break
        if x_ts < y_ts:
            x_bar = next(x_iter, None)
            continue
        if y_ts < x_ts:
            y_bar = next(y_iter, None)
            continue

        values = (
            float(x_bar[1]), float(x_bar[2]), float(x_bar[4]), float(x_bar[3]),
            float(y_bar[1]), float(y_bar[2]), float(y_bar[4]), float(y_bar[3]),
        )
        if measurement_start_ts is not None:
            if x_ts < measurement_start_ts:
                warmup.append(x_ts, values)
            else:
                output.append(x_ts, values)
        elif start_ts is None or x_ts >= start_ts:
            output.append(x_ts, values)

        x_bar = next(x_iter, None)
        y_bar = next(y_iter, None)

    warm_ts, warm_values = warmup.to_numpy()
    train_ts, train_values = output.to_numpy()
    if warm_ts.size:
        timestamps = np.ascontiguousarray(np.concatenate((warm_ts, train_ts)), dtype=np.int64)
        matrix = np.ascontiguousarray(np.vstack((warm_values, train_values)), dtype=np.float64)
    else:
        timestamps = train_ts
        matrix = train_values
    names = (
        "x_open", "x_high", "x_low", "x_close",
        "y_open", "y_high", "y_low", "y_close",
    )
    return timestamps, {
        name: np.ascontiguousarray(matrix[:, index], dtype=np.float64)
        for index, name in enumerate(names)
    }


class _CompactAlignedRows:
    def __init__(self):
        self.timestamps = array("q")
        self.values = [array("d") for _ in range(8)]

    def append(self, timestamp, values):
        self.timestamps.append(int(timestamp))
        for column, value in zip(self.values, values):
            column.append(float(value))

    def to_numpy(self):
        timestamps = np.asarray(self.timestamps, dtype=np.int64)
        if not self.timestamps:
            return timestamps, np.empty((0, 8), dtype=np.float64)
        matrix = np.column_stack([
            np.asarray(column, dtype=np.float64) for column in self.values
        ])
        return np.ascontiguousarray(timestamps), np.ascontiguousarray(matrix)


class _WarmupRing:
    def __init__(self, capacity):
        self.capacity = max(0, int(capacity))
        self.timestamps = np.empty(self.capacity, dtype=np.int64)
        self.values = np.empty((self.capacity, 8), dtype=np.float64)
        self.count = 0
        self.next_index = 0

    def append(self, timestamp, values):
        if self.capacity == 0:
            return
        self.timestamps[self.next_index] = int(timestamp)
        self.values[self.next_index] = values
        self.next_index = (self.next_index + 1) % self.capacity
        self.count = min(self.capacity, self.count + 1)

    def to_numpy(self):
        if self.count == 0:
            return np.empty(0, dtype=np.int64), np.empty((0, 8), dtype=np.float64)
        if self.count < self.capacity:
            indices = np.arange(self.count)
        else:
            indices = np.concatenate((
                np.arange(self.next_index, self.capacity),
                np.arange(0, self.next_index),
            ))
        return (
            np.ascontiguousarray(self.timestamps[indices], dtype=np.int64),
            np.ascontiguousarray(self.values[indices], dtype=np.float64),
        )


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
