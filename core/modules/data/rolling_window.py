from __future__ import annotations

from collections import deque
from itertools import islice
from typing import Iterable, Iterator, TypeVar

import numpy as np


T = TypeVar("T")


def ensure_deque(values: Iterable[T] | None, maxlen: int) -> deque[T]:
    """Return a bounded deque, preserving the most recent values."""
    maxlen = max(1, int(maxlen))
    if isinstance(values, deque) and values.maxlen == maxlen:
        return values
    if values is None:
        return deque(maxlen=maxlen)
    return deque(values, maxlen=maxlen)


def tail_values(values: Iterable[T] | None, count: int) -> list[T]:
    """Return the most recent count values from a list/deque-like object."""
    if values is None:
        return []
    count = max(0, int(count))
    if count == 0:
        return []

    length = len(values)  # type: ignore[arg-type]
    if length <= count:
        return list(values)

    if isinstance(values, deque):
        recent = list(islice(reversed(values), count))
        recent.reverse()
        return recent
    return list(values[-count:])  # type: ignore[index]


def tail_iter(values: Iterable[T] | None, count: int) -> Iterator[T]:
    """Iterate over the most recent count values without materializing a list."""
    if values is None:
        return iter(())
    count = max(0, int(count))
    if count == 0:
        return iter(())

    length = len(values)  # type: ignore[arg-type]
    if length <= count:
        return iter(values)

    if isinstance(values, deque):
        recent = list(islice(reversed(values), count))
        recent.reverse()
        return iter(recent)
    return iter(values[-count:])  # type: ignore[index]


def percentiles_from_tail(
    values: Iterable[float] | None,
    count: int,
    percentiles: Iterable[float],
) -> list[float | None]:
    """Calculate exact percentiles over the most recent values without full sorting."""
    requested = list(percentiles)
    window = list(tail_iter(values, count))
    if not window:
        return [None for _ in requested]
    if len(window) == 1:
        only = float(window[0])
        return [only for _ in requested]

    arr = np.asarray(window, dtype=float)
    if arr.size == 0:
        return [None for _ in requested]
    if not np.all(np.isfinite(arr)):
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return [None for _ in requested]
        if arr.size == 1:
            only = float(arr[0])
            return [only for _ in requested]

    ranks = [_percentile_rank(arr.size, percentile) for percentile in requested]
    needed = sorted({idx for lower, upper, _weight in ranks for idx in (lower, upper)})
    partitioned = np.partition(arr, needed)
    return [
        _percentile_from_partitioned(partitioned, lower, upper, weight)
        for lower, upper, weight in ranks
    ]


def _percentile_rank(length: int, percentile: float) -> tuple[int, int, float]:
    percentile = min(100.0, max(0.0, float(percentile)))
    rank = percentile / 100.0 * (length - 1)
    lower = int(rank)
    upper = min(lower + 1, length - 1)
    weight = rank - lower
    return lower, upper, weight


def _percentile_from_partitioned(values, lower: int, upper: int, weight: float) -> float:
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)
