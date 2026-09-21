from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import re
from math import ceil


_TIMEFRAME_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(m|min|h|d)?\s*$",
    re.IGNORECASE,
)


def timeframe_to_minutes(value) -> int:
    """Normalize a model timeframe such as 1m, 15m or 1h."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minutes = float(value)
    else:
        text = str(value or "1m").strip().lower()
        match = _TIMEFRAME_RE.fullmatch(text)
        if not match:
            raise ValueError(f"invalid model timeframe: {value!r}")
        number, unit = match.groups()
        multiplier = {None: 1.0, "m": 1.0, "min": 1.0, "h": 60.0, "d": 1440.0}[unit]
        minutes = float(number) * multiplier
    if minutes < 1 or not minutes.is_integer():
        raise ValueError("model timeframe must be a positive whole number of minutes")
    return int(minutes)


def timeframe_bars(raw_minutes, timeframe_minutes: int, *, label: str) -> int:
    """Convert a source-minute window to complete model bars."""
    value = float(raw_minutes)
    if value <= 0 or not value == value:
        raise ValueError(f"{label} must be positive")
    return max(1, int(ceil(value / float(timeframe_minutes))))


@dataclass
class _Bucket:
    start_ts: int
    last_ts: int
    count: int
    open: float
    high: float
    close: float
    low: float
    volume: float
    contiguous: bool = True

    def append(self, bar: list) -> None:
        ts = int(bar[0])
        if ts != self.last_ts + 60_000:
            self.contiguous = False
        self.last_ts = ts
        self.count += 1
        self.high = max(self.high, float(bar[2]))
        self.close = float(bar[3])
        self.low = min(self.low, float(bar[4]))
        self.volume += float(bar[5])

    def complete(self, interval_minutes: int) -> bool:
        expected = int(interval_minutes)
        return (
            self.count == expected
            and self.start_ts % (expected * 60_000) == 0
            and self.last_ts - self.start_ts == (expected - 1) * 60_000
            and self.contiguous
        )

    def bar(self) -> list:
        return [self.last_ts, self.open, self.high, self.close, self.low, self.volume]


class PairTimeframeResampler:
    """Stream 1-minute bars into completed model bars for active pairs."""

    def __init__(self, interval_minutes: int, pair_specs: list[tuple]):
        self.interval_minutes = timeframe_to_minutes(interval_minutes)
        self.interval_ms = self.interval_minutes * 60_000
        self.pair_specs = list(pair_specs)
        self._buckets: dict[tuple[str, str], _Bucket] = {}
        # Incomplete/non-contiguous source buckets are intentionally not
        # emitted as model bars.  Keep a small diagnostic trail so callers can
        # surface data-quality issues instead of silently losing decisions.
        self.discarded_bucket_count = 0
        self.discarded_bucket_examples: list[dict] = []
        # Keep only completed bars that have not yet been aligned for a Pair.
        # This avoids scanning/storing the entire model history on every raw bar.
        self._pair_queues: dict[str, dict[str, deque]] = {}
        self._pairs_by_leg: dict[tuple[str, str], list[str]] = defaultdict(list)
        for pair_id, x_exchange, x_symbol, y_exchange, y_symbol in self.pair_specs:
            pair_id = str(pair_id)
            self._pair_queues[pair_id] = {
                # A lagging/missing leg cannot match bars older than its
                # next completed timestamp; bounding both queues prevents a
                # one-sided data stream from retaining the whole backtest.
                "x": deque(maxlen=2),
                "y": deque(maxlen=2),
                "x_key": (str(x_exchange), str(x_symbol)),
                "y_key": (str(y_exchange), str(y_symbol)),
                "x_exchange": str(x_exchange),
                "x_symbol": str(x_symbol),
                "y_exchange": str(y_exchange),
                "y_symbol": str(y_symbol),
            }
            self._pairs_by_leg[(str(x_exchange), str(x_symbol))].append(pair_id)
            self._pairs_by_leg[(str(y_exchange), str(y_symbol))].append(pair_id)

    def update(self, bars_by_exchange: dict) -> list[dict]:
        """Consume one raw timestamp and return ready pair model bars."""
        if self.interval_minutes == 1:
            return self._direct_pair_events(bars_by_exchange)
        finalized: list[tuple[str, str, list]] = []
        for exchange, symbols in (bars_by_exchange or {}).items():
            for symbol, raw_bar in (symbols or {}).items():
                bar = self._normalize_bar(raw_bar)
                ts = int(bar[0])
                bucket_start = (ts // self.interval_ms) * self.interval_ms
                key = (str(exchange), str(symbol))
                bucket = self._buckets.get(key)
                if bucket is None:
                    self._buckets[key] = self._new_bucket(bucket_start, bar)
                    continue
                if bucket_start < bucket.start_ts:
                    raise ValueError(f"bars must be sorted ascending for {exchange}.{symbol}")
                if bucket_start == bucket.start_ts:
                    bucket.append(bar)
                    if bucket.complete(self.interval_minutes):
                        finalized.append((key[0], key[1], bucket.bar()))
                        self._buckets.pop(key, None)
                    continue
                if bucket.complete(self.interval_minutes):
                    finalized.append((key[0], key[1], bucket.bar()))
                else:
                    self._record_discarded_bucket(key, bucket)
                self._buckets[key] = self._new_bucket(bucket_start, bar)

        ready_by_ts: dict[int, dict] = {}
        for exchange, symbol, bar in finalized:
            self._offer_completed(exchange, symbol, bar, ready_by_ts)
        return [ready_by_ts[ts] for ts in sorted(ready_by_ts)]

    def _direct_pair_events(self, bars_by_exchange: dict) -> list[dict]:
        events = {}
        for pair_id, x_exchange, x_symbol, y_exchange, y_symbol in self.pair_specs:
            x_bar = (bars_by_exchange.get(x_exchange, {}) or {}).get(x_symbol)
            y_bar = (bars_by_exchange.get(y_exchange, {}) or {}).get(y_symbol)
            if x_bar is None or y_bar is None:
                continue
            # Keep the 1m fast path compatible with the aggregated path: both
            # accept either canonical lists or mapping-style bars.
            x_bar = self._normalize_bar(x_bar)
            y_bar = self._normalize_bar(y_bar)
            ts = int(x_bar[0])
            if int(y_bar[0]) != ts:
                continue
            events.setdefault(str(x_exchange), {})[str(x_symbol)] = x_bar
            events.setdefault(str(y_exchange), {})[str(y_symbol)] = y_bar
        return [events] if events else []

    def _record_discarded_bucket(self, key: tuple[str, str], bucket: _Bucket) -> None:
        self.discarded_bucket_count += 1
        if len(self.discarded_bucket_examples) < 10:
            self.discarded_bucket_examples.append(
                {
                    "exchange": key[0],
                    "symbol": key[1],
                    "start_ts": bucket.start_ts,
                    "last_ts": bucket.last_ts,
                    "count": bucket.count,
                    "expected": self.interval_minutes,
                    "contiguous": bucket.contiguous,
                }
            )

    def _offer_completed(self, exchange: str, symbol: str, bar: list, ready_by_ts: dict[int, dict]) -> None:
        """Offer one completed symbol bar to only the Pairs that use it."""
        leg_key = (str(exchange), str(symbol))
        for pair_id in self._pairs_by_leg.get(leg_key, ()):
            state = self._pair_queues[pair_id]
            leg = "x" if state["x_key"] == leg_key else "y"
            state[leg].append((int(bar[0]), bar))
            x_queue = state["x"]
            y_queue = state["y"]
            while x_queue and y_queue:
                x_ts, x_bar = x_queue[0]
                y_ts, y_bar = y_queue[0]
                if x_ts == y_ts:
                    x_queue.popleft()
                    y_queue.popleft()
                    event = ready_by_ts.setdefault(x_ts, {})
                    event.setdefault(state["x_exchange"], {})[state["x_symbol"]] = x_bar
                    event.setdefault(state["y_exchange"], {})[state["y_symbol"]] = y_bar
                elif x_ts < y_ts:
                    # The other leg has advanced past x_ts; no future bar can
                    # match the older x bar because source bars are ordered.
                    x_queue.popleft()
                else:
                    y_queue.popleft()

    @staticmethod
    def _new_bucket(start_ts: int, bar: list) -> _Bucket:
        return _Bucket(
            start_ts=start_ts,
            last_ts=int(bar[0]),
            count=1,
            open=float(bar[1]),
            high=float(bar[2]),
            close=float(bar[3]),
            low=float(bar[4]),
            volume=float(bar[5]),
        )

    @staticmethod
    def _normalize_bar(bar) -> list:
        if isinstance(bar, dict):
            return [int(bar["ts"]), float(bar["open"]), float(bar["high"]), float(bar["close"]), float(bar["low"]), float(bar["volume"])]
        if not isinstance(bar, list) or len(bar) < 6:
            raise ValueError("raw bar must contain [ts, open, high, close, low, volume]")
        return [int(bar[0]), *[float(value) for value in bar[1:6]]]
