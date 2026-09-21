"""Probes for the 2026-09-21 model-timeframe change (1m -> 15m/1h/2h/4h aggregation).

Read-only: builds in-memory fakes plus direct resampler calls; no backtest is run.
Run from the repo root:  python research/scripts/review_0921_probe.py
"""
from __future__ import annotations

import sys
from math import ceil
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.modules.data.resampler import (  # noqa: E402
    PairTimeframeResampler,
    timeframe_bars,
    timeframe_to_minutes,
)
from core.modules.models.pipeline_types import (  # noqa: E402
    PortfolioState,
    PositionProtectionState,
    RawPairTarget,
    SizingState,
)
from core.modules.strategy.config import (  # noqa: E402
    ProfitablePositionReplacementConfig,
    RebalanceConfig,
)
from core.modules.strategy.multi_pair_strategy import (  # noqa: E402
    MultiPairStrategy,
    _decision_bars,
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        FAILURES.append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# --------------------------------------------------------------- conversions
def probe_conversions() -> None:
    print("\n== A. source-minute -> model-bar conversion ==")
    check("A1 timeframe parsing", [timeframe_to_minutes(x) for x in
          ("1m", "15m", "1h", "2h", "4h", "90", "0.5h")] == [1, 15, 60, 120, 240, 90, 30])
    check("A2 2880 minutes @15m -> 192 bars", timeframe_bars(2880, 15, label="l") == 192)
    check("A3 240 minutes @15m -> 16 bars", timeframe_bars(240, 15, label="l") == 16)
    check("A4 ceil keeps the window at least as long as configured",
          timeframe_bars(2881, 15, label="l") == 193 and 193 * 15 >= 2881,
          f"{timeframe_bars(2881, 15, label='l')} bars")
    check("A5 a 1-minute window at 15m never becomes 0 bars",
          timeframe_bars(1, 15, label="l") == 1)
    check("A6 _decision_bars keeps 0 meaning disabled",
          _decision_bars(0, 15, "x") == 0)
    check("A7 _decision_bars converts 1440 minutes @15m to 96",
          _decision_bars(1440, 15, "x") == 96)
    check("A8 _decision_bars converts 60 minutes @15m to 4",
          _decision_bars(60, 15, "x") == 4)
    try:
        _decision_bars(-5, 15, "x")
        check("A9 negative values raise", False)
    except ValueError:
        check("A9 negative values raise", True)
    check("A10 1m timeframe is the identity",
          timeframe_bars(2880, 1, label="l") == 2880 and _decision_bars(60, 1, "x") == 60)


# ----------------------------------------------------------------- resampler
def _bar(ts: int, price: float) -> list:
    return [ts, price, price + 2.0, price + 1.0, price - 1.0, 3.0]


def probe_resampler_edges() -> None:
    print("\n== B. resampler edge cases ==")
    spec = [("pair", "binance", "X", "binance", "Y")]

    # B1 duplicate minute kills only that bucket
    r = PairTimeframeResampler("15m", spec)
    import random
    ts0 = 0
    events = []
    for i in range(30):
        minute = ts0 + i * 60_000
        events.extend(r.update({"binance": {"X": _bar(minute, 100 + i), "Y": _bar(minute, 200 + i)}}))
        if i == 7:  # duplicate the 8th minute
            events.extend(r.update({"binance": {"X": _bar(minute, 100 + i), "Y": _bar(minute, 200 + i)}}))
    check("B1 a duplicate minute drops only its own 15m bucket",
          [e["binance"]["X"][0] for e in events] == [29 * 60_000],
          f"emitted ts={[e['binance']['X'][0] for e in events]}")

    # B2 one-sided outage: alignment never mis-pairs, and late recovery still matches
    r = PairTimeframeResampler("15m", spec)
    events = []
    for i in range(90):
        minute = i * 60_000
        bars = {"binance": {"X": _bar(minute, 100 + i)}}
        if 30 <= i < 75:  # Y silent for 45 minutes (3 buckets)
            pass
        else:
            bars["binance"]["Y"] = _bar(minute, 200 + i)
        events.extend(r.update(bars))
    emitted = [(e["binance"]["X"], e["binance"]["Y"]) for e in events]
    # _bar(ts, p) = [ts, p, p+2, p+1, p-1, 3]; X prices 100+i, Y prices 200+i.
    # A bucket ending at minute `end` spans [end-14, end]: open from the first
    # minute, close from the last, high/low over the span, volume 15*3.
    def bucket_ok(x_bar, y_bar) -> bool:
        end = x_bar[0] // 60_000
        start = end - 14
        return (
            x_bar[1] == 100 + start and x_bar[3] == 100 + end + 1
            and x_bar[2] == 100 + end + 2 and x_bar[4] == 100 + start - 1
            and abs(x_bar[5] - 45.0) < 1e-9
            and y_bar[1] == 200 + start and y_bar[3] == 200 + end + 1
            and y_bar[0] == x_bar[0]
        )

    check("B2 after a one-sided outage the legs still pair by exact timestamp",
          [x[0] // 60_000 for x, _y in emitted] == [14, 29, 89]
          and all(bucket_ok(x, y) for x, y in emitted),
          f"emitted minutes={[x[0] // 60_000 for x, _y in emitted]}")

    # B3 queue bound
    r = PairTimeframeResampler("15m", spec)
    for i in range(15 * 50):
        r.update({"binance": {"X": _bar(i * 60_000, 100.0)}})
    check("B3 one-sided stream keeps at most 2 completed bars",
          len(r._pair_queues["pair"]["x"]) <= 2, f"{len(r._pair_queues['pair']['x'])}")

    # B4 out-of-order raw bars raise instead of silently mis-aggregating
    r = PairTimeframeResampler("15m", spec)
    r.update({"binance": {"X": _bar(20 * 60_000, 1.0), "Y": _bar(20 * 60_000, 1.0)}})
    try:
        r.update({"binance": {"X": _bar(5 * 60_000, 1.0), "Y": _bar(5 * 60_000, 1.0)}})
        check("B4 descending raw bars raise", False)
    except ValueError:
        check("B4 descending raw bars raise", True)

    # B5 1m direct path shape + mismatched ts
    r = PairTimeframeResampler("1m", spec)
    x = _bar(60_000, 100.0)
    y = _bar(60_000, 200.0)
    check("B5 1m path passes raw bars through unchanged",
          r.update({"binance": {"X": x, "Y": y}}) == [{"binance": {"X": x, "Y": y}}])
    check("B6 1m path drops a pair whose legs are not on the same minute",
          r.update({"binance": {"X": _bar(120_000, 1.0), "Y": _bar(180_000, 1.0)}}) == [])
    check("B7 dict bars reach the aggregated path",
          PairTimeframeResampler("15m", spec).update(
              {"binance": {"X": {"ts": 0, "open": 1, "high": 2, "close": 2, "low": 1, "volume": 3}}}) == [])
    try:
        PairTimeframeResampler("1m", spec).update(
            {"binance": {"X": {"ts": 0, "open": 1, "high": 2, "close": 2, "low": 1, "volume": 3},
                         "Y": {"ts": 0, "open": 1, "high": 2, "close": 2, "low": 1, "volume": 3}}})
        check("B8 1m path accepts dict bars too", True)
    except Exception as exc:  # noqa: BLE001
        check("B8 1m path accepts dict bars too", False, f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------- eviction bar counter
def _pipeline(pid: str, side: str | None = "long_x", gross: float = 600_000.0,
              x_qty: float = 1000.0, y_qty: float = 1000.0,
              entry_bar: int | None = 1):
    sizing = SizingState(position_side=side, position_id=pid, x_quantity=x_qty,
                         y_quantity=y_qty, current_gross_notional=gross,
                         target_hedge_ratio=1.0, entry_count=1 if side else 0)
    protection = PositionProtectionState(
        active=side is not None, side=side, entry_x_price=100.0, entry_y_price=50.0,
        entry_gross_notional=gross, entry_x_quantity=x_qty, entry_y_quantity=y_qty,
        entry_bar_index=entry_bar,
    )
    return SimpleNamespace(
        pair_def=SimpleNamespace(pair_id=pid, x_symbol=f"{pid}X", y_symbol=f"{pid}Y",
                                 target_capital=gross),
        state=SimpleNamespace(sizing_state=sizing, protection_state=protection, last_bar_index=1000),
        estimator=SimpleNamespace(model_lookback_bars=192),
    )


def _candidate(pid: str = "new", target: float = 600_000.0) -> dict:
    raw = RawPairTarget(pair_id=pid, ready=True, side="long_x", x_weight=0.5, y_weight=0.5,
                        x_price=100.0, y_price=50.0, gross_notional=target, target_capital=target)
    raw.para = {"protection": {"theoretical_zero_return_x_move": 0.25}}
    return {"pair_id": pid, "raw_target": raw,
            "result": SimpleNamespace(estimator=SimpleNamespace(cointegration_pvalue=0.3))}


def _strategy(available: float, eviction_bars: int):
    s = MultiPairStrategy.__new__(MultiPairStrategy)
    s.rebalance_cfg = RebalanceConfig(
        enabled=True, minimum_entry_capital_ratio=0.5,
        closed_pair_freeze_model_lookback_multiplier=0.5,
        eviction_min_holding_bars=eviction_bars,
        profitable_position_replacement=ProfitablePositionReplacementConfig(
            enabled=True, min_theoretical_zero_return_x_move=0.20, max_adf_pvalue=0.4),
    )
    s.estimator_cfg = SimpleNamespace(model_lookback_bars=2880)
    s.fee_rate = 0.0005
    s.slippage_rate = 0.0001
    s.portfolio_state = PortfolioState(ready=True, equity=3_000_000, available_balance=available)
    s.pipelines = {}
    s._pending_rebalance_pair_ids = set()
    s._pending_rebalance_release = 0.0
    s.closed_log = []

    def fake_close(pair_id, pipeline, pair_def, fallback_hedge, **kwargs):
        s.closed_log.append(pair_id)
        return [SimpleNamespace(pair_id=pair_id)]

    s._close_orders = fake_close
    return s


def bundle_with_index(pair_id: str, bar_index: int, x_close: float = 90.0, y_close: float = 50.0):
    return SimpleNamespace(
        pair_id=pair_id,
        bar_index=bar_index,
        x_bar=SimpleNamespace(close=x_close),
        y_bar=SimpleNamespace(close=y_close),
    )


def probe_eviction_counter() -> None:
    print("\n== C. eviction_min_holding_bars counter units ==")
    guard = 96

    # The guard must use the Pair-local model-bar index, never the global one.
    s = _strategy(available=100_000.0, eviction_bars=guard)
    s.pipelines = {"new": _pipeline("new", side=None, gross=0.0, x_qty=0.0, y_qty=0.0,
                                    entry_bar=None),
                   "old": _pipeline("old", entry_bar=5)}
    s._plan_rebalance({"new": _candidate()},
                      {"old": bundle_with_index("old", bar_index=5)}, bar_index=1000)
    check("C1 fresh Pair-local index is not evicted even though the global index is 1000",
          s.closed_log == [], f"closed={s.closed_log}")

    s3 = _strategy(available=100_000.0, eviction_bars=guard)
    s3.pipelines = {"new": _pipeline("new", side=None, gross=0.0, x_qty=0.0, y_qty=0.0,
                                     entry_bar=None),
                    "old": _pipeline("old", entry_bar=5)}
    s3._plan_rebalance({"new": _candidate()},
                       {"old": bundle_with_index("old", bar_index=5 + guard - 1)}, bar_index=1000)
    check("C2 one bar short of the threshold is still blocked",
          s3.closed_log == [], f"closed={s3.closed_log}")

    s2 = _strategy(available=100_000.0, eviction_bars=guard)
    s2.pipelines = {"new": _pipeline("new", side=None, gross=0.0, x_qty=0.0, y_qty=0.0,
                                     entry_bar=None),
                    "old": _pipeline("old", entry_bar=5)}
    s2._plan_rebalance({"new": _candidate()},
                       {"old": bundle_with_index("old", bar_index=5 + guard)}, bar_index=1000)
    check("C3 a Pair held for the full guard is evictable",
          s2.closed_log == ["old"], f"closed={s2.closed_log}")

    s4 = _strategy(available=100_000.0, eviction_bars=guard)
    s4.pipelines = {"new": _pipeline("new", side=None, gross=0.0, x_qty=0.0, y_qty=0.0,
                                     entry_bar=None),
                    "old": _pipeline("old", entry_bar=None)}
    s4._plan_rebalance({"new": _candidate()},
                       {"old": bundle_with_index("old", bar_index=500)}, bar_index=1000)
    check("C4 unknown entry bar skips the guard (evictable) - documented behaviour",
          s4.closed_log == ["old"], f"closed={s4.closed_log}")


def probe_effective_windows() -> None:
    """What the real config's bar-valued settings become at each model timeframe."""
    print("\n== D. effective windows for the real config (source minutes -> model bars) ==")
    import copy
    from types import SimpleNamespace

    from core.modules.config import load_config
    from core.modules.strategy.factory import build_strategy

    config = load_config(ROOT / "config" / "config.yaml")
    header = f"{'setting':38s} {'src min':>8s} " + " ".join(f"{tf:>13s}" for tf in
                                                             ("1m", "15m", "1h", "2h", "4h"))
    print(header)
    rows: dict[str, list[str]] = {}
    for timeframe in ("1m", "15m", "1h", "2h", "4h"):
        setup = SimpleNamespace(
            pairs=config.strategy.pairs,
            pipeline=copy.deepcopy(config.strategy.pipeline),
            rebalance=copy.deepcopy(getattr(config.strategy, "rebalance", {}) or {}),
        )
        setup.pipeline.setdefault("estimator", {})["model_timeframe"] = timeframe
        strategy = build_strategy(setup, config.symbols,
                                  fee_rate=config.fee_rate, slippage_bps=config.slippage_bps)
        pipe = next(iter(strategy.pipelines.values()))
        values = {
            "estimator.model_lookback_bars": pipe.estimator.model_lookback_bars,
            "estimator.model_update_interval_bars": pipe.estimator.model_update_interval_bars,
            "signal.reversion_ma_lookback_bars": pipe.signal.reversion_ma_lookback_bars,
            "signal.reversion_ma_short_lookback_bars": pipe.signal.reversion_ma_short_lookback_bars,
            "signal.reversion_min_samples": pipe.signal.reversion_min_samples,
            "sizing.hedge_beta_lookback_bars": pipe.sizing.hedge_beta_lookback_bars,
            "sizing.hedge_beta_min_samples": pipe.sizing.hedge_beta_min_samples,
            "portfolio.add_cooldown_bars": strategy.add_cooldown_bars,
            "rebalance.eviction_min_holding_bars": strategy.rebalance_cfg.eviction_min_holding_bars,
        }
        for key, value in values.items():
            minutes = value * timeframe_to_minutes(timeframe)
            rows.setdefault(key, []).append(f"{value:>5d}({minutes:>5d}m)")
    source = {
        "estimator.model_lookback_bars": 2880,
        "estimator.model_update_interval_bars": 240,
        "signal.reversion_ma_lookback_bars": 120,
        "signal.reversion_ma_short_lookback_bars": 30,
        "signal.reversion_min_samples": 120,
        "sizing.hedge_beta_lookback_bars": 1440,
        "sizing.hedge_beta_min_samples": 1440,
        "portfolio.add_cooldown_bars": 720,
        "rebalance.eviction_min_holding_bars": 60,
    }
    for key, cells in rows.items():
        print(f"{key:38s} {source[key]:>8d} " + " ".join(f"{c:>13s}" for c in cells))

    m1 = rows["signal.reversion_ma_lookback_bars"]
    check("D1 120-minute MA window converts exactly at 15m",
          m1[1].startswith("    8") and "(  120m)" in m1[1], m1[1].strip())
    check("D2 at 4h the 120-minute MA window becomes 5 bars = 1200 minutes (10x)",
          m1[4].startswith("    5") and "( 1200m)" in m1[4], m1[4].strip())
    check("D3 the estimator lookback tracks the source minutes exactly at every timeframe",
          all(cell.endswith("( 2880m)") for cell in rows["estimator.model_lookback_bars"]),
          " ".join(c.split("(")[1].rstrip(")") for c in rows["estimator.model_lookback_bars"]))


def main() -> int:
    probe_conversions()
    probe_resampler_edges()
    probe_eviction_counter()
    probe_effective_windows()
    print("\n==================== SUMMARY ====================")
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("all probes passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
