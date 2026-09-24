import pytest

from core.modules.data.resampler import PairTimeframeResampler, timeframe_bars, timeframe_to_minutes


def _bar(ts, price):
    return [ts, price, price + 2.0, price + 1.0, price - 1.0, 3.0]


def _bars_for(ts0, count, price0=100.0):
    return [_bar(ts0 + i * 60_000, price0 + i) for i in range(count)]


def _feed(resampler, bars_x, bars_y, *, x_symbol="X", y_symbol="Y"):
    events = []
    for x_bar, y_bar in zip(bars_x, bars_y):
        result = resampler.update({"binance": {x_symbol: x_bar, y_symbol: y_bar}})
        events.extend(result)
    return events


def test_timeframe_to_minutes_and_bar_conversion():
    assert timeframe_to_minutes("1m") == 1
    assert timeframe_to_minutes("15m") == 15
    assert timeframe_to_minutes("1h") == 60
    assert timeframe_to_minutes("2h") == 120
    assert timeframe_to_minutes("4h") == 240
    assert timeframe_bars(2880, 15, label="lookback") == 192
    assert timeframe_bars(240, 15, label="update") == 16


def test_timeframe_rejects_invalid_values():
    with pytest.raises(ValueError):
        timeframe_to_minutes("bad")
    with pytest.raises(ValueError):
        timeframe_to_minutes("7.5m")


def test_15m_aggregation_uses_ohlcv_and_requires_complete_pair():
    resampler = PairTimeframeResampler(
        "15m", [("pair", "binance", "X", "binance", "Y")]
    )
    ts0 = 0
    x = _bars_for(ts0, 15, 100.0)
    y = _bars_for(ts0, 15, 200.0)
    events = _feed(resampler, x, y)

    assert len(events) == 1
    event = events[0]["binance"]
    x_agg = event["X"]
    assert x_agg == [
        14 * 60_000,
        100.0,
        116.0,
        115.0,
        99.0,
        45.0,
    ]


def test_incomplete_or_missing_minute_bar_does_not_emit_model_bar():
    resampler = PairTimeframeResampler(
        "15m", [("pair", "binance", "X", "binance", "Y")]
    )
    ts0 = 0
    x = _bars_for(ts0, 15, 100.0)
    y = _bars_for(ts0, 15, 200.0)
    # Remove one minute from Y; the pair must not be emitted.
    events = []
    for i, x_bar in enumerate(x):
        bars = {"binance": {"X": x_bar}}
        if i != 7:
            bars["binance"]["Y"] = y[i]
        events.extend(resampler.update(bars))
    assert events == []
    # Advance into the next bucket so the incomplete one is finalized and
    # recorded as a discarded source bucket.
    next_ts = 15 * 60_000
    resampler.update({"binance": {"X": _bar(next_ts, 115.0), "Y": _bar(next_ts, 215.0)}})
    assert resampler.discarded_bucket_count == 1
    assert resampler.discarded_bucket_examples[0]["symbol"] == "Y"

    # A partial bucket is also not emitted.
    resampler = PairTimeframeResampler(
        "15m", [("pair", "binance", "X", "binance", "Y")]
    )
    assert _feed(resampler, x[:14], y[:14]) == []


def test_1m_direct_path_keeps_pair_alignment_and_raw_values():
    resampler = PairTimeframeResampler(
        "1m", [("pair", "binance", "X", "binance", "Y")]
    )
    x = _bar(60_000, 100.0)
    y = _bar(60_000, 200.0)
    events = resampler.update({"binance": {"X": x, "Y": y}})
    assert events == [{"binance": {"X": x, "Y": y}}]

    # Mapping-style bars are accepted by the 1m fast path as well.
    x_dict = {"ts": 240_000, "open": 100.0, "high": 102.0, "close": 101.0, "low": 99.0, "volume": 3.0}
    y_dict = {"ts": 240_000, "open": 200.0, "high": 202.0, "close": 201.0, "low": 199.0, "volume": 3.0}
    events = resampler.update({"binance": {"X": x_dict, "Y": y_dict}})
    assert events == [{"binance": {"X": [240_000, 100.0, 102.0, 101.0, 99.0, 3.0],
                                     "Y": [240_000, 200.0, 202.0, 201.0, 199.0, 3.0]}}]

    # Mismatched timestamps are not passed to the model.
    assert resampler.update({"binance": {"X": _bar(120_000, 101.0), "Y": _bar(180_000, 201.0)}}) == []

def test_pair_queues_align_when_legs_finish_in_different_updates():
    resampler = PairTimeframeResampler(
        "15m", [("pair", "binance", "X", "binance", "Y")]
    )
    ts0 = 0
    x = _bars_for(ts0, 30, 100.0)
    y = _bars_for(ts0, 30, 200.0)
    # X has the first model bar available before Y catches up.
    for bar in x[:15]:
        assert resampler.update({"binance": {"X": bar}}) == []
    events = []
    for i in range(15):
        bars = {"binance": {"Y": y[i]}}
        if i == 14:
            bars["binance"]["X"] = x[15]
        events.extend(resampler.update(bars))
    assert len(events) == 1
    assert events[0]["binance"]["X"][0] == 14 * 60_000


def test_pipeline_converts_legacy_source_minute_windows_once():
    from types import SimpleNamespace
    from core.modules.strategy.factory import build_strategy

    setup = SimpleNamespace(
        pairs=[{
            "pair_id": "p", "x_symbol": "X", "y_symbol": "Y",
            "target_capital": 600000.0,
        }],
        pipeline={
            "estimator": {
                "method": "tls", "regression_method": "log_price",
                "model_timeframe": "15m",
                "model_lookback_bars": 2880,
                "model_update_interval_bars": 240,
            },
            "signal": {
                "method": "zscore_reversion_ma",
                "reversion_ma_lookback_bars": 60,
                "reversion_ma_short_lookback_bars": 15,
            },
            "sizing": {"method": "beta_neutral"},
            "portfolio": {
                "method": "pair_target_capital",
                "add_cooldown_bars": 30,
                "min_hold_bars": 45,
            },
            "protection": {
                "enabled": True,
                "pair_loss_stop_enabled": True,
                "pair_loss_stop_freeze_bars": 1440,
            },
            "rebalance": {
                "eviction_min_holding_model_lookback_multiplier": 0.5
            },
        },
    )
    strategy = build_strategy(setup, {"X": {"exchange": "binance"}, "Y": {"exchange": "binance"}})
    pipeline = strategy.pipelines["p"]
    assert pipeline.estimator.model_lookback_bars == 192
    assert pipeline.estimator.model_update_interval_bars == 16
    # MA signal keeps its existing minimum long window of 5 model bars.
    assert pipeline.signal.reversion_ma_lookback_bars == 5
    assert pipeline.signal.reversion_ma_short_lookback_bars == 1
    assert pipeline.signal.pair_quality_min_samples == 2
    assert strategy.add_cooldown_bars == 2
    assert strategy.min_hold_bars == 3
    assert strategy.protection_cfg.pair_loss_stop_freeze_bars == 96
    assert strategy.rebalance_cfg.eviction_min_holding_model_lookback_multiplier == 0.5
    assert strategy._rebalance_min_holding_bars(pipeline) == 96


@pytest.mark.parametrize("location", ["setup", "pipeline"])
def test_factory_rejects_legacy_rebalance_holding_key_at_either_config_layer(location):
    from types import SimpleNamespace
    from core.modules.strategy.factory import build_strategy

    setup = SimpleNamespace(
        pairs=[{"pair_id": "p", "x_symbol": "X", "y_symbol": "Y"}],
        pipeline={
            "estimator": {"method": "tls", "regression_method": "price", "model_lookback_bars": 10},
            "signal": {"method": "zscore"},
            "sizing": {"method": "beta_neutral"},
            "portfolio": {"method": "equal_weight"},
            "execution": {},
        },
    )
    if location == "setup":
        setup.rebalance = {"eviction_min_holding_bars": 60}
    else:
        setup.pipeline["rebalance"] = {"eviction_min_holding_bars": 60}

    with pytest.raises(ValueError, match="eviction_min_holding_bars is no longer supported"):
        build_strategy(setup, {"X": {}, "Y": {}})


def test_factory_rejects_legacy_rebalance_key_even_when_both_layers_exist():
    from types import SimpleNamespace
    from core.modules.strategy.factory import build_strategy

    setup = SimpleNamespace(
        pairs=[{"pair_id": "p", "x_symbol": "X", "y_symbol": "Y"}],
        rebalance={"enabled": True},
        pipeline={
            "estimator": {"method": "tls", "regression_method": "price", "model_lookback_bars": 10},
            "signal": {"method": "zscore"},
            "sizing": {"method": "beta_neutral"},
            "portfolio": {"method": "equal_weight"},
            "execution": {},
            "rebalance": {"eviction_min_holding_bars": 60},
        },
    )

    with pytest.raises(ValueError, match="pipeline.rebalance.eviction_min_holding_bars"):
        build_strategy(setup, {"X": {}, "Y": {}})


def test_one_sided_stream_does_not_retain_unbounded_completed_bars():
    resampler = PairTimeframeResampler(
        "15m", [("pair", "binance", "X", "binance", "Y")]
    )
    for bar in _bars_for(0, 15 * 20, 100.0):
        assert resampler.update({"binance": {"X": bar}}) == []
    assert len(resampler._pair_queues["pair"]["x"]) <= 2
    assert len(resampler._pair_queues["pair"]["y"]) == 0
