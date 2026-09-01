from types import SimpleNamespace
from pathlib import Path

from core.backtest.backtest import Backtest
from core.backtest.sweep_backtest import SweepBacktest
from core.modules.data.funding import load_funding_data
from core.modules.data.stream import iter_csv_bars
from core.modules.config import load_config


def test_funding_loader_preserves_one_four_and_eight_hour_events(tmp_path):
    path = tmp_path / "funding.csv"
    path.write_text(
        "symbol,fundingTime,fundingRate\n"
        "TEST,1735689600015,0.0001\n"
        "TEST,1735693200001,0.0002\n"
        "TEST,1735704000020,0.0003\n"
        "TEST,1735718400000,0.0004\n",
        encoding="utf-8",
    )

    frame = load_funding_data(path)

    assert frame["ts"].to_list() == [
        1735689600000,
        1735693200000,
        1735704000000,
        1735718400000,
    ]
    assert frame["funding_rate"].to_list() == [0.0001, 0.0002, 0.0003, 0.0004]


def test_funding_events_before_backtest_start_are_skipped():
    backtest = Backtest.__new__(Backtest)
    funding_data = {
        1000: {"A": 0.01},
        2000: {"A": 0.02},
        3000: {"A": 0.03},
    }

    events, index = backtest._funding_events_for_bar(
        funding_data,
        sorted(funding_data),
        funding_index=0,
        previous_ts=None,
        ts=2000,
    )

    assert events == [{"ts": 2000, "rates": {"A": 0.02}}]
    assert index == 2


def test_funding_waits_for_the_symbols_next_available_bar():
    backtest = Backtest.__new__(Backtest)
    events = [{"ts": 1000, "rates": {"A": 0.01, "B": 0.02}}]

    payable, pending = backtest._available_funding_events(
        [],
        events,
        {"binance": {"A": [1000, 1, 1, 1, 1, 1]}},
    )

    assert payable == [{"ts": 1000, "rates": {"A": 0.01}}]
    assert pending == [{"ts": 1000, "symbol": "B", "rate": 0.02}]

    payable, pending = backtest._available_funding_events(
        pending,
        [],
        {"binance": {"B": [2000, 1, 1, 1, 1, 1]}},
    )

    assert payable == [{"ts": 1000, "rates": {"B": 0.02}}]
    assert pending == []


def test_multiple_deferred_events_are_not_merged_or_dropped():
    backtest = Backtest.__new__(Backtest)
    pending = [
        {"ts": 1000, "symbol": "A", "rate": 0.01},
        {"ts": 2000, "symbol": "A", "rate": 0.02},
    ]

    payable, remaining = backtest._available_funding_events(
        pending,
        [],
        {"binance": {"A": [3000, 1, 1, 1, 1, 1]}},
    )

    assert payable == [
        {"ts": 1000, "rates": {"A": 0.01}},
        {"ts": 2000, "rates": {"A": 0.02}},
    ]
    assert remaining == []


def test_one_hour_funding_events_follow_actual_timestamps():
    backtest = Backtest.__new__(Backtest)
    hour = 60 * 60 * 1000
    funding_data = {
        hour: {"A": 0.01},
        2 * hour: {"A": 0.02},
        3 * hour: {"A": 0.03},
    }

    events, index = backtest._funding_events_for_bar(
        funding_data,
        sorted(funding_data),
        funding_index=0,
        previous_ts=0,
        ts=3 * hour,
    )

    assert events == [
        {"ts": hour, "rates": {"A": 0.01}},
        {"ts": 2 * hour, "rates": {"A": 0.02}},
        {"ts": 3 * hour, "rates": {"A": 0.03}},
    ]
    assert index == 3


def test_funding_alignment_audit_does_not_materialize_market_data(monkeypatch):
    backtest = Backtest.__new__(Backtest)
    backtest.config = SimpleNamespace(funding_enabled=True)
    monkeypatch.setattr(
        "core.backtest.backtest.load_csv_data",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("market data must remain streaming")
        ),
    )

    backtest._check_funding_alignment(
        {1000: {"A": 0.01}},
        {"A": {"exchange": "binance", "path": "unused.csv"}},
    )


def test_sweep_backtest_keeps_market_paths_streaming():
    backtest = SweepBacktest.__new__(SweepBacktest)
    backtest.config = SimpleNamespace(
        strategy=SimpleNamespace(
            pairs=[{"x_symbol": "A", "y_symbol": "B", "enabled": True}]
        ),
        symbols={
            "A": {"exchange": "binance", "path": "a.csv"},
            "B": {"exchange": "binance", "path": "b.csv"},
        },
    )

    market_data = backtest._load_market_data()

    assert market_data == {
        "A": {"exchange": "binance", "path": Path("a.csv")},
        "B": {"exchange": "binance", "path": Path("b.csv")},
    }
    assert all("frame" not in item for item in market_data.values())


def test_stream_encoding_detection_does_not_read_the_whole_file(tmp_path, monkeypatch):
    path = tmp_path / "bars.csv"
    path.write_text(
        "ts,open,high,close,low,volume\n"
        "1735689600000,1,2,1.5,0.5,10\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda _self: (_ for _ in ()).throw(
            AssertionError("streaming encoding detection must not read the whole file")
        ),
    )

    assert list(iter_csv_bars(path)) == [
        [1735689600000, 1.0, 2.0, 1.5, 0.5, 10.0]
    ]


def test_optional_backtest_end_time_limits_streaming_bars(tmp_path):
    path = tmp_path / "bars.csv"
    path.write_text(
        "ts,open,high,close,low,volume\n"
        "1000,1,2,1.5,0.5,10\n"
        "2000,2,3,2.5,1.5,11\n"
        "3000,3,4,3.5,2.5,12\n",
        encoding="utf-8",
    )

    backtest = Backtest.__new__(Backtest)
    backtest.config = SimpleNamespace(backtest_start_ts=2_000_000, backtest_end_ts=2_000_000)
    item = {"path": path}

    assert list(backtest._symbol_bar_iterator(item)) == [
        [2_000_000, 2.0, 3.0, 2.5, 1.5, 11.0]
    ]


def test_load_config_parses_optional_backtest_end_time(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
data:
  symbols:
    A:
      exchange: binance
      path: a.csv
    B:
      exchange: binance
      path: b.csv
  benchmarks: {}
backtest:
  initial_cash: 100000
  start_time: '2025-01-01'
  end_time: '2025-01-31'
active_setup: setup
setups:
  setup:
    pairs:
      - pair_id: p
        x_symbol: A
        y_symbol: B
    pipeline:
      estimator: {method: rolling_ols}
      signal: {method: zscore}
      sizing: {method: beta_neutral}
      portfolio: {method: equal_weight}
      execution: {order_type: market}
cost:
  fee_rate: 0.0005
  slippage_bps: 1
  funding_enabled: false
risk: {}
""",
        encoding="utf-8",
    )

    config = load_config(path)
    assert config.backtest_end_ts == 1738281600000
