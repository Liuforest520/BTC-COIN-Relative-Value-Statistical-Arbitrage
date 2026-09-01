from pathlib import Path
from types import SimpleNamespace

import polars as pl

from core.backtest.backtest import Backtest
from core.modules.data.stream import iter_csv_bars
from core.modules.config import Config, SetupConfig
from core.modules.exchange import Exchange, ExchangeManager
from core.modules.models import Order, OrderAction, OrderSide, OrderType


def _frame(timestamps):
    return pl.DataFrame(
        {
            "ts": timestamps,
            "open": [float(ts) for ts in timestamps],
            "high": [float(ts) + 1.0 for ts in timestamps],
            "close": [float(ts) + 0.5 for ts in timestamps],
            "low": [float(ts) - 1.0 for ts in timestamps],
            "volume": [100.0 for _ in timestamps],
        }
    )


def _config():
    return Config(
        symbols={
            "BTCUSDT": {"exchange": "binance", "path": ""},
            "COINUSDT": {"exchange": "binance", "path": ""},
            "MSTRUSDT": {"exchange": "binance", "path": ""},
        },
        benchmarks={},
        initial_cash=100000.0,
        backtest_start_time=None,
        backtest_start_ts=None,
        active_setup="test",
        strategy=SetupConfig(
            name="test",
            strategy_type="multi_pair",
            pairs=[
                {"pair_id": "btc_coin", "x_symbol": "BTCUSDT", "y_symbol": "COINUSDT"},
                {"pair_id": "btc_mstr", "x_symbol": "BTCUSDT", "y_symbol": "MSTRUSDT"},
            ],
            pipeline={},
        ),
        fee_rate=0.0,
        slippage_bps=0.0,
        funding_enabled=False,
        risk={},
    )


def test_streaming_multi_pair_data_alignment_is_pair_inner_portfolio_outer():
    backtest = Backtest(
        _config(),
        strategy=SimpleNamespace(),
        risk_manager=SimpleNamespace(),
        exchange_manager=SimpleNamespace(),
        show_progress=False,
    )
    market_data = {
        "BTCUSDT": {"exchange": "binance", "frame": _frame([1, 2, 3, 4, 5])},
        "COINUSDT": {"exchange": "binance", "frame": _frame([2, 3])},
        "MSTRUSDT": {"exchange": "binance", "frame": _frame([4, 5])},
    }

    streamed_bars = list(backtest._iter_market_bars(market_data))
    assert [backtest._bars_ts(item) for item in streamed_bars] == [2, 3, 4, 5]
    assert set(streamed_bars[0]["binance"]) == {"BTCUSDT", "COINUSDT"}
    assert set(streamed_bars[-1]["binance"]) == {"BTCUSDT", "MSTRUSDT"}


def test_backtest_records_compact_signal_on_every_bar_and_detailed_on_events():
    calls = []

    class Strategy:
        def snapshot_signal_state(self, ts, compact=False):
            calls.append((ts, compact))
            row = {"ts": ts, "pair_zscore": float(ts)}
            if not compact:
                row["pair_beta"] = 1.5
            return row

    backtest = Backtest(
        _config(), strategy=Strategy(), risk_manager=SimpleNamespace(),
        exchange_manager=SimpleNamespace(), show_progress=False,
    )

    backtest._record_signal_state(1, detailed=False)
    backtest._record_signal_state(2, detailed=True)

    assert calls == [(1, True), (2, False)]
    assert backtest.signal_curve == [
        {"ts": 1, "pair_zscore": 1.0},
        {"ts": 2, "pair_zscore": 2.0, "pair_beta": 1.5},
    ]


def test_streaming_market_data_works_from_paths(tmp_path: Path):
    btc = tmp_path / "BTCUSDT.csv"
    coin = tmp_path / "COINUSDT.csv"
    mstr = tmp_path / "MSTRUSDT.csv"
    btc.write_text(
        "ts,open,high,close,low,volume\n"
        "1,1,2,1.5,0.5,100\n"
        "2,2,3,2.5,1.5,100\n"
        "3,3,4,3.5,2.5,100\n",
        encoding="utf-8",
    )
    coin.write_text(
        "ts,open,high,close,low,volume\n"
        "2,2,3,2.5,1.5,100\n",
        encoding="utf-8",
    )
    mstr.write_text(
        "ts,open,high,close,low,volume\n"
        "3,3,4,3.5,2.5,100\n",
        encoding="utf-8",
    )

    assert list(iter_csv_bars(btc))[0] == [1000, 1.0, 2.0, 1.5, 0.5, 100.0]

    backtest = Backtest(
        _config(),
        strategy=SimpleNamespace(),
        risk_manager=SimpleNamespace(),
        exchange_manager=SimpleNamespace(),
        show_progress=False,
    )
    market_data = {
        "BTCUSDT": {"exchange": "binance", "path": btc},
        "COINUSDT": {"exchange": "binance", "path": coin},
        "MSTRUSDT": {"exchange": "binance", "path": mstr},
    }

    bars = list(backtest._iter_market_bars(market_data))
    assert [backtest._bars_ts(item) for item in bars] == [2000, 3000]
    assert set(bars[0]["binance"]) == {"BTCUSDT", "COINUSDT"}
    assert set(bars[1]["binance"]) == {"BTCUSDT", "MSTRUSDT"}


def test_grouped_market_orders_wait_until_all_legs_have_current_bars():
    exchange = Exchange("binance", initial_cash=100000.0, fee_rate=0.0, slippage_bps=0.0, max_leverage=10.0)
    manager = ExchangeManager({"binance": exchange})
    orders = [
        Order(
            order_id="o1",
            group_id="g1",
            exchange="binance",
            symbol="BTCUSDT",
            action=OrderAction.OPEN,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            pair_id="btc_coin",
        ),
        Order(
            order_id="o2",
            group_id="g1",
            exchange="binance",
            symbol="COINUSDT",
            action=OrderAction.OPEN,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=10.0,
            pair_id="btc_coin",
        ),
    ]
    manager.place_orders(orders)

    first = manager.on_bar({"binance": {"BTCUSDT": [1, 100.0, 101.0, 100.5, 99.0, 1000.0]}})
    assert first["new_trades"] == []
    assert len(exchange.orders) == 2

    second = manager.on_bar(
        {
            "binance": {
                "BTCUSDT": [2, 100.0, 101.0, 100.5, 99.0, 1000.0],
                "COINUSDT": [2, 10.0, 11.0, 10.5, 9.0, 1000.0],
            }
        }
    )
    assert len(second["new_trades"]) == 2
    assert exchange.orders == []


def test_backtest_start_time_filters_streamed_bars():
    config = _config()
    config.backtest_start_time = "1970-01-01T00:00:03Z"
    config.backtest_start_ts = 3000
    backtest = Backtest(
        config,
        strategy=SimpleNamespace(),
        risk_manager=SimpleNamespace(),
        exchange_manager=SimpleNamespace(),
        show_progress=False,
    )
    market_data = {
        "BTCUSDT": {"exchange": "binance", "frame": _frame([1000, 2000, 3000, 4000, 5000])},
        "COINUSDT": {"exchange": "binance", "frame": _frame([2000, 3000, 4000])},
        "MSTRUSDT": {"exchange": "binance", "frame": _frame([4000, 5000])},
    }

    bars = list(backtest._iter_market_bars(market_data))

    assert [backtest._bars_ts(item) for item in bars] == [3000, 4000, 5000]


def test_final_position_valuation_marks_open_positions_at_last_prices():
    backtest = Backtest.__new__(Backtest)
    backtest.exchange_manager = SimpleNamespace(
        exchanges={
            "binance": SimpleNamespace(
                cash=900.0,
                positions={"A": 2.0, "B": -3.0},
                last_bars={},
            )
        }
    )
    bars = {
        "binance": {
            "A": [10, 9.0, 11.0, 10.0, 8.0, 100.0],
            "B": [10, 3.0, 5.0, 4.0, 2.0, 100.0],
        }
    }

    valuation = backtest._final_position_valuation(bars)

    assert valuation["ts"] == 10
    assert valuation["cash"] == 900.0
    assert valuation["position_value"] == 8.0
    assert valuation["long_value"] == 20.0
    assert valuation["short_value"] == 12.0
    assert valuation["gross_exposure"] == 32.0
    assert valuation["net_exposure"] == 8.0
    assert valuation["equity"] == 908.0
