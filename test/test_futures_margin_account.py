from math import isclose
from types import SimpleNamespace

from core.backtest.backtest import Backtest
from core.modules.config import SetupConfig
from core.modules.exchange import ExchangeManager
from core.modules.exchange.exchange import Exchange
from core.modules.execution.order_planner import OrderPlanner
from core.modules.models import Order, OrderAction, OrderSide, OrderType
from core.modules.models.pipeline_types import AllocatedPairTarget, PortfolioState, RawPairTarget
from core.modules.portfolio.allocator import EquitySlotAllocator
from core.modules.strategy.config import PairDefinition
from core.modules.strategy.factory import build_strategy


def _bar(ts=1, price=10.0):
    return [ts, price, price, price, price, 1.0]


def _ohlc_bar(ts, open_price, close_price):
    return [
        ts,
        open_price,
        max(open_price, close_price),
        close_price,
        min(open_price, close_price),
        1.0,
    ]


def _order(order_id, group_id, symbol, side, quantity, action=OrderAction.OPEN):
    return Order(
        order_id=order_id,
        group_id=group_id,
        exchange="binance",
        symbol=symbol,
        action=action,
        side=side,
        order_type=OrderType.MARKET,
        quantity=quantity,
        pair_id=group_id,
        position_id=group_id,
    )


def test_short_open_consumes_margin_and_never_credits_wallet():
    exchange = Exchange("binance", initial_cash=100.0, fee_rate=0.0, slippage_bps=0.0)
    manager = ExchangeManager({"binance": exchange})
    manager.place_orders([_order("s", "p", "Y", OrderSide.SELL, 5.0)])

    result = manager.on_bar({"binance": {"Y": _bar(price=10.0)}})

    assert len(result["new_trades"]) == 1
    assert exchange.wallet_balance == 100.0
    assert exchange.used_margin == 50.0
    assert exchange.available_balance == 50.0


def test_long_and_short_margin_is_absolute_gross_not_net():
    exchange = Exchange("binance", initial_cash=100.0, fee_rate=0.0, slippage_bps=0.0)
    manager = ExchangeManager({"binance": exchange})
    manager.place_orders([
        _order("x", "p", "X", OrderSide.BUY, 8.0),
        _order("y", "p", "Y", OrderSide.SELL, 8.0),
    ])

    result = manager.on_bar({"binance": {"X": _bar(), "Y": _bar()}})
    fills = {trade.symbol: trade for trade in result["new_trades"]}

    assert fills["X"].quantity == fills["Y"].quantity == 5.0
    assert fills["X"].fill_scale == fills["Y"].fill_scale == 0.625
    assert exchange.wallet_balance == 100.0
    assert exchange.used_margin == 100.0
    assert exchange.available_balance == 0.0


def test_multiple_groups_on_same_bar_share_one_margin_budget():
    exchange = Exchange("binance", initial_cash=100.0, fee_rate=0.0, slippage_bps=0.0)
    manager = ExchangeManager({"binance": exchange})
    manager.place_orders([
        _order("a1", "a", "A1", OrderSide.BUY, 4.0),
        _order("a2", "a", "A2", OrderSide.SELL, 4.0),
        _order("b1", "b", "B1", OrderSide.BUY, 4.0),
        _order("b2", "b", "B2", OrderSide.SELL, 4.0),
    ])
    bars = {symbol: _bar() for symbol in ("A1", "A2", "B1", "B2")}

    result = manager.on_bar({"binance": bars})
    fills = {(trade.group_id, trade.symbol): trade for trade in result["new_trades"]}

    assert fills[("a", "A1")].quantity == 4.0
    assert fills[("a", "A2")].quantity == 4.0
    assert isclose(fills[("b", "B1")].quantity, 1.0)
    assert isclose(fills[("b", "B2")].quantity, 1.0)
    assert isclose(exchange.used_margin, 100.0)
    assert isclose(exchange.available_balance, 0.0, abs_tol=1e-9)


def test_open_scale_reserves_fees_as_well_as_gross_margin():
    exchange = Exchange(
        "binance", initial_cash=100.0, fee_rate=0.001,
        slippage_bps=0.0, max_leverage=1.0,
    )
    manager = ExchangeManager({"binance": exchange})
    manager.place_orders([
        _order("x", "p", "X", OrderSide.BUY, 5.0),
        _order("y", "p", "Y", OrderSide.SELL, 5.0),
    ])

    result = manager.on_bar({"binance": {"X": _bar(), "Y": _bar()}})
    fills = {trade.symbol: trade for trade in result["new_trades"]}

    expected_scale = 100.0 / 100.1
    assert isclose(fills["X"].fill_scale, expected_scale, rel_tol=1e-9)
    assert isclose(fills["Y"].fill_scale, expected_scale, rel_tol=1e-9)
    assert exchange.available_balance >= -1e-8


def test_closing_releases_margin_and_realizes_long_and_short_pnl():
    exchange = Exchange("binance", initial_cash=100.0, fee_rate=0.0, slippage_bps=0.0)
    manager = ExchangeManager({"binance": exchange})
    manager.place_orders([
        _order("x-open", "p", "X", OrderSide.BUY, 5.0),
        _order("y-open", "p", "Y", OrderSide.SELL, 5.0),
    ])
    manager.on_bar({"binance": {"X": _bar(1, 10.0), "Y": _bar(1, 10.0)}})
    manager.place_orders([
        _order("x-close", "p-close", "X", OrderSide.SELL, 5.0, OrderAction.CLOSE),
        _order("y-close", "p-close", "Y", OrderSide.BUY, 5.0, OrderAction.CLOSE),
    ])
    for order in exchange.orders:
        order.position_id = "p"
    manager.on_bar({"binance": {"X": _bar(2, 11.0), "Y": _bar(2, 9.0)}})

    assert exchange.positions == {}
    assert exchange.used_margin == 0.0
    assert exchange.available_balance == exchange.equity == 110.0
    assert exchange.wallet_balance == 110.0
    assert exchange.position_pair_ids == {}


def test_risk_pass_through_does_not_disable_exchange_leverage_limit():
    backtest = Backtest.__new__(Backtest)
    backtest.config = SimpleNamespace(
        risk={"enabled": False, "mode": "pass_through", "max_leverage": 1.0}
    )
    assert backtest._max_leverage() == 1.0




def test_rebalance_close_groups_release_margin_before_open_groups():
    exchange = Exchange("binance", initial_cash=100.0, fee_rate=0.0, slippage_bps=0.0)
    manager = ExchangeManager({"binance": exchange})
    manager.place_orders([
        _order("px", "p-open", "X", OrderSide.BUY, 5.0),
        _order("py", "p-open", "Y", OrderSide.SELL, 5.0),
    ])
    manager.on_bar({"binance": {"X": _bar(price=10.0), "Y": _bar(price=10.0)}})
    # The replacement is queued together: it must close p before opening q.
    close_x = _order("cx", "p-close", "X", OrderSide.SELL, 5.0, OrderAction.CLOSE)
    close_y = _order("cy", "p-close", "Y", OrderSide.BUY, 5.0, OrderAction.CLOSE)
    close_x.position_id = close_y.position_id = "p-open"
    open_x = _order("qx", "q-open", "QX", OrderSide.BUY, 5.0)
    open_y = _order("qy", "q-open", "QY", OrderSide.SELL, 5.0)
    manager.place_orders([close_x, close_y, open_x, open_y])
    result = manager.on_bar({"binance": {
        "X": _bar(2, 10.0), "Y": _bar(2, 10.0),
        "QX": _bar(2, 10.0), "QY": _bar(2, 10.0),
    }})
    assert {trade.group_id for trade in result["new_trades"]} == {"p-close", "q-open"}
    assert exchange.used_margin == 100.0
    assert exchange.available_balance == 0.0


def test_adverse_price_move_changes_equity_but_not_unallocated_cash():
    exchange = Exchange("binance", initial_cash=200.0, fee_rate=0.0, slippage_bps=0.0)
    manager = ExchangeManager({"binance": exchange})
    manager.place_orders([
        _order("x", "p", "X", OrderSide.BUY, 5.0),
        _order("y", "p", "Y", OrderSide.SELL, 5.0),
    ])
    manager.on_bar({"binance": {"X": _bar(1, 10.0), "Y": _bar(1, 10.0)}})

    result = manager.on_bar({"binance": {
        "X": _ohlc_bar(2, 10.0, 10.0),
        "Y": _ohlc_bar(2, 10.0, 12.0),
    }})

    assert result["new_trades"] == []
    assert result["forced_deleveraging_triggered"] is False
    assert isclose(exchange.unrealized_pnl, -10.0)
    assert isclose(exchange.equity, 190.0)
    assert isclose(exchange.used_margin, 100.0)
    assert isclose(exchange.available_balance, 100.0)
    assert isclose(abs(exchange.positions["X"]), abs(exchange.positions["Y"]), rel_tol=1e-12)


def test_margin_deficit_fully_closes_worst_pair_instead_of_shaving_all_pairs():
    exchange = Exchange("binance", initial_cash=100.0, fee_rate=0.0, slippage_bps=0.0)
    manager = ExchangeManager({"binance": exchange})
    manager.place_orders([
        _order("a1", "a", "A1", OrderSide.BUY, 2.0),
        _order("a2", "a", "A2", OrderSide.SELL, 2.0),
        _order("b1", "b", "B1", OrderSide.BUY, 2.0),
        _order("b2", "b", "B2", OrderSide.SELL, 2.0),
    ])
    manager.on_bar({"binance": {
        symbol: _bar(1, 10.0) for symbol in ("A1", "A2", "B1", "B2")
    }})

    # Simulate an account-level charge (for example funding) that makes free
    # cash negative. Pair A is the worst current position and must be closed
    # completely; Pair B must not be reduced at all.
    exchange.wallet_balance -= 25.0
    exchange.cash = exchange.wallet_balance
    result = manager.on_bar({"binance": {
        "A1": _ohlc_bar(2, 10.0, 10.0),
        "A2": _ohlc_bar(2, 12.0, 12.0),
        "B1": _ohlc_bar(2, 10.0, 10.0),
        "B2": _ohlc_bar(2, 10.0, 10.0),
    }})

    forced = [
        trade for trade in result["new_trades"]
        if trade.exit_reason == "forced_margin_pair_exit"
    ]
    assert len(forced) == 2
    assert {trade.pair_id for trade in forced} == {"a"}
    assert all(isclose(float(trade.fill_scale), 1.0) for trade in forced)
    assert "a" not in exchange.position_lots
    assert "b" in exchange.position_lots
    assert isclose(abs(exchange.positions["B1"]), 2.0)
    assert isclose(abs(exchange.positions["B2"]), 2.0)
    assert exchange.available_balance >= -1e-8
    assert result["forced_deleveraging_triggered"] is True


def test_position_occupancy_uses_occupied_plus_unused_capital():
    backtest = Backtest.__new__(Backtest)
    snapshot = backtest._position_snapshot(
        1,
        {},
        {
            "equity": 100.0,
            "cash": 100.0,
            "wallet_balance": 100.0,
            "unrealized_pnl": 0.0,
            "used_margin": 80.0,
            "available_balance": 20.0,
            "gross_exposure": 80.0,
            "net_exposure": 0.0,
            "open_pair_count": 1,
            "pending_open_pair_count": 0,
        },
    )

    assert snapshot["free_cash"] == 20.0
    assert snapshot["gross_exposure_ratio"] == 0.8
    assert snapshot["margin_utilization"] == 0.8
    assert snapshot["gross_exposure_ratio"] <= 1.0


