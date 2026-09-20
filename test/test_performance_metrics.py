import math

import polars as pl

from core.modules.metrics.performance import (
    MILLISECONDS_PER_DAY,
    _to_frame,
    performance_metrics,
    trading_metrics,
)


def _equity(values, elapsed_days):
    step_ms = elapsed_days * MILLISECONDS_PER_DAY / (len(values) - 1)
    return pl.DataFrame(
        {
            "ts": [round(index * step_ms) for index in range(len(values))],
            "equity": values,
        }
    )


def test_annualized_return_is_cagr_from_actual_elapsed_time():
    metrics = performance_metrics(_equity([100.0, 90.0, 80.0], elapsed_days=730))

    assert math.isclose(metrics["total_return"], -0.20)
    assert math.isclose(metrics["annualized_return"], math.sqrt(0.8) - 1.0)
    assert metrics["annualized_return"] >= -1.0


def test_sharpe_uses_annualized_arithmetic_mean_but_calmar_uses_cagr():
    equity = _equity([100.0, 110.0, 99.0], elapsed_days=365)
    metrics = performance_metrics(equity)

    returns = [0.10, -0.10]
    expected_mean = sum(returns) / len(returns)
    expected_std = math.sqrt(sum((value - expected_mean) ** 2 for value in returns) / (len(returns) - 1))
    periods_per_year = 2.0
    expected_annualized_mean = expected_mean * periods_per_year
    expected_volatility = expected_std * math.sqrt(periods_per_year)

    assert math.isclose(metrics["annualized_mean_return"], expected_annualized_mean, abs_tol=1e-12)
    assert math.isclose(metrics["annualized_volatility"], expected_volatility)
    assert math.isclose(metrics["sharpe"], expected_annualized_mean / expected_volatility, abs_tol=1e-12)
    assert math.isclose(metrics["annualized_return"], -0.01)
    assert math.isclose(metrics["max_drawdown"], -0.10)
    assert math.isclose(metrics["calmar"], -0.10)


def test_zero_final_equity_has_minus_one_cagr():
    metrics = performance_metrics(_equity([100.0, 0.0], elapsed_days=365))

    assert metrics["annualized_return"] == -1.0
    assert metrics["calmar"] == -1.0


def test_cagr_requires_timestamps_and_nonnegative_equity():
    no_timestamps = performance_metrics(pl.DataFrame({"equity": [100.0, 110.0]}))
    negative_final = performance_metrics(_equity([100.0, -1.0], elapsed_days=365))

    assert no_timestamps["annualized_return"] is None
    assert no_timestamps["calmar"] is None
    assert negative_final["annualized_return"] is None
    assert negative_final["calmar"] is None


def test_equity_is_ordered_by_timestamp_before_endpoint_metrics():
    equity = pl.DataFrame(
        {
            "ts": [MILLISECONDS_PER_DAY * 365, 0, MILLISECONDS_PER_DAY * 180],
            "equity": [121.0, 100.0, 110.0],
        }
    )

    metrics = performance_metrics(equity)

    assert math.isclose(metrics["total_return"], 0.21)
    assert math.isclose(metrics["annualized_return"], 0.21)


def test_trade_count_counts_closed_positions_and_win_rate_includes_funding():
    trades = pl.DataFrame(
        {
            "position_id": ["p1", "p1", "p1", "p1"],
            "action": ["open", "open", "close", "close"],
            "side": ["buy", "sell", "sell", "buy"],
            "symbol": ["X", "Y", "X", "Y"],
            "exchange": ["binance"] * 4,
            "quantity": [1.0] * 4,
            "price": [100.0, 100.0, 101.0, 100.0],
            "notional": [100.0, 100.0, 101.0, 100.0],
            "fee": [0.0] * 4,
            "ts": [1_000, 1_000, 2_000, 2_000],
        }
    )
    funding = pl.DataFrame(
        {
            "exchange": ["binance"],
            "symbol": ["X"],
            "ts": [1_500],
            "funding_rate": [0.02],
            "mark_price": [100.0],
            "payment": [2.0],
        }
    )

    metrics = trading_metrics(trades, funding_payments=funding)

    assert metrics["fill_count"] == 4
    assert metrics["trade_count"] == 1
    assert metrics["win_rate"] == 0.0


def test_sparse_protective_exit_metadata_does_not_create_null_builder():
    rows = [
        {"group_id": f"g-{index}", "action": "open", "exit_reason": None}
        for index in range(101)
    ]
    rows.append(
        {
            "group_id": "g-close",
            "action": "close",
            "exit_reason": "protective_max_holding_time",
        }
    )

    frame = _to_frame(rows)

    assert frame.schema["exit_reason"] == pl.Utf8
    assert frame.tail(1)["exit_reason"].item() == "protective_max_holding_time"
