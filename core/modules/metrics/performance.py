from dataclasses import asdict, is_dataclass
from enum import Enum
from math import isfinite, sqrt

import polars as pl


MINUTES_PER_YEAR = 365 * 24 * 60
MILLISECONDS_PER_DAY = 24 * 60 * 60 * 1000
MILLISECONDS_PER_YEAR = 365 * MILLISECONDS_PER_DAY


def calculate_metrics(
    equity_curve,
    orders=None,
    trades=None,
    funding_payments=None,
    benchmark_returns=None,
    hedge_ratio_tolerance=0.02,
) -> dict:
    equity = _to_frame(equity_curve)
    trade_df = _to_frame(trades)
    order_df = _to_frame(orders)
    funding_df = _to_frame(funding_payments)

    result = {}
    result.update(performance_metrics(equity))
    result.update(trading_metrics(trade_df, equity, funding_df))
    result.update(cost_metrics(trade_df, funding_df))
    return result


def performance_metrics(equity_curve) -> dict:
    equity = _to_frame(equity_curve)
    if equity.is_empty() or "equity" not in equity.columns:
        return _empty_performance_metrics()

    has_timestamps = "ts" in equity.columns
    equity = equity.select(["ts", "equity"]) if has_timestamps else equity.select(["equity"])
    equity = equity.with_columns(pl.col("equity").cast(pl.Float64, strict=False))
    equity = equity.drop_nulls("equity")
    if has_timestamps:
        equity = equity.sort("ts")
    if equity.height < 2:
        return _empty_performance_metrics()

    equity = equity.with_columns(
        (pl.col("equity") / pl.col("equity").shift(1) - 1).alias("return")
    )
    returns = equity.drop_nulls("return")

    initial_equity = equity["equity"][0]
    final_equity = equity["equity"][-1]
    total_return = final_equity / initial_equity - 1 if initial_equity else None

    mean_return = _series_mean(returns, "return")
    std_return = _series_std(returns, "return")
    annualization_factor = _annualization_factor(equity)
    annualized_mean_return = mean_return * annualization_factor if mean_return is not None else None
    annualized_return = _compound_annualized_return(equity, initial_equity, final_equity)
    annualized_volatility = std_return * sqrt(annualization_factor) if std_return is not None else None
    sharpe = annualized_mean_return / annualized_volatility if annualized_volatility else None

    equity = equity.with_columns(pl.col("equity").cum_max().alias("peak"))
    equity = equity.with_columns((pl.col("equity") / pl.col("peak") - 1).alias("drawdown"))
    max_drawdown = _series_min(equity, "drawdown")
    calmar = _calmar_ratio(annualized_return, max_drawdown)

    return {
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_mean_return": annualized_mean_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "calmar": calmar,
        "max_drawdown": max_drawdown,
    }


def trading_metrics(trades, equity_curve=None, funding_payments=None) -> dict:
    trades = _normalize_trades(_to_frame(trades))
    if trades.is_empty():
        return {
            "trade_count": 0,
            "fill_count": 0,
            "win_rate": None,
            "profit_loss_ratio": None,
            "average_holding_minutes": None,
            "daily_turnover": None,
        }

    fill_count = trades.height
    position_pnl = _position_cashflow(trades)
    position_funding = _position_funding(trades, _to_frame(funding_payments))
    if not position_pnl.is_empty() and position_funding:
        position_pnl = position_pnl.with_columns(
            pl.col("position_key")
            .map_elements(lambda value: position_funding.get(str(value), 0.0), return_dtype=pl.Float64)
            .alias("funding_fee")
        ).with_columns((pl.col("pnl") - pl.col("funding_fee")).alias("pnl"))
    group_column = "position_id" if "position_id" in trades.columns else "group_id"
    trade_count = (
        trades[group_column].drop_nulls().n_unique()
        if group_column in trades.columns else position_pnl.height
    )
    win_rate = None
    profit_loss_ratio = None

    if not position_pnl.is_empty():
        wins = position_pnl.filter(pl.col("pnl") > 0)
        losses = position_pnl.filter(pl.col("pnl") < 0)
        win_rate = wins.height / position_pnl.height if position_pnl.height else None

        avg_win = _series_mean(wins, "pnl")
        avg_loss = _series_mean(losses.with_columns(pl.col("pnl").abs()), "pnl")
        profit_loss_ratio = avg_win / avg_loss if avg_win is not None and avg_loss else None

    total_notional = trades["notional"].sum()
    average_equity = None
    if equity_curve is not None and not equity_curve.is_empty() and "equity" in equity_curve.columns:
        average_equity = equity_curve["equity"].mean()
    daily_turnover = _daily_turnover(trades, total_notional, average_equity, equity_curve)

    return {
        "trade_count": trade_count,
        "fill_count": fill_count,
        "win_rate": win_rate,
        "profit_loss_ratio": profit_loss_ratio,
        "average_holding_minutes": _average_holding_minutes(trades),
        "daily_turnover": daily_turnover,
    }


def cost_metrics(trades, funding_payments=None) -> dict:
    trades = _normalize_trades(_to_frame(trades))
    funding_payments = _to_frame(funding_payments)
    funding_stats = _funding_stats(funding_payments)
    if trades.is_empty():
        return {
            "total_fee": 0.0,
            "total_slippage": 0.0,
            **funding_stats,
            "funding_payment_count": funding_payments.height,
        }

    total_fee = trades["fee"].sum() if "fee" in trades.columns else 0.0
    total_slippage = trades["slippage"].sum() if "slippage" in trades.columns else None

    return {
        "total_fee": total_fee,
        "total_slippage": total_slippage,
        **funding_stats,
        "funding_payment_count": funding_payments.height,
    }


def neutrality_metrics(trades, orders=None, benchmark_returns=None, hedge_ratio_tolerance=0.02) -> dict:
    trades = _normalize_trades(_to_frame(trades))
    orders = _to_frame(orders)
    hedge_ratio = _hedge_ratio_metrics(trades, orders, hedge_ratio_tolerance)

    result = {
        **hedge_ratio,
        "open_symmetry_pass_rate": _open_symmetry_pass_rate(trades, orders),
        "average_position_deviation": _average_trade_deviation(trades),
        "portfolio_beta_btc": None,
        "portfolio_beta_coin": None,
        "portfolio_beta_spy": None,
        "portfolio_beta_qqq": None,
        "portfolio_corr_btc": None,
        "portfolio_corr_coin": None,
        "portfolio_corr_spy": None,
        "portfolio_corr_qqq": None,
    }

    if benchmark_returns is not None:
        result.update(_benchmark_metrics(benchmark_returns))

    return result


def _normalize_trades(trades):
    if trades.is_empty():
        return trades

    trades = trades.with_columns(
        pl.col("quantity").cast(pl.Float64, strict=False),
        pl.col("price").cast(pl.Float64, strict=False),
    )
    if "notional" in trades.columns:
        trades = trades.with_columns(pl.col("notional").cast(pl.Float64, strict=False).abs())
    else:
        trades = trades.with_columns((pl.col("quantity").abs() * pl.col("price").abs()).alias("notional"))

    if "fee" not in trades.columns:
        trades = trades.with_columns(pl.lit(0.0).alias("fee"))

    return trades


def _position_cashflow(trades):
    if trades.is_empty():
        return pl.DataFrame()

    group_column = "position_id" if "position_id" in trades.columns else "group_id"
    if group_column not in trades.columns:
        return pl.DataFrame()

    trades = trades.with_columns(
        pl.when(pl.col("side") == "sell")
        .then(pl.col("notional") - pl.col("fee"))
        .otherwise(-pl.col("notional") - pl.col("fee"))
        .alias("cashflow")
    )
    return (
        trades.drop_nulls(group_column)
        .group_by(group_column)
        .agg(
            pl.col("cashflow").sum().alias("pnl"),
            (pl.col("action").cast(pl.Utf8).str.to_lowercase() == "close").any().alias("is_closed"),
        )
        .filter(pl.col("is_closed"))
        .rename({group_column: "position_key"})
        .drop("is_closed")
    )


def _position_funding(trades: pl.DataFrame, funding: pl.DataFrame) -> dict[str, float]:
    """Allocate portfolio funding records to open position ids."""
    if trades.is_empty() or funding.is_empty():
        return {}
    group_column = "position_id" if "position_id" in trades.columns else "group_id"
    required_trade = {group_column, "ts", "symbol", "side", "quantity"}
    if not required_trade.issubset(trades.columns) or not {"ts", "symbol"}.issubset(funding.columns):
        return {}

    trade_rows = sorted(trades.to_dicts(), key=lambda row: int(row.get("ts") or 0))
    funding_rows = sorted(funding.to_dicts(), key=lambda row: int(row.get("ts") or 0))
    quantities: dict[tuple[str, str | None, str], float] = {}
    result: dict[str, float] = {}
    trade_index = 0
    for event in funding_rows:
        event_ts = int(event.get("ts") or 0)
        while trade_index < len(trade_rows) and int(trade_rows[trade_index].get("ts") or 0) < event_ts:
            row = trade_rows[trade_index]
            position_key = str(row.get(group_column) or "")
            if position_key:
                quantity = abs(float(row.get("quantity") or 0.0))
                signed = quantity if str(row.get("side") or "").lower() == "buy" else -quantity
                key = (position_key, row.get("exchange"), str(row.get("symbol") or ""))
                quantities[key] = quantities.get(key, 0.0) + signed
            trade_index += 1

        symbol = str(event.get("symbol") or "")
        exchange = event.get("exchange")
        active = {
            key: quantity for key, quantity in quantities.items()
            if key[2] == symbol
            and (exchange is None or key[1] is None or key[1] == exchange)
            and abs(quantity) > 1e-12
        }
        rate = _float_value(event.get("funding_rate"))
        mark = _float_value(event.get("mark_price"))
        payment = _float_value(event.get("payment"))
        if rate is not None and mark is not None:
            allocations = {key: quantity * mark * rate for key, quantity in active.items()}
        else:
            total_quantity = sum(active.values())
            if payment is None or abs(total_quantity) <= 1e-12:
                continue
            allocations = {key: payment * quantity / total_quantity for key, quantity in active.items()}
        for key, allocated in allocations.items():
            result[key[0]] = result.get(key[0], 0.0) + allocated
    return result


def _open_symmetry_pass_rate(trades, orders=None):
    orders = _to_frame(orders)
    if orders.is_empty() or "group_id" not in orders.columns:
        return None

    open_orders = orders.filter(pl.col("action").cast(pl.Utf8).str.to_lowercase().str.contains("open"))
    if open_orders.is_empty():
        return None

    open_orders = open_orders.with_columns(
        pl.col("quantity").cast(pl.Float64, strict=False),
        pl.col("price").cast(pl.Float64, strict=False),
    )
    open_orders = open_orders.with_columns(
        pl.when(pl.col("price").is_not_null())
        .then((pl.col("price") * pl.col("quantity")).abs())
        .otherwise(pl.lit(None))
        .alias("target_notional")
    )
    if open_orders["target_notional"].null_count() == open_orders.height:
        open_orders = open_orders.with_columns(pl.lit(1.0).alias("target_notional"))

    grouped = open_orders.group_by("group_id").agg(
        pl.when(pl.col("side").cast(pl.Utf8).str.to_lowercase().str.contains("buy"))
        .then(pl.col("target_notional"))
        .otherwise(0.0)
        .sum()
        .alias("long_notional"),
        pl.when(pl.col("side").cast(pl.Utf8).str.to_lowercase().str.contains("sell"))
        .then(pl.col("target_notional"))
        .otherwise(0.0)
        .sum()
        .alias("short_notional"),
    )
    grouped = grouped.with_columns(
        ((pl.col("long_notional") - pl.col("short_notional")).abs() / pl.max_horizontal("long_notional", "short_notional")).alias("deviation")
    )
    return grouped.filter(pl.col("deviation") <= 0.02).height / grouped.height if grouped.height else None


def _average_trade_deviation(trades):
    if trades.is_empty() or "group_id" not in trades.columns:
        return None

    grouped = trades.group_by("group_id").agg(
        pl.when(pl.col("side") == "buy")
        .then(pl.col("notional"))
        .otherwise(0.0)
        .sum()
        .alias("long_notional"),
        pl.when(pl.col("side") == "sell")
        .then(pl.col("notional"))
        .otherwise(0.0)
        .sum()
        .alias("short_notional"),
    )
    grouped = grouped.with_columns(
        ((pl.col("long_notional") - pl.col("short_notional")).abs() / pl.max_horizontal("long_notional", "short_notional")).alias("deviation")
    )
    return _series_mean(grouped.drop_nulls("deviation"), "deviation")


def _hedge_ratio_metrics(trades, orders=None, tolerance=0.02):
    result = _hedge_ratio_values(trades, tolerance)
    if result["average_hedge_ratio_deviation"] is None:
        result = _hedge_ratio_values(orders, tolerance)
    return result


def _hedge_ratio_values(rows, tolerance):
    frame = _to_frame(rows)
    if frame.is_empty() or "group_id" not in frame.columns:
        return {
            "hedge_ratio_pass_rate": None,
            "average_hedge_ratio_deviation": None,
        }

    groups = {}
    for row in frame.iter_rows(named=True):
        if "open" not in _lower_value(row.get("action")):
            continue

        group_id = row.get("group_id")
        if group_id is None:
            continue

        notional = _row_notional(row)
        if notional is None or notional <= 0:
            continue

        target_ratio = _float_value(row.get("target_hedge_ratio"))
        group = groups.setdefault(
            group_id,
            {"long_notional": 0.0, "short_notional": 0.0, "target_ratio": None},
        )
        if target_ratio is not None and target_ratio > 0 and group["target_ratio"] is None:
            group["target_ratio"] = target_ratio

        side = _lower_value(row.get("side"))
        if "buy" in side:
            group["long_notional"] += notional
        elif "sell" in side:
            group["short_notional"] += notional

    deviations = []
    for group in groups.values():
        long_notional = group["long_notional"]
        short_notional = group["short_notional"]
        target_ratio = group["target_ratio"]

        if long_notional <= 0 or short_notional <= 0 or not target_ratio:
            continue

        actual_ratio = long_notional / short_notional
        deviations.append(abs(actual_ratio - target_ratio) / target_ratio)

    if not deviations:
        return {
            "hedge_ratio_pass_rate": None,
            "average_hedge_ratio_deviation": None,
        }

    return {
        "hedge_ratio_pass_rate": sum(value <= tolerance for value in deviations) / len(deviations),
        "average_hedge_ratio_deviation": sum(deviations) / len(deviations),
    }


def _row_notional(row):
    notional = _float_value(row.get("notional"))
    if notional is not None:
        return abs(notional)

    quantity = _float_value(row.get("quantity"))
    price = _float_value(row.get("price"))
    if quantity is None or price is None:
        return None
    return abs(quantity * price)


def _float_value(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _lower_value(value):
    if value is None:
        return ""
    return str(value).lower()


def _average_holding_minutes(trades):
    if trades.is_empty() or "ts" not in trades.columns:
        return None

    group_column = "position_id" if "position_id" in trades.columns else "group_id"
    if group_column not in trades.columns:
        return None

    grouped = trades.drop_nulls(group_column).group_by(group_column).agg(
        pl.col("ts").filter(pl.col("action") == "open").min().alias("start_ts"),
        pl.col("ts").filter(pl.col("action") == "close").max().alias("end_ts"),
    )
    grouped = grouped.drop_nulls(["start_ts", "end_ts"])
    if grouped.is_empty():
        return None

    try:
        grouped = grouped.with_columns(((pl.col("end_ts") - pl.col("start_ts")) / 60000).alias("holding_minutes"))
        return _series_mean(grouped, "holding_minutes")
    except Exception:
        return None


def _daily_turnover(trades, total_notional, average_equity, equity_curve=None):
    if trades.is_empty() or not average_equity:
        return None

    elapsed_days = _elapsed_days(equity_curve)
    if elapsed_days is None:
        elapsed_days = _elapsed_days(trades)

    if not elapsed_days:
        return None

    return total_notional / average_equity / elapsed_days


def _elapsed_days(frame):
    frame = _to_frame(frame)
    if frame.is_empty() or "ts" not in frame.columns or frame.height < 2:
        return None

    try:
        start = frame["ts"].min()
        end = frame["ts"].max()
        elapsed_days = (end - start) / MILLISECONDS_PER_DAY
    except Exception:
        return None

    return elapsed_days if elapsed_days > 0 else None


def _annualization_factor(equity):
    if equity.is_empty() or "ts" not in equity.columns:
        return MINUTES_PER_YEAR

    timestamps = (
        equity.select(pl.col("ts").cast(pl.Int64, strict=False).alias("ts"))
        .drop_nulls("ts")
        .sort("ts")
    )
    if timestamps.height < 2:
        return MINUTES_PER_YEAR

    diffs = (
        timestamps.with_columns(pl.col("ts").diff().alias("interval_ms"))
        .drop_nulls("interval_ms")
        .filter(pl.col("interval_ms") > 0)
    )
    if diffs.is_empty():
        return MINUTES_PER_YEAR

    interval_ms = float(diffs["interval_ms"].median())
    if interval_ms <= 0 or not isfinite(interval_ms):
        return MINUTES_PER_YEAR
    return MILLISECONDS_PER_YEAR / interval_ms


def _compound_annualized_return(equity, initial_equity, final_equity):
    """Return CAGR from the equity endpoints and their actual elapsed time."""
    if initial_equity is None or final_equity is None:
        return None

    try:
        initial_equity = float(initial_equity)
        final_equity = float(final_equity)
    except (TypeError, ValueError):
        return None

    if not isfinite(initial_equity) or not isfinite(final_equity):
        return None
    if initial_equity <= 0 or final_equity < 0:
        return None

    elapsed_days = _elapsed_days(equity)
    if elapsed_days is None:
        return None
    if final_equity == 0:
        return -1.0

    years = elapsed_days / 365.0
    if years <= 0 or not isfinite(years):
        return None

    value = (final_equity / initial_equity) ** (1.0 / years) - 1.0
    return value if isfinite(value) else None


def _calmar_ratio(annualized_return, max_drawdown):
    if annualized_return is None or max_drawdown is None:
        return None
    if max_drawdown < 0:
        return annualized_return / abs(max_drawdown)
    if max_drawdown == 0:
        if annualized_return > 0:
            return float("inf")
        if annualized_return < 0:
            return float("-inf")
        return 0.0
    return None


def _benchmark_metrics(benchmark_returns):
    frame = _to_frame(benchmark_returns)
    result = {}
    if frame.is_empty() or "portfolio_return" not in frame.columns:
        return result

    for name in ["btc", "coin", "spy", "qqq"]:
        column = f"{name}_return"
        if column not in frame.columns:
            continue
        result[f"portfolio_beta_{name}"] = _beta(frame, "portfolio_return", column)
        result[f"portfolio_corr_{name}"] = _corr(frame, "portfolio_return", column)
    return result


def _beta(frame, y_col, x_col):
    frame = frame.select([y_col, x_col]).drop_nulls()
    if frame.height < 2:
        return None
    covariance = frame.select(pl.cov(y_col, x_col)).item()
    variance = frame.select(pl.var(x_col)).item()
    return covariance / variance if variance else None


def _corr(frame, y_col, x_col):
    frame = frame.select([y_col, x_col]).drop_nulls()
    if frame.height < 2:
        return None
    return frame.select(pl.corr(y_col, x_col)).item()


def _funding_stats(funding_payments):
    funding_payments = _to_frame(funding_payments)
    if funding_payments.is_empty() or "payment" not in funding_payments.columns:
        return {
            "funding_fee": 0.0,
            "funding_paid": 0.0,
            "funding_received": 0.0,
        }

    funding_payments = funding_payments.with_columns(pl.col("payment").cast(pl.Float64, strict=False))
    return {
        "funding_fee": funding_payments["payment"].sum(),
        "funding_paid": funding_payments.filter(pl.col("payment") > 0)["payment"].sum(),
        "funding_received": -funding_payments.filter(pl.col("payment") < 0)["payment"].sum(),
    }


def _to_frame(data):
    if data is None:
        return pl.DataFrame()
    if isinstance(data, pl.DataFrame):
        return data
    if is_dataclass(data):
        return _rows_to_frame([asdict(data)])
    if isinstance(data, list):
        if not data:
            return pl.DataFrame()
        rows = [asdict(item) if is_dataclass(item) else item for item in data]
        if rows and all(isinstance(row, dict) for row in rows):
            return _rows_to_frame(rows)
        return pl.DataFrame(data)
    if isinstance(data, dict):
        return pl.DataFrame(data)
    return pl.DataFrame(data)


def _rows_to_frame(rows: list[dict]) -> pl.DataFrame:
    """Build metric inputs without inferring sparse exit metadata as Null.

    Most fills have no exit metadata; only protective close fills contain
    strings such as ``protective_max_holding_time``.  Polars' default
    ``infer_schema_length=100`` can therefore infer a Null builder from the
    first rows and fail when it reaches a later close.  Explicit overrides
    keep the metric path consistent with the report exporter.
    """
    normalized = []
    for row in rows:
        value = {}
        for key, item in row.items():
            if key == "para":
                continue
            if isinstance(item, Enum):
                item = item.value
            value[key] = item
        normalized.append(value)

    string_columns = {
        "order_id", "group_id", "exchange", "symbol", "action", "side",
        "order_type", "status", "cancel_order_id", "position_id", "pair_id",
        "exit_reason", "protection_trigger", "exit_class", "protection_stop_reason", "protection_rule",
        "rebalance_batch_id",
    }
    bool_columns = {"reopen_lock_pending"}
    int_columns = {
        "ts", "protection_max_holding_bars", "protection_max_holding_deadline_bar",
        "protection_freeze_bars", "protection_freeze_until_bar",
        "protection_pair_loss_stop_freeze_bars", "protection_pair_loss_stop_freeze_until_bar",
        "entry_count", "last_entry_bar_index", "add_cooldown_remaining_bars",
        "min_hold_remaining_bars",
    }
    float_columns = {
        "quantity", "price", "notional", "fee", "slippage", "requested_quantity",
        "fill_scale", "funding_fee", "target_hedge_ratio", "protection_stop_x_price",
        "protection_take_profit_return", "protection_target_residual",
        "protection_pair_loss_stop_return",
    }
    overrides = {}
    if normalized:
        columns = {key for row in normalized for key in row}
        overrides.update({key: pl.Utf8 for key in columns if key in string_columns})
        overrides.update({key: pl.Boolean for key in columns if key in bool_columns})
        overrides.update({key: pl.Int64 for key in columns if key in int_columns})
        overrides.update({key: pl.Float64 for key in columns if key in float_columns})
    return pl.from_dicts(
        normalized,
        schema_overrides=overrides or None,
        infer_schema_length=100,
        strict=False,
    )


def _series_mean(frame, column):
    if frame.is_empty() or column not in frame.columns:
        return None
    return frame[column].mean()


def _series_std(frame, column):
    if frame.is_empty() or column not in frame.columns:
        return None
    value = frame[column].std()
    return value if value is not None else None


def _series_min(frame, column):
    if frame.is_empty() or column not in frame.columns:
        return None
    return frame[column].min()


def _empty_performance_metrics():
    return {
        "initial_equity": None,
        "final_equity": None,
        "total_return": None,
        "annualized_return": None,
        "annualized_mean_return": None,
        "annualized_volatility": None,
        "sharpe": None,
        "calmar": None,
        "max_drawdown": None,
    }
