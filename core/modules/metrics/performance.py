from dataclasses import asdict, is_dataclass
from math import sqrt

import polars as pl


MILLISECONDS_PER_DAY = 24 * 60 * 60 * 1000


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
    result.update(trading_metrics(trade_df, equity))
    result.update(cost_metrics(trade_df, funding_df))
    result.update(neutrality_metrics(trade_df, order_df, benchmark_returns, hedge_ratio_tolerance))
    return result


def performance_metrics(equity_curve) -> dict:
    equity = _to_frame(equity_curve)
    if equity.is_empty() or "equity" not in equity.columns:
        return _empty_performance_metrics()

    equity = equity.select(["ts", "equity"]) if "ts" in equity.columns else equity.select(["equity"])
    equity = equity.with_columns(pl.col("equity").cast(pl.Float64, strict=False))
    equity = equity.drop_nulls("equity")
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
    annualized_return = mean_return * MINUTES_PER_YEAR if mean_return is not None else None
    annualized_volatility = std_return * sqrt(MINUTES_PER_YEAR) if std_return is not None else None
    sharpe = annualized_return / annualized_volatility if annualized_volatility else None

    equity = equity.with_columns(pl.col("equity").cum_max().alias("peak"))
    equity = equity.with_columns((pl.col("equity") / pl.col("peak") - 1).alias("drawdown"))
    max_drawdown = _series_min(equity, "drawdown")
    calmar = annualized_return / abs(max_drawdown) if max_drawdown and max_drawdown < 0 else None

    return {
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "calmar": calmar,
        "max_drawdown": max_drawdown,
    }


def trading_metrics(trades, equity_curve=None) -> dict:
    trades = _normalize_trades(_to_frame(trades))
    if trades.is_empty():
        return {
            "trade_count": 0,
            "win_rate": None,
            "profit_loss_ratio": None,
            "average_holding_minutes": None,
            "daily_turnover": None,
        }

    trade_count = trades.height
    position_pnl = _position_cashflow(trades)
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
            "total_slippage": None,
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
    return trades.drop_nulls(group_column).group_by(group_column).agg(pl.col("cashflow").sum().alias("pnl"))


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
        return pl.DataFrame([asdict(data)])
    if isinstance(data, list):
        if not data:
            return pl.DataFrame()
        rows = [asdict(item) if is_dataclass(item) else item for item in data]
        return pl.DataFrame(rows)
    if isinstance(data, dict):
        return pl.DataFrame(data)
    return pl.DataFrame(data)


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
        "annualized_volatility": None,
        "sharpe": None,
        "calmar": None,
        "max_drawdown": None,
    }
