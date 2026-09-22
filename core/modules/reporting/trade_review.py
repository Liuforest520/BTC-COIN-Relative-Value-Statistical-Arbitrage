from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from html import escape
from math import exp, isfinite, log
from json import dumps, loads
from pathlib import Path
from time import perf_counter

import polars as pl
import yaml

from core.modules.data import load_csv_data
from core.modules.reporting.pair_summary import (
    build_pair_summary,
    required_position_columns,
)
from core.modules.logger import logger
from core.modules.reporting.review_assets import (
    OVERVIEW_TEMPLATE,
    PAIR_TEMPLATE,
    REVIEW_CSS,
    REVIEW_JS,
)
from core.modules.reporting.utils import (
    REPORT_BOOL_COLUMNS,
    REPORT_INT_COLUMNS,
    REPORT_STRING_COLUMNS,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "results" / "review" / "trade_review.html"
def latest_backtest_dir() -> Path:
    backtest_root = ROOT / "results" / "backtests"
    runs = [path for path in backtest_root.iterdir() if path.is_dir()]
    if not runs:
        raise FileNotFoundError("results/backtests 下面没有回测结果")
    return max(runs, key=lambda path: path.stat().st_mtime)


def read_frame(path: Path, columns: list[str] | None = None) -> pl.DataFrame:
    if not path.exists():
        return _empty_frame(columns)
    try:
        requested = columns or []
        schema_overrides = {
            column: pl.Utf8
            for column in requested
            if column in REPORT_STRING_COLUMNS
        }
        schema_overrides.update(
            {
                column: pl.Boolean
                for column in requested
                if column in REPORT_BOOL_COLUMNS
            }
        )
        schema_overrides.update(
            {
                column: pl.Int64
                for column in requested
                if column in REPORT_INT_COLUMNS
            }
        )
        if columns is None:
            frame = pl.read_csv(path, infer_schema_length=10000)
        else:
            available = set(pl.scan_csv(path).collect_schema().names())
            selected = [column for column in columns if column in available]
            frame = pl.read_csv(
                path,
                columns=selected,
                schema_overrides=schema_overrides or None,
                infer_schema_length=10000,
            )
    except Exception:
        return _empty_frame(columns)
    if columns is None:
        return frame
    for column in columns:
        if column not in frame.columns:
            frame = frame.with_columns(pl.lit(None).alias(column))
    return frame.select(columns)


def _read_final_position_valuation(run_dir: Path) -> dict:
    path = run_dir / "final_position_valuation.json"
    if not path.exists():
        return {}
    try:
        value = loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _empty_frame(columns: list[str] | None = None) -> pl.DataFrame:
    values = {}
    for column in columns or []:
        dtype = pl.Int64 if column == "ts" else pl.Null
        values[column] = pl.Series(column, [], dtype=dtype)
    return pl.DataFrame(values)


def fmt_time(ts: int | float | None) -> str:
    if ts is None:
        return ""
    return datetime.fromtimestamp(float(ts) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def drawdown_window(equity: pl.DataFrame) -> dict:
    if equity.is_empty() or "ts" not in equity.columns or "equity" not in equity.columns:
        return empty_window("最大回撤")
    frame = equity.with_columns(
        pl.col("equity").cum_max().alias("peak"),
    ).with_columns(
        (pl.col("equity") / pl.col("peak") - 1).alias("drawdown"),
    ).drop_nulls(["ts", "equity", "drawdown"])
    if frame.is_empty():
        return empty_window("最大回撤")
    trough = frame.sort("drawdown").row(0, named=True)
    peak_before = frame.filter(pl.col("ts") <= trough["ts"]).filter(pl.col("equity") == trough["peak"]).tail(1)
    start_ts = peak_before["ts"][0] if peak_before.height else frame["ts"][0]
    end_ts = trough["ts"]
    padding = max(12 * 60 * 60 * 1000, int((end_ts - start_ts) * 0.25))
    return {
        "name": "最大回撤",
        "start": max(int(frame["ts"].min()), int(start_ts - padding)),
        "end": min(int(frame["ts"].max()), int(end_ts + padding)),
        "peak_ts": int(start_ts),
        "trough_ts": int(end_ts),
        "drawdown": float(trough["drawdown"]),
        "description": f"{fmt_time(start_ts)} 到 {fmt_time(end_ts)}，资金从阶段高点回落 {abs(float(trough['drawdown'])):.2%}",
    }


def best_gain_window(equity: pl.DataFrame) -> dict:
    if equity.is_empty() or "ts" not in equity.columns or "equity" not in equity.columns:
        return empty_window("最高收益段")
    values = equity.select(["ts", "equity"]).to_dicts()
    if not values:
        return empty_window("最高收益段")
    min_idx = 0
    best_start = 0
    best_end = 0
    best_gain = 0.0
    for idx, row in enumerate(values):
        current_gain = row["equity"] / values[min_idx]["equity"] - 1
        if current_gain > best_gain:
            best_gain = current_gain
            best_start = min_idx
            best_end = idx
        if row["equity"] < values[min_idx]["equity"]:
            min_idx = idx

    if best_gain <= 0:
        start_ts = int(equity["ts"].min())
        end_ts = int(equity["ts"].max())
        return {
            "name": "无正收益区间",
            "start": start_ts,
            "end": end_ts,
            "low_ts": start_ts,
            "high_ts": end_ts,
            "gain": 0.0,
            "description": "该回测区间内没有正收益窗口",
        }

    start_ts = int(values[best_start]["ts"])
    end_ts = int(values[best_end]["ts"])
    padding = max(12 * 60 * 60 * 1000, int((end_ts - start_ts) * 0.15))
    return {
        "name": "最高收益段",
        "start": max(int(equity["ts"].min()), start_ts - padding),
        "end": min(int(equity["ts"].max()), end_ts + padding),
        "low_ts": start_ts,
        "high_ts": end_ts,
        "gain": float(best_gain),
        "description": f"{fmt_time(start_ts)} 到 {fmt_time(end_ts)}，资金从阶段低点上涨 {best_gain:.2%}",
    }


def empty_window(name: str) -> dict:
    return {
        "name": name,
        "start": None,
        "end": None,
        "description": "没有足够数据",
    }


def trade_rows(trades: pl.DataFrame, signal: pl.DataFrame | None = None) -> list[dict]:
    if trades.is_empty():
        return []
    for column in (
        "target_hedge_ratio", "funding_fee", "exit_reason", "protection_trigger",
        "exit_class", "reopen_lock_pending",
        "protection_stop_x_price", "protection_take_profit_return",
        "protection_target_residual", "protection_stop_reason",
        "protection_max_holding_bars", "protection_max_holding_deadline_bar",
        "protection_rule", "protection_freeze_bars", "protection_freeze_until_bar",
    ):
        if column not in trades.columns:
            trades = trades.with_columns(pl.lit(None).alias(column))
    group_columns = ["group_id", "action", "ts", "position_id"]
    if "pair_id" in trades.columns:
        group_columns.append("pair_id")
    grouped = trades.group_by(group_columns).agg(
        pl.col("symbol").alias("symbols"),
        pl.col("side").alias("sides"),
        pl.col("quantity").alias("quantities"),
        pl.col("price").alias("prices"),
        pl.col("slippage").alias("slippages"),
        pl.col("notional").sum().alias("gross_notional"),
        pl.col("fee").sum().alias("fee"),
        pl.col("slippage").sum().alias("slippage"),
        pl.col("funding_fee").sum().alias("funding_fee"),
        pl.col("target_hedge_ratio").drop_nulls().first().alias("target_hedge_ratio"),
        pl.col("exit_reason").drop_nulls().first().alias("exit_reason"),
        pl.col("protection_trigger").drop_nulls().first().alias("protection_trigger"),
        pl.col("exit_class").drop_nulls().first().alias("exit_class"),
        pl.col("reopen_lock_pending").drop_nulls().first().alias("reopen_lock_pending"),
        pl.col("protection_stop_x_price").drop_nulls().first().alias("protection_stop_x_price"),
        pl.col("protection_take_profit_return").drop_nulls().first().alias("protection_take_profit_return"),
        pl.col("protection_target_residual").drop_nulls().first().alias("protection_target_residual"),
        pl.col("protection_stop_reason").drop_nulls().first().alias("protection_stop_reason"),
        pl.col("protection_max_holding_bars").drop_nulls().first().alias("protection_max_holding_bars"),
        pl.col("protection_max_holding_deadline_bar").drop_nulls().first().alias("protection_max_holding_deadline_bar"),
        pl.col("protection_rule").drop_nulls().first().alias("protection_rule"),
        pl.col("protection_freeze_bars").drop_nulls().first().alias("protection_freeze_bars"),
        pl.col("protection_freeze_until_bar").drop_nulls().first().alias("protection_freeze_until_bar"),
    ).sort("ts")

    rows = []
    for row in grouped.to_dicts():
        legs = []
        leg_details = []
        for symbol, side, quantity, price, slippage in zip(
            row["symbols"], row["sides"], row["quantities"], row["prices"], row["slippages"]
        ):
            legs.append(f"{symbol} {side}")
            quantity_value = _float(quantity) or 0.0
            price_value = _float(price)
            leg_details.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity_value,
                    "price": price_value,
                    "slippage": _float(slippage) or 0.0,
                    "reference_price": _reference_price(price_value, quantity_value, side, slippage),
                    "cashflow": _trade_cashflow(side, quantity_value, price_value),
                }
            )
        rows.append(
            {
                "ts": int(row["ts"]),
                "time": fmt_time(row["ts"]),
                "group_id": row["group_id"],
                "position_id": row["position_id"],
                "pair_id": row.get("pair_id"),
                "action": row["action"],
                "legs": " / ".join(legs),
                "leg_details": leg_details,
                "gross_notional": float(row["gross_notional"] or 0),
                "fee": float(row["fee"] or 0),
                "slippage": float(row["slippage"] or 0),
                "funding_fee": float(row["funding_fee"] or 0),
                "target_hedge_ratio": float(row["target_hedge_ratio"]) if row["target_hedge_ratio"] is not None else None,
                "exit_reason": row.get("exit_reason"),
                "protection_trigger": row.get("protection_trigger"),
                "exit_class": row.get("exit_class"),
                "reopen_lock_pending": row.get("reopen_lock_pending"),
                "protection_stop_x_price": _float(row.get("protection_stop_x_price")),
                "protection_take_profit_return": _float(row.get("protection_take_profit_return")),
                "protection_target_residual": _float(row.get("protection_target_residual")),
                "protection_stop_reason": row.get("protection_stop_reason"),
                "protection_max_holding_bars": _int(row.get("protection_max_holding_bars")),
                "protection_max_holding_deadline_bar": _int(row.get("protection_max_holding_deadline_bar")),
                "protection_rule": row.get("protection_rule"),
                "protection_freeze_bars": _int(row.get("protection_freeze_bars")),
                "protection_freeze_until_bar": _int(row.get("protection_freeze_until_bar")),
            }
        )
    return _attach_trade_signal_context(rows, signal)


def _trade_cashflow(side: str | None, quantity: float, price: float | None) -> float:
    """Return the cash change for a fill; buys spend cash and sells receive it."""
    if price is None or quantity == 0:
        return 0.0
    return -quantity * price if str(side).lower() == "buy" else quantity * price


def _complete_trade_rows(
    event_rows: list[dict],
    pair_curve: pl.DataFrame | None = None,
    funding_payments: pl.DataFrame | None = None,
) -> list[dict]:
    """Combine open/add/close events into one card per position."""
    if not event_rows:
        return []

    groups: dict[tuple, dict] = {}
    for event in sorted(event_rows, key=lambda item: (item.get("ts") or 0, item.get("group_id") or "")):
        pair_id = event.get("pair_id") or "_unknown"
        position_id = event.get("position_id")
        group_id = event.get("group_id")
        position_key = (pair_id, "position", position_id) if position_id else (pair_id, "group", group_id)
        item = groups.setdefault(
            position_key,
            {
                "trade_id": f"{pair_id}:{position_id or group_id}",
                "pair_id": event.get("pair_id"),
                "position_id": position_id,
                "group_id": group_id,
                "events": [],
            },
        )
        event["trade_id"] = item["trade_id"]
        item["events"].append(event)

    state_lookup = _pair_state_lookup(pair_curve)
    funding_rows = _funding_rows(funding_payments)
    result = []
    for item in groups.values():
        events = sorted(item["events"], key=lambda row: row.get("ts") or 0)
        opens = [row for row in events if row.get("action") in {"open", "add"}]
        closes = [row for row in events if row.get("action") == "close"]
        if not opens:
            continue

        open_ts = min(int(row["ts"]) for row in opens)
        close_ts = max((int(row["ts"]) for row in closes), default=None)
        open_notional = sum(float(row.get("gross_notional") or 0.0) for row in opens)
        close_notional = sum(float(row.get("gross_notional") or 0.0) for row in closes)
        fee = sum(float(row.get("fee") or 0.0) for row in events)
        slippage = sum(float(row.get("slippage") or 0.0) for row in events)
        trade_funding = sum(float(row.get("funding_fee") or 0.0) for row in events)
        if close_ts is not None:
            trade_funding += _trade_funding_from_position(
                events, funding_rows, open_ts, close_ts
            )

        gross_pnl = sum(
            _reference_cashflow(detail, event.get("slippage"), event.get("leg_details"))
            for event in events
            for detail in event.get("leg_details", [])
        )
        # For a closed position, costs are deducted explicitly from the no-cost PnL.
        net_pnl = gross_pnl - fee - slippage - trade_funding if close_ts is not None else None
        # Orders execute on the following tradable bar.  Scenario parameters
        # must therefore come from the signal timestamp, not a model update
        # that may happen on the fill bar itself.
        open_signal_ts = _first_value(opens, "signal_ts")
        close_signal_ts = _last_value(closes, "signal_ts")
        open_state = _state_at(
            state_lookup,
            item.get("pair_id"),
            int(open_signal_ts) if open_signal_ts is not None else open_ts,
        )
        close_state = (
            _state_at(
                state_lookup,
                item.get("pair_id"),
                int(close_signal_ts) if close_signal_ts is not None else close_ts,
            )
            if close_ts is not None else None
        )

        result.append(
            {
                "trade_id": item["trade_id"],
                "pair_id": item.get("pair_id"),
                "position_id": item.get("position_id"),
                "group_id": item.get("group_id"),
                "ts": open_ts,
                "open_ts": open_ts,
                "close_ts": close_ts,
                "open_time": fmt_time(open_ts),
                "close_time": fmt_time(close_ts) if close_ts is not None else None,
                "holding_minutes": (close_ts - open_ts) / 60000.0 if close_ts is not None else None,
                "direction": opens[0].get("legs"),
                "target_hedge_ratio": opens[0].get("target_hedge_ratio"),
                "open_legs": _join_event_legs(opens),
                "close_legs": _join_event_legs(closes) if closes else None,
                "open_zscore": _first_value(opens, "signal_zscore"),
                "close_zscore": _last_value(closes, "signal_zscore"),
                "exit_reason": _last_value(closes, "exit_reason"),
                "protection_trigger": _last_value(closes, "protection_trigger"),
                "exit_class": _last_value(closes, "exit_class"),
                "reopen_lock_pending": _last_value(closes, "reopen_lock_pending"),
                "protection_stop_x_price": _last_value(opens, "protection_stop_x_price"),
                "protection_take_profit_return": _last_value(opens, "protection_take_profit_return"),
                "protection_target_residual": _last_value(opens, "protection_target_residual"),
                "protection_stop_reason": _last_value(opens, "protection_stop_reason"),
                "protection_max_holding_bars": _int(_first_value(opens, "protection_max_holding_bars")),
                "protection_max_holding_deadline_bar": _int(_first_value(opens, "protection_max_holding_deadline_bar")),
                "protection_rule": _last_value(closes, "protection_rule"),
                "protection_freeze_bars": _int(_last_value(closes, "protection_freeze_bars")),
                "protection_freeze_until_bar": _int(_last_value(closes, "protection_freeze_until_bar")),
                "open_fill_zscore": _first_value(opens, "fill_zscore"),
                "close_fill_zscore": _last_value(closes, "fill_zscore"),
                "open_notional": open_notional,
                "close_notional": close_notional,
                "gross_notional": open_notional + close_notional,
                "fee": fee,
                "slippage": slippage,
                "funding_fee": trade_funding,
                "gross_pnl": gross_pnl if close_ts is not None else None,
                "gross_return": gross_pnl / open_notional if close_ts is not None and open_notional else None,
                "net_pnl": net_pnl,
                "net_return": net_pnl / open_notional if net_pnl is not None and open_notional else None,
                "open_alpha": _coalesce(_first_value(opens, "signal_alpha"), _float(open_state.get("alpha")) if open_state else None),
                "open_beta": _coalesce(_first_value(opens, "signal_spread_beta", "signal_beta"), _float(open_state.get("beta")) if open_state else None),
                "open_spread_mean": _coalesce(_first_value(opens, "signal_spread_mean"), _float(open_state.get("spread_mean")) if open_state else None),
                "open_spread_std": _coalesce(_first_value(opens, "signal_spread_std"), _float(open_state.get("spread_std")) if open_state else None),
                "close_alpha": _coalesce(_last_value(closes, "signal_alpha"), _float(close_state.get("alpha")) if close_state else None),
                "close_beta": _coalesce(_last_value(closes, "signal_spread_beta", "signal_beta"), _float(close_state.get("beta")) if close_state else None),
            }
        )

    return sorted(result, key=lambda row: (row.get("open_ts") or 0, row.get("trade_id") or ""))


def _trade_funding_from_position(
    events: list[dict], funding_rows: list[dict], open_ts: int, close_ts: int
) -> float:
    """Rebuild one position's funding without charging other Pair positions."""
    quantities: dict[tuple[str | None, str], float] = {}
    ordered_events = sorted(events, key=lambda row: int(row.get("ts") or 0))
    event_index = 0
    total = 0.0
    for funding in sorted(funding_rows, key=lambda row: int(row.get("ts") or 0)):
        funding_ts = int(funding.get("ts") or 0)
        if funding_ts < open_ts or funding_ts > close_ts:
            continue
        # Funding is settled before fills carrying the same timestamp.
        while event_index < len(ordered_events) and int(ordered_events[event_index].get("ts") or 0) < funding_ts:
            event = ordered_events[event_index]
            for detail in event.get("leg_details", []):
                symbol = str(detail.get("symbol") or "")
                exchange = detail.get("exchange") or event.get("exchange")
                quantity = float(detail.get("quantity") or 0.0)
                signed = quantity if str(detail.get("side") or "").lower() == "buy" else -quantity
                key = (exchange, symbol)
                quantities[key] = quantities.get(key, 0.0) + signed
            event_index += 1
        symbol = str(funding.get("symbol") or "")
        exchange = funding.get("exchange")
        quantity = sum(
            value for (item_exchange, item_symbol), value in quantities.items()
            if item_symbol == symbol
            and (exchange is None or item_exchange is None or item_exchange == exchange)
        )
        rate = _float(funding.get("funding_rate"))
        mark = _float(funding.get("mark_price"))
        if rate is not None and mark is not None:
            total += quantity * mark * rate
    return total


def _join_event_legs(events: list[dict]) -> str:
    values = []
    for event in events:
        details = event.get("leg_details") or []
        if not details:
            if event.get("legs"):
                values.append(event["legs"])
            continue
        legs = []
        for detail in details:
            symbol = detail.get("symbol") or "?"
            side = detail.get("side") or "?"
            quantity = _float(detail.get("quantity"))
            price = _float(detail.get("price"))
            reference_price = _float(detail.get("reference_price"))
            legs.append(
                f"{symbol} {side} 数量 {_format_number(quantity)} 成交价 {_format_number(price)} "
                f"基准价 {_format_number(reference_price)}"
            )
        if legs:
            values.append(" / ".join(legs))
    return "\n".join(values) or None


def _format_number(value) -> str:
    value = _float(value)
    if value is None:
        return "-"
    return f"{value:.8g}"


def _coalesce(*values):
    return next((value for value in values if value is not None), None)


def _first_value(rows: list[dict], *keys: str):
    for row in rows:
        for key in keys:
            if row.get(key) is not None:
                return row.get(key)
    return None


def _last_value(rows: list[dict], *keys: str):
    for row in reversed(rows):
        for key in keys:
            if row.get(key) is not None:
                return row.get(key)
    return None


def _reference_cashflow(detail: dict, _event_slippage, _all_details) -> float:
    price = detail.get("price")
    quantity = float(detail.get("quantity") or 0.0)
    if price is None or quantity == 0:
        return 0.0
    # Exchange slippage is an absolute price difference times quantity.
    # Reconstruct the bar-open reference price so slippage is not included in gross PnL.
    reference_price = _reference_price(price, quantity, detail.get("side"), detail.get("slippage"))
    return _trade_cashflow(detail.get("side"), quantity, reference_price)


def _reference_price(price, quantity, side, slippage) -> float:
    price = _float(price)
    quantity = abs(float(quantity or 0.0))
    if price is None or quantity == 0:
        return price
    slippage_per_unit = float(slippage or 0.0) / quantity
    return price - slippage_per_unit if str(side or "").lower() == "buy" else price + slippage_per_unit


def _funding_rows(frame: pl.DataFrame | None) -> list[dict]:
    if frame is None or frame.is_empty():
        return []
    return frame.to_dicts()


def _pair_state_lookup(pair_curve: pl.DataFrame | None) -> dict[str, list[dict]]:
    if pair_curve is None or pair_curve.is_empty() or "pair_id" not in pair_curve.columns:
        return {}
    result: dict[str, list[dict]] = {}
    for row in pair_curve.sort("ts").to_dicts():
        result.setdefault(str(row.get("pair_id")), []).append(row)
    return result


def _state_at(lookup: dict[str, list[dict]], pair_id: str | None, ts: int | None) -> dict | None:
    if not pair_id or ts is None:
        return None
    rows = lookup.get(str(pair_id), [])
    state = None
    for row in rows:
        row_ts = row.get("ts")
        if row_ts is None or int(row_ts) > ts:
            break
        state = row
    return state


def _attach_trade_signal_context(rows: list[dict], signal: pl.DataFrame | None) -> list[dict]:
    if not rows or signal is None or signal.is_empty() or "ts" not in signal.columns:
        return rows

    from bisect import bisect_left

    signal = signal.sort("ts")
    ts_values = [int(value) for value in signal["ts"].to_list()]
    available_columns = set(signal.columns)
    pair_ids = sorted({row.get("pair_id") for row in rows if row.get("pair_id")})
    signal_by_pair = {}
    for pair_id in pair_ids:
        prefix = f"{pair_id}_"
        wanted = [
            "zscore", "action", "side", "alpha", "beta", "spread_beta",
            "spread_mean", "spread_std", "latest_spread",
            "last_model_update_index",
        ]
        series = {
            name: signal[f"{prefix}{name}"]
            for name in wanted
            if f"{prefix}{name}" in available_columns
        }
        if not series:
            continue
        signal_by_pair[pair_id] = series

    for row in rows:
        pair_id = row.get("pair_id")
        if pair_id not in signal_by_pair:
            continue

        series = signal_by_pair[pair_id]
        trade_ts = int(row["ts"])
        prev_idx = bisect_left(ts_values, trade_ts) - 1
        fill_idx = bisect_left(ts_values, trade_ts)

        if prev_idx >= 0:
            signal_ts = ts_values[prev_idx]
            row["signal_ts"] = signal_ts
            row["signal_time"] = fmt_time(signal_ts)
            row["signal_zscore"] = _float(
                series["zscore"][prev_idx] if "zscore" in series else None
            )
            row["signal_action"] = (
                series["action"][prev_idx] if "action" in series else None
            )
            row["signal_side"] = (
                series["side"][prev_idx] if "side" in series else None
            )
            for name in [
                "alpha", "beta", "spread_beta", "spread_mean", "spread_std",
                "latest_spread", "last_model_update_index",
            ]:
                row[f"signal_{name}"] = _float(
                    series[name][prev_idx] if name in series else None
                )

        if fill_idx < len(ts_values) and ts_values[fill_idx] == trade_ts:
            row["fill_zscore"] = _float(
                series["zscore"][fill_idx] if "zscore" in series else None
            )

    return rows


def threshold_rows(run_dir: Path) -> list[dict]:
    config_path = run_dir / "config.yaml"
    default_entry_z = 2.0
    default_exit_z = 0.5
    entry_rule_method = "fixed_z"
    two_stage_levels = None
    two_stage_trigger_z = None
    two_stage_entry_z = None

    if not config_path.exists():
        entry_z = default_entry_z
        exit_z = default_exit_z
    else:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        active_setup = raw.get("active_setup")
        setup = raw.get("setups", {}).get(active_setup, {})
        pipeline_signal = (setup.get("pipeline") or {}).get("signal")
        signal = pipeline_signal or {}
        entry_z = float(signal.get("entry_z", default_entry_z))
        exit_z = float(signal.get("exit_z", default_exit_z))
        entry_rule_method = signal.get("entry_rule_method", "fixed_z")
        signal_method = str(signal.get("method", "simple_zscore"))
        two_stage_levels = signal.get("two_stage_levels")
        if _signal_uses_two_stage(signal_method, signal.get("two_stage_enabled")):
            two_stage_trigger_z = signal.get("two_stage_trigger_z")
            two_stage_entry_z = signal.get("two_stage_entry_z", entry_z)

    if exit_z > 0.0:
        rows = [
            {"value": exit_z, "label": f"平仓 +{exit_z:.2f}", "kind": "exit"},
            {"value": -exit_z, "label": f"平仓 -{exit_z:.2f}", "kind": "exit"},
        ]
    elif exit_z == 0.0:
        rows = [{"value": 0.0, "label": "反转平仓 0.00", "kind": "exit"}]
    else:
        reversal = abs(exit_z)
        rows = [
            {"value": reversal, "label": f"反向平仓 +{reversal:.2f}", "kind": "exit"},
            {"value": -reversal, "label": f"反向平仓 -{reversal:.2f}", "kind": "exit"},
        ]
    if two_stage_levels:
        for trigger_z, entry_level_z in _threshold_two_stage_levels(two_stage_levels):
            rows.extend(
                [
                    {"value": trigger_z, "label": f"触发 +{trigger_z:.2f}", "kind": "trigger"},
                    {"value": -trigger_z, "label": f"触发 -{trigger_z:.2f}", "kind": "trigger"},
                    {"value": entry_level_z, "label": f"回落开仓 +{entry_level_z:.2f}", "kind": "entry"},
                    {"value": -entry_level_z, "label": f"回落开仓 -{entry_level_z:.2f}", "kind": "entry"},
                ]
            )
        return rows
    if two_stage_trigger_z is not None:
        trigger_z = abs(float(two_stage_trigger_z))
        entry_level_z = abs(float(two_stage_entry_z if two_stage_entry_z is not None else entry_z))
        rows.extend(
            [
                {"value": trigger_z, "label": f"触发 +{trigger_z:.2f}", "kind": "trigger"},
                {"value": -trigger_z, "label": f"触发 -{trigger_z:.2f}", "kind": "trigger"},
                {"value": entry_level_z, "label": f"开仓 +{entry_level_z:.2f}", "kind": "entry"},
                {"value": -entry_level_z, "label": f"开仓 -{entry_level_z:.2f}", "kind": "entry"},
            ]
        )
        return rows
    if entry_rule_method != "rolling_quantile":
        rows.extend(
            [
                {"value": entry_z, "label": f"开仓 +{entry_z:.2f}", "kind": "entry"},
                {"value": -entry_z, "label": f"开仓 -{entry_z:.2f}", "kind": "entry"},
            ]
        )
    return rows


def _signal_uses_two_stage(signal_method: str, two_stage_enabled) -> bool:
    """True only when the configured signal really runs the two-stage machine.

    Legacy configs carry ``two_stage_trigger_z`` even with the two-stage switch
    off; drawing those thresholds would put phantom lines on the z-score chart.
    """
    method = str(signal_method or "").strip().lower()
    if method == "zscore_reversion_two_stage":
        return True
    if method == "simple_zscore":
        return False
    return bool(two_stage_enabled)


def _threshold_two_stage_levels(levels) -> list[tuple[float, float]]:
    parsed = []
    for item in levels or []:
        if isinstance(item, dict):
            trigger_z = item.get("trigger_z", item.get("trigger"))
            entry_z = item.get("entry_z", item.get("entry"))
        else:
            trigger_z, entry_z = item
        parsed.append((abs(float(trigger_z)), abs(float(entry_z))))
    return sorted(parsed, key=lambda pair: pair[0], reverse=True)


def _sort_if_has_column(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    if frame.is_empty() or column not in frame.columns:
        return frame
    return frame.sort(column)


HOLDING_BINS = [
    ("0-30分钟", 0, 30),
    ("30-60分钟", 30, 60),
    ("1-2小时", 60, 120),
    ("2-4小时", 120, 240),
    ("4-8小时", 240, 480),
    ("8-24小时", 480, 1440),
    ("1-3天", 1440, 4320),
    ("3-7天", 4320, 10080),
    ("7-14天", 10080, 20160),
    ("14-30天", 20160, 43200),
    ("30天以上", 43200, None),
]


def _holding_bin_index(minutes: float) -> int:
    for idx, (_label, low, high) in enumerate(HOLDING_BINS):
        if high is None or minutes < high:
            return idx
    return len(HOLDING_BINS) - 1


def holding_time_distribution(trades: pl.DataFrame) -> dict:
    """按 position 聚合持仓时间（最早 open -> 最晚 close），输出分桶分布。"""
    empty = {
        "bins": [],
        "closed_count": 0,
        "open_count": 0,
        "avg_minutes": None,
        "median_minutes": None,
        "min_minutes": None,
        "max_minutes": None,
    }
    if trades.is_empty() or "ts" not in trades.columns or "action" not in trades.columns:
        return empty

    frame = trades
    if "position_id" not in frame.columns:
        frame = frame.with_columns(pl.lit(None).alias("position_id"))
    key_column = "position_id" if frame["position_id"].drop_nulls().len() > 0 else "group_id"
    if key_column not in frame.columns:
        return empty

    positions = frame.group_by(key_column).agg(
        pl.col("ts").filter(pl.col("action") == "open").min().alias("open_ts"),
        pl.col("ts").filter(pl.col("action") == "close").max().alias("close_ts"),
    )
    closed = positions.filter(
        pl.col("open_ts").is_not_null() & pl.col("close_ts").is_not_null()
    )
    open_count = positions.filter(
        pl.col("open_ts").is_not_null() & pl.col("close_ts").is_null()
    ).height
    if closed.is_empty():
        empty["open_count"] = open_count
        return empty

    counts = [0] * len(HOLDING_BINS)
    holding_minutes = []
    for row in closed.to_dicts():
        minutes = (row["close_ts"] - row["open_ts"]) / 60000.0
        holding_minutes.append(minutes)
        counts[_holding_bin_index(minutes)] += 1

    total = sum(counts)
    bins = []
    cumulative = 0
    for (label, _low, _high), count in zip(HOLDING_BINS, counts):
        cumulative += count
        bins.append(
            {
                "label": label,
                "count": count,
                "pct": round(count / total * 100, 2) if total else 0.0,
                "cum_pct": round(cumulative / total * 100, 2) if total else 0.0,
            }
        )

    holding_sorted = sorted(holding_minutes)
    median = holding_sorted[len(holding_sorted) // 2] if len(holding_sorted) % 2 else (
        (holding_sorted[len(holding_sorted) // 2 - 1] + holding_sorted[len(holding_sorted) // 2]) / 2
    )
    return {
        "bins": bins,
        "closed_count": total,
        "open_count": open_count,
        "avg_minutes": round(sum(holding_minutes) / len(holding_minutes), 1),
        "median_minutes": round(median, 1),
        "min_minutes": round(min(holding_minutes), 1),
        "max_minutes": round(max(holding_minutes), 1),
    }


def _portfolio_points(equity: pl.DataFrame, position: pl.DataFrame, trades: pl.DataFrame, max_points: int) -> list[dict]:
    timeline = _sample_timeline_with_trade_context(equity.select("ts"), trades, max_points)
    merged = timeline.join(equity.select(["ts", "equity"]), on="ts", how="left")
    if not position.is_empty() and "ts" in position.columns:
        merged = merged.join(position, on="ts", how="left")
    merged = merged.sort("ts")
    points = []
    peak: float | None = None
    for row in merged.to_dicts():
        equity = _float(row.get("equity"))
        if equity is not None and (peak is None or equity > peak):
            peak = equity
        points.append(
            {
                "ts": int(row["ts"]),
                "time": fmt_time(row["ts"]),
                "equity": equity,
                "gross_exposure_ratio": _float(row.get("gross_exposure_ratio")),
                "margin_deficit": _float(row.get("margin_deficit")),
                "forced_deleveraging_triggered": _float(
                    row.get("forced_deleveraging_triggered")
                ),
                "peak": peak,
            }
        )
    return points


def _pair_summary_position_frame(run_dir: Path, pair_defs: list[dict]) -> pl.DataFrame:
    path = run_dir / "position_curve.csv"
    if not path.exists():
        return pl.DataFrame()
    try:
        header = pl.read_csv(path, n_rows=0).columns
    except Exception:
        return pl.DataFrame()
    wanted = required_position_columns(pair_defs)
    columns = [column for column in wanted if column in header]
    if "ts" not in columns:
        return pl.DataFrame()
    try:
        return pl.read_csv(path, columns=columns, infer_schema_length=10000)
    except Exception:
        return pl.DataFrame()


def _configured_pairs(run_dir: Path) -> list[dict]:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return []

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    setup = raw.get("setups", {}).get(raw.get("active_setup"), {})
    pairs = []
    for item in setup.get("pairs") or []:
        if not item.get("enabled", True):
            continue
        x_symbol = item.get("x_symbol", item.get("long_symbol"))
        y_symbol = item.get("y_symbol", item.get("short_symbol"))
        if not x_symbol or not y_symbol:
            continue
        pairs.append({
            "pair_id": item.get("pair_id") or f"{x_symbol}_{y_symbol}",
            "x_symbol": x_symbol,
            "y_symbol": y_symbol,
        })

    if pairs:
        return pairs

    pair = setup.get("pair", raw.get("pair", {}))
    x_symbol = pair.get("x_symbol", pair.get("long_symbol"))
    y_symbol = pair.get("y_symbol", pair.get("short_symbol"))
    if x_symbol and y_symbol:
        pairs.append({
            "pair_id": pair.get("pair_id") or f"{x_symbol}_{y_symbol}",
            "x_symbol": x_symbol,
            "y_symbol": y_symbol,
        })
    return pairs


def _estimator_review_parameters(run_dir: Path) -> dict:
    """Read estimator settings needed to provide chart context in Pair review."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return {
            "model_lookback_bars": None,
            "model_update_interval_bars": None,
            "regression_method": None,
            "exit_z": 0.5,
        }
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, TypeError, yaml.YAMLError):
        return {
            "model_lookback_bars": None,
            "model_update_interval_bars": None,
            "regression_method": None,
            "exit_z": 0.5,
        }
    setup = (raw.get("setups") or {}).get(raw.get("active_setup"), {}) or {}
    pipeline = setup.get("pipeline") or {}
    estimator = pipeline.get("estimator") or {}
    signal = pipeline.get("signal") or {}
    return {
        "model_lookback_bars": _float(estimator.get("model_lookback_bars")),
        "model_update_interval_bars": _float(estimator.get("model_update_interval_bars")),
        "regression_method": estimator.get("regression_method"),
        "exit_z": _coalesce(_float(signal.get("exit_z")), 0.5),
    }


def _raw_pair_prices(
    trade_ts: int | None,
    x_symbol: str,
    y_symbol: str,
    x_prices: dict,
    y_prices: dict,
    leg_events: list[dict] | None = None,
) -> tuple[float | None, float | None, float | None]:
    """Return X price, Y price and signed raw spread (Y-X) at a trade timestamp."""
    if trade_ts is None:
        return None, None, None
    x_price = None
    y_price = None
    # Use the exact execution reference price for this event.  Market-data
    # close is only a fallback because orders are filled against the next bar's
    # open/reference price.
    if leg_events:
        for event in leg_events:
            if int(event.get("ts") or -1) != int(trade_ts):
                continue
            for detail in event.get("leg_details", []):
                symbol = detail.get("symbol")
                price = _float(detail.get("reference_price")) or _float(detail.get("price"))
                if symbol == x_symbol:
                    x_price = price
                elif symbol == y_symbol:
                    y_price = price
    if x_price is None:
        x_price = _float(x_prices.get(int(trade_ts)))
    if y_price is None:
        y_price = _float(y_prices.get(int(trade_ts)))
    spread = y_price - x_price if x_price is not None and y_price is not None else None
    return x_price, y_price, spread


SCENARIO_X_MOVES = (-0.20, -0.10, 0.0, 0.10, 0.20)


def _trade_scenario_analysis(
    trade: dict,
    open_events: list[dict],
    x_symbol: str,
    y_symbol: str,
    regression_method: str | None,
    exit_z: float,
) -> dict:
    """Build entry-time X-price scenarios without replaying the backtest."""
    method = str(regression_method or "").lower()
    result = {
        "available": False,
        "regression_method": method or None,
        "exit_z": float(exit_z),
        "scenarios": [],
    }
    if method not in {"price", "log_price"}:
        result["reason"] = "当前回归类型不支持价格变动情景。"
        return result

    alpha = _float(trade.get("open_alpha"))
    beta = _float(trade.get("open_beta"))
    spread_mean = _float(trade.get("open_spread_mean"))
    spread_std = _float(trade.get("open_spread_std"))
    entry_x = _float(trade.get("open_x_price"))
    entry_y = _float(trade.get("open_y_price"))
    qx = _signed_event_quantity(open_events, x_symbol)
    qy = _signed_event_quantity(open_events, y_symbol)
    required = (alpha, beta, spread_mean, spread_std, entry_x, entry_y, qx, qy)
    if any(value is None for value in required) or spread_std <= 0 or entry_x <= 0 or entry_y <= 0:
        result["reason"] = "缺少开仓模型状态、价格或两腿成交数量，无法计算情景。"
        return result
    if abs(qx) <= 1e-15 or abs(qy) <= 1e-15:
        result["reason"] = "开仓两腿成交数量不完整，无法计算情景。"
        return result
    entry_z = _coalesce(trade.get("open_zscore"), trade.get("open_fill_zscore"))
    entry_z = _float(entry_z)
    if entry_z is not None and abs(entry_z) > 1e-12:
        residual_sign = 1.0 if entry_z > 0 else -1.0
    else:
        # Positive residual positions normally buy X and sell Y.
        residual_sign = 1.0 if qx > 0 else -1.0
    target_z = 0.0 if float(exit_z) == 0.0 else residual_sign * float(exit_z)
    entry_gross_notional = abs(qx * entry_x) + abs(qy * entry_y)
    if entry_gross_notional <= 1e-12:
        result["reason"] = "开仓总名义金额为零，无法计算收益率。"
        return result

    target_residual = spread_mean + target_z * spread_std

    def scenario_at(x_move: float) -> dict | None:
        x_price = entry_x * (1.0 + float(x_move))
        if x_price <= 0:
            return None
        try:
            if method == "log_price":
                y_price = exp(alpha + beta * log(x_price) + target_residual)
            else:
                y_price = alpha + beta * x_price + target_residual
        except (OverflowError, ValueError):
            return None
        if not isfinite(y_price) or y_price <= 0:
            return None
        gross_pnl = qx * (x_price - entry_x) + qy * (y_price - entry_y)
        return {
            "x_move": float(x_move),
            "x_price": x_price,
            "y_target_price": y_price,
            "gross_pnl": gross_pnl,
            "gross_return": gross_pnl / entry_gross_notional,
        }

    def scenario_payload() -> dict | None:
        scenarios = []
        invalid_x_moves = []
        for move in SCENARIO_X_MOVES:
            row = scenario_at(move)
            if row is None:
                invalid_x_moves.append(float(move))
            else:
                scenarios.append(row)
        if not scenarios:
            return None

        dense_scenarios = []
        for index in range(401):
            row = scenario_at(-0.20 + index * 0.001)
            if row is not None:
                dense_scenarios.append(row)
        if not dense_scenarios:
            return None

        range_min = min(dense_scenarios, key=lambda row: row["gross_return"])
        range_max = max(dense_scenarios, key=lambda row: row["gross_return"])
        return {
            "target_residual": target_residual,
            "scenarios": scenarios,
            "invalid_x_moves": invalid_x_moves,
            "summary": {
                "zero_x": scenario_at(0.0),
                "range_min": range_min,
                "range_max": range_max,
                "range_fully_valid": len(dense_scenarios) == 401,
                "all_profitable": (
                    all(row["gross_return"] > 0.0 for row in dense_scenarios)
                    if len(dense_scenarios) == 401 else None
                ),
                "break_even_x_move": _scenario_break_even(dense_scenarios),
            },
        }

    first_payload = scenario_payload()
    if first_payload is None:
        result["reason"] = "所有价格情景均产生无效理论Y价格，无法计算情景表。"
        return result

    result.update({
        "available": True,
        "target_z": target_z,
        "target_residual": target_residual,
        "entry_x_price": entry_x,
        "entry_y_price": entry_y,
        "x_quantity": qx,
        "y_quantity": qy,
        "entry_gross_notional": entry_gross_notional,
        "scenarios": first_payload["scenarios"],
        "invalid_x_moves": first_payload["invalid_x_moves"],
        "assumption": "假设价格关系回到目标Z",
        "summary": first_payload["summary"],
    })

    close_x = _float(trade.get("close_x_price"))
    close_y = _float(trade.get("close_y_price"))
    if close_x is not None and close_x > 0:
        actual_move = close_x / entry_x - 1.0
        exact = scenario_at(actual_move)
        lower, upper, band = _scenario_neighbors(
            first_payload["scenarios"], actual_move
        )
        result["actual"] = {
            "x_move": actual_move,
            "x_price": close_x,
            "y_price": close_y,
            "scenario_band": band,
            "lower_scenario": lower,
            "upper_scenario": upper,
            "theoretical_at_actual_x": exact,
            "actual_gross_return": _float(trade.get("gross_return")),
            "actual_net_return": _float(trade.get("net_return")),
            "actual_gross_pnl": _float(trade.get("gross_pnl")),
            "actual_net_pnl": _float(trade.get("net_pnl")),
        }
        if exact is not None and result["actual"]["actual_gross_return"] is not None:
            result["actual"]["gross_return_gap"] = (
                result["actual"]["actual_gross_return"] - exact["gross_return"]
            )
    return result


def _signed_event_quantity(events: list[dict], symbol: str) -> float | None:
    quantity = 0.0
    found = False
    for event in events:
        for detail in event.get("leg_details") or []:
            if detail.get("symbol") != symbol:
                continue
            value = _float(detail.get("quantity"))
            if value is None:
                continue
            quantity += value if str(detail.get("side") or "").lower() == "buy" else -value
            found = True
    return quantity if found else None


def _scenario_neighbors(
    scenarios: list[dict], actual_move: float
) -> tuple[dict | None, dict | None, str]:
    ordered = sorted(scenarios, key=lambda row: row["x_move"])
    if actual_move <= ordered[0]["x_move"]:
        return None, ordered[0], f"≤ {ordered[0]['x_move']:+.0%}"
    if actual_move >= ordered[-1]["x_move"]:
        return ordered[-1], None, f"≥ {ordered[-1]['x_move']:+.0%}"
    for lower, upper in zip(ordered, ordered[1:]):
        if lower["x_move"] <= actual_move <= upper["x_move"]:
            return lower, upper, f"{lower['x_move']:+.0%} 至 {upper['x_move']:+.0%}"
    return None, None, "-"


def _scenario_break_even(scenarios: list[dict]) -> float | None:
    """Return the zero-PnL X move closest to zero within the scenario range."""
    roots: list[float] = []
    ordered = sorted(scenarios, key=lambda row: row["x_move"])
    for row in ordered:
        if abs(float(row["gross_return"])) <= 1e-12:
            roots.append(float(row["x_move"]))
    for left, right in zip(ordered, ordered[1:]):
        left_value = float(left["gross_return"])
        right_value = float(right["gross_return"])
        if left_value == 0.0 or right_value == 0.0 or left_value * right_value > 0.0:
            continue
        span = right_value - left_value
        if abs(span) <= 1e-15:
            continue
        weight = -left_value / span
        roots.append(float(left["x_move"]) + weight * (float(right["x_move"]) - float(left["x_move"])))
    return min(roots, key=abs) if roots else None


def _percentile(values: list[float], probability: float) -> float | None:
    finite = sorted(float(value) for value in values if value is not None and isfinite(float(value)))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    position = max(0.0, min(1.0, float(probability))) * (len(finite) - 1)
    lower = int(position)
    upper = min(lower + 1, len(finite) - 1)
    fraction = position - lower
    return finite[lower] * (1.0 - fraction) + finite[upper] * fraction


def _distribution(values: list[float]) -> dict:
    clean = [float(value) for value in values if value is not None and isfinite(float(value))]
    return {
        "count": len(clean),
        "p10": _percentile(clean, 0.10),
        "median": _percentile(clean, 0.50),
        "p90": _percentile(clean, 0.90),
    }


def _pair_scenario_summary(trades: list[dict]) -> dict | None:
    analyses = [
        trade.get("scenario_analysis")
        for trade in trades
        if isinstance(trade.get("scenario_analysis"), dict)
        and trade["scenario_analysis"].get("available")
    ]
    if not analyses:
        return None
    zero_returns = []
    range_min_returns = []
    range_max_returns = []
    for analysis in analyses:
        summary = analysis.get("summary") or {}
        zero = summary.get("zero_x") or {}
        minimum = summary.get("range_min") or {}
        maximum = summary.get("range_max") or {}
        if zero.get("gross_return") is not None:
            zero_returns.append(float(zero["gross_return"]))
        if minimum.get("gross_return") is not None:
            range_min_returns.append(float(minimum["gross_return"]))
        if maximum.get("gross_return") is not None:
            range_max_returns.append(float(maximum["gross_return"]))

    closed = [trade for trade in trades if trade.get("net_return") is not None]
    actual_gross = [float(trade["gross_return"]) for trade in closed if trade.get("gross_return") is not None]
    actual_net = [float(trade["net_return"]) for trade in closed if trade.get("net_return") is not None]
    return {
        "scenario_count": len(analyses),
        "closed_count": len(closed),
        "range_min": _distribution(range_min_returns),
        "zero_x": _distribution(zero_returns),
        "range_max": _distribution(range_max_returns),
        "actual_gross": _distribution(actual_gross),
        "actual_net": _distribution(actual_net),
        "theoretical_positive_rate": (
            sum(value > 0.0 for value in zero_returns) / len(zero_returns)
            if zero_returns else None
        ),
        "cost_coverage_rate": (
            sum(value > 0.0 for value in actual_net) / len(actual_net)
            if actual_net else None
        ),
    }


def _sample_timeline_with_trade_context(
    timeline: pl.DataFrame,
    trades: pl.DataFrame,
    max_points: int,
) -> pl.DataFrame:
    if timeline.is_empty() or "ts" not in timeline.columns:
        return timeline

    # Review charts must retain every available bar. ``max_points`` remains in
    # the signature for CLI compatibility but is intentionally ignored.
    base = timeline
    if trades.is_empty() or "ts" not in trades.columns:
        return base

    trade_ts = (
        trades.select(pl.col("ts").cast(pl.Int64, strict=False).alias("ts"))
        .drop_nulls("ts")
        .unique()
        .sort("ts")
    )
    if trade_ts.is_empty():
        return base
    context_ts = (
        pl.concat(
            [
                trade_ts,
                trade_ts.with_columns((pl.col("ts") - 60_000).alias("ts")),
            ],
            how="vertical",
        )
        .unique()
        .join(timeline.select("ts"), on="ts", how="inner")
    )
    return pl.concat([base, context_ts], how="vertical").unique().sort("ts")


def _symbol_close_frame(run_dir: Path, symbol: str, output_name: str) -> pl.DataFrame:
    data_path = _symbol_data_path(run_dir, symbol)
    if data_path is None:
        return pl.DataFrame({
            "ts": pl.Series("ts", [], dtype=pl.Int64),
            output_name: pl.Series(output_name, [], dtype=pl.Float64),
        })
    try:
        frame = load_csv_data(data_path)
    except Exception:
        return pl.DataFrame({
            "ts": pl.Series("ts", [], dtype=pl.Int64),
            output_name: pl.Series(output_name, [], dtype=pl.Float64),
        })
    return frame.select(["ts", "close"]).rename({"close": output_name})


def _symbol_data_path(run_dir: Path, symbol: str) -> Path | None:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return None
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    path = (raw.get("data", {}).get("symbols", {}).get(symbol, {}) or {}).get("path")
    if not path:
        return None
    data_path = Path(path)
    if data_path.is_absolute():
        candidates = [data_path]
    else:
        candidates = [(ROOT / data_path).resolve()]

    # Older configs point to data/SYMBOL_1m.csv while the current data layout
    # stores bars under data/SYMBOL/SYMBOL-1m.csv.
    candidates.extend(
        [
            ROOT / "data" / symbol / f"{symbol}-1m.csv",
            ROOT / "data" / symbol / f"{symbol}_1m.csv",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def _float(value) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
        if not isfinite(value):
            return None
        return value
    except (TypeError, ValueError):
        return None


def _int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if isfinite(value):
            return value
        if value > 0:
            return "Infinity"
        if value < 0:
            return "-Infinity"
        return None
    return value


def export_trade_review_html(
    run_dir: Path,
    output: Path | None = None,
    max_points: int = 2000,
    max_pairs: int = 20,
) -> Path:
    """生成组合总览页 + 每个 Pair 独立复盘页（v2 结构）。

    - trade_review.html              组合总览（不加载任何 Pair 分钟明细）
    - trade_review_pairs/<pair>.html 每个 Pair 独立页面
    - trade_review_assets/           共享 review.css / review.js
    - trade_review_data/             组合数据 + Pair 月度分块 + manifest

    max_points / max_pairs 保留为旧命令兼容参数，不再裁剪任何数据。
    """
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    output = output or (run_dir / "trade_review.html")
    if not output.is_absolute():
        output = ROOT / output
    out_dir = output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    review_started = perf_counter()
    logger.info("Trade Review：开始读取回测产物，run_dir={}", run_dir)

    # ---- 读取回测产物 ----
    pair_defs = _configured_pairs(run_dir)
    signal_fields = ["zscore"]
    signal_columns = ["ts"] + [
        f"{pair['pair_id']}_{field}"
        for pair in pair_defs
        for field in signal_fields
    ]
    equity = _sort_if_has_column(read_frame(run_dir / "equity_curve.csv", ["ts", "equity"]), "ts")
    position = _sort_if_has_column(
        read_frame(
            run_dir / "position_curve.csv",
            [
                "ts", "gross_exposure_ratio", "margin_deficit",
                "forced_deleveraging_triggered",
            ],
        ),
        "ts",
    )
    signal = _sort_if_has_column(
        read_frame(run_dir / "signal_curve.csv", signal_columns),
        "ts",
    )
    pair_curve = _sort_if_has_column(read_frame(run_dir / "pair_curve.csv"), "ts")
    trades = _sort_if_has_column(
        read_frame(
            run_dir / "trades.csv",
            [
                "group_id", "symbol", "action", "side", "quantity", "price", "notional",
                "fee", "slippage", "ts", "position_id", "target_hedge_ratio", "pair_id", "funding_fee",
                "exit_reason", "protection_trigger", "protection_stop_x_price",
                "exit_class", "reopen_lock_pending",
                "protection_take_profit_return",
                "protection_target_residual", "protection_stop_reason",
                "protection_max_holding_bars", "protection_max_holding_deadline_bar",
                "protection_rule", "protection_freeze_bars", "protection_freeze_until_bar",
            ],
        ),
        "ts",
    )
    funding = _sort_if_has_column(
        read_frame(
            run_dir / "funding_payments.csv",
            ["symbol", "ts", "payment", "funding_rate", "mark_price"],
        ),
        "ts",
    )
    try:
        metrics = loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    except Exception:
        metrics = {}
    if equity.is_empty() or "ts" not in equity.columns:
        raise ValueError(f"{run_dir} has no equity_curve.csv data")
    logger.info(
        "Trade Review：产物读取完成，pairs={} equity_rows={} signal_rows={} trades={}",
        len(pair_defs),
        equity.height,
        signal.height,
        trades.height,
    )

    estimator_review_parameters = _estimator_review_parameters(run_dir)
    initial_equity = float(metrics.get("initial_equity") or 0.0)

    # ---- 组合总览数据（不包含任何 Pair 分钟明细） ----
    logger.info("Trade Review：开始整理交易卡片与组合总览数据")
    event_trades = trade_rows(trades, signal)
    complete_trades = _complete_trade_rows(event_trades, pair_curve, funding)
    portfolio_points = _portfolio_points(equity, position, trades, max_points)
    overview_payload = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "range": {"start": int(equity["ts"].min()), "end": int(equity["ts"].max())},
        "metrics": _overview_metrics(metrics, equity),
        "points": _pack_series(
            portfolio_points,
            [
                "equity", "gross_exposure_ratio", "margin_deficit",
                "forced_deleveraging_triggered", "peak",
            ],
        ),
        "holding_distribution": holding_time_distribution(trades),
        "windows": {"drawdown": drawdown_window(equity), "gain": best_gain_window(equity)},
        "pairs": [],
    }
    overview_payload["metrics"].update(_complete_trade_metric_overrides(complete_trades))
    logger.info(
        "Trade Review：交易卡片基础数据完成，events={} completed_trades={}",
        len(event_trades),
        len(complete_trades),
    )

    # ---- Pair 汇总基础数据 ----
    pair_summary_base = {
        row["pair_id"]: row
        for row in build_pair_summary(
            trades=trades,
            position_curve=_pair_summary_position_frame(run_dir, pair_defs),
            funding_payments=funding,
            pair_defs=pair_defs,
            initial_equity=initial_equity,
            price_source=run_dir / "config.yaml",
            timeline=equity,
            final_position_valuation=_read_final_position_valuation(run_dir),
        )
    }

    data_dir = out_dir / "trade_review_data"
    pairs_data_dir = data_dir / "pairs"
    pairs_html_dir = out_dir / "trade_review_pairs"
    assets_dir = out_dir / "trade_review_assets"
    pairs_html_dir.mkdir(parents=True, exist_ok=True)
    pairs_data_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    trades_rows = trades.to_dicts()
    funding_rows = funding.to_dicts()
    thresholds = threshold_rows(run_dir)
    month_manifest: dict[str, dict] = {}
    backtest_min = int(equity["ts"].min())
    backtest_max = int(equity["ts"].max())

    total_pairs = len(pair_defs)
    for pair_index, pair_def in enumerate(pair_defs, start=1):
        pair_id = pair_def["pair_id"]
        x_symbol = pair_def["x_symbol"]
        y_symbol = pair_def["y_symbol"]
        logger.info(
            "Trade Review Pair [{}/{}] 开始: {} ({} / {})",
            pair_index,
            total_pairs,
            pair_id,
            x_symbol,
            y_symbol,
        )
        x_frame = _symbol_close_frame(run_dir, x_symbol, "x_close")
        y_frame = _symbol_close_frame(run_dir, y_symbol, "y_close")
        # Pair 时间轴裁剪到回测窗口内，且从 X/Y 首次共同有效价格开始，
        # 避免 X 有数据而 Y 尚未上市的 Pair（如 BTC/MSTR）默认窗口落在空区间
        x_ts_list = [int(value) for value in x_frame["ts"].to_list()]
        x_close_list = x_frame["x_close"].to_list()
        y_close_map = dict(zip(y_frame["ts"].to_list(), y_frame["y_close"].to_list()))
        first_common: int | None = None
        for ts, xv in zip(x_ts_list, x_close_list):
            if backtest_min <= ts <= backtest_max and xv is not None and y_close_map.get(ts) is not None:
                first_common = ts
                break
        if first_common is None:
            first_common = backtest_min
        timeline_ts = [ts for ts in x_ts_list if first_common <= ts <= backtest_max]
        x_prices = dict(zip(x_frame["ts"].to_list(), x_frame["x_close"].to_list()))
        y_prices = dict(zip(y_frame["ts"].to_list(), y_frame["y_close"].to_list()))
        z_rows = _zscore_forward_rows(signal, pair_id, pair_curve)
        points = _pair_contribution_points(
            pair_id, x_symbol, y_symbol,
            trades_rows, funding_rows,
            timeline_ts, x_prices, y_prices, z_rows,
            initial_equity=initial_equity,
            regression_method=estimator_review_parameters["regression_method"],
        )
        chunks = _chunk_points_by_month(points)
        max_dd, dd_start, dd_end = _pair_curve_drawdown(points, initial_equity)
        base = pair_summary_base.get(pair_id, {})

        # 持仓时间从完整交易对象重算（build_pair_summary 只输出天口径）
        pair_trades = [row for row in complete_trades if row.get("pair_id") == pair_id]
        pair_events = [row for row in event_trades if row.get("pair_id") == pair_id]
        holdings = [
            float(row["holding_minutes"])
            for row in pair_trades
            if row.get("holding_minutes") is not None
        ]
        avg_hold = sum(holdings) / len(holdings) if holdings else None
        sorted_hold = sorted(holdings)
        median_hold = (
            sorted_hold[len(sorted_hold) // 2]
            if len(sorted_hold) % 2
            else (sorted_hold[len(sorted_hold) // 2 - 1] + sorted_hold[len(sorted_hold) // 2]) / 2
            if len(sorted_hold)
            else None
        )

        # 交易卡片资金费改为按该 pair 的贡献曲线累计资金费取区间差值
        # （共享 symbol 时不再把整笔资金费重复算进多个 Pair）
        funding_curve = [point["cum_funding"] for point in points]
        funding_ts = [point["ts"] for point in points]

        def _cum_funding_at(ts: int) -> float:
            import bisect
            idx = bisect.bisect_right(funding_ts, ts) - 1
            return funding_curve[idx] if idx >= 0 else 0.0

        for trade in pair_trades:
            trade_events = [
                event for event in pair_events
                if event.get("trade_id") == trade.get("trade_id")
            ]
            initial_open_events = [
                event for event in trade_events
                if event.get("action") == "open"
                and int(event.get("ts") or -1) == int(trade.get("open_ts") or -2)
            ]
            if not initial_open_events:
                initial_open_events = [
                    event for event in trade_events
                    if event.get("action") in {"open", "add"}
                    and int(event.get("ts") or -1) == int(trade.get("open_ts") or -2)
                ]
            close_events = [
                event for event in trade_events if event.get("action") == "close"
            ]
            open_x, open_y, open_spread = _raw_pair_prices(
                trade.get("open_ts"), x_symbol, y_symbol, x_prices, y_prices,
                initial_open_events,
            )
            close_x, close_y, close_spread = _raw_pair_prices(
                trade.get("close_ts"), x_symbol, y_symbol, x_prices, y_prices,
                close_events,
            )
            trade.update({
                "open_x_price": open_x,
                "open_y_price": open_y,
                "open_raw_spread": open_spread,
                "close_x_price": close_x,
                "close_y_price": close_y,
                "close_raw_spread": close_spread,
            })
            if funding_ts:
                open_ts = int(trade["open_ts"])
                close_ts = int(trade["close_ts"]) if trade.get("close_ts") is not None else funding_ts[-1]
                trade["funding_fee"] = _cum_funding_at(close_ts) - _cum_funding_at(open_ts)
                # 资金费修正后同步重算净收益，保证卡片展示一致
                if trade.get("close_ts") is not None and trade.get("gross_pnl") is not None:
                    trade["net_pnl"] = (
                        trade["gross_pnl"]
                        - float(trade.get("fee") or 0.0)
                        - float(trade.get("slippage") or 0.0)
                        - float(trade.get("funding_fee") or 0.0)
                    )
                    open_notional = trade.get("open_notional")
                    trade["net_return"] = trade["net_pnl"] / open_notional if open_notional else None
            trade["scenario_analysis"] = _trade_scenario_analysis(
                trade=trade,
                open_events=initial_open_events,
                x_symbol=x_symbol,
                y_symbol=y_symbol,
                regression_method=estimator_review_parameters["regression_method"],
                exit_z=estimator_review_parameters["exit_z"],
            )

        final_pnl = points[-1]["pnl"] if points else 0.0
        final_funding = points[-1]["cum_funding"] if points else 0.0
        summary = {
            "pair_id": pair_id,
            "x_symbol": x_symbol,
            "y_symbol": y_symbol,
            "total_positions": int(base.get("total_positions") or 0),
            "closed_positions": int(base.get("closed_positions") or 0),
            "open_positions": int(base.get("open_positions") or 0),
            "win_rate": base.get("win_rate"),
            "profit_loss_ratio": base.get("profit_loss_ratio"),
            "total_pnl": final_pnl,
            "contribution_to_initial_equity": final_pnl / initial_equity if initial_equity else None,
            "max_drawdown": max_dd,
            "max_drawdown_window": {"start": dd_start, "end": dd_end},
            "avg_holding_minutes": avg_hold,
            "median_holding_minutes": median_hold,
            "max_holding_minutes": max(holdings) if holdings else None,
            "total_fee": base.get("total_fee"),
            "total_slippage": base.get("total_slippage"),
            "funding_fee": final_funding,
            "scenario_space": _pair_scenario_summary(pair_trades),
        }
        overview_payload["pairs"].append(summary)

        # ---- 月度分块数据文件 ----
        pair_data_dir = pairs_data_dir / pair_id
        pair_data_dir.mkdir(parents=True, exist_ok=True)
        # 清理旧月份文件，避免重新生成后残留回测范围外的数据
        for stale in pair_data_dir.glob("*.json"):
            if stale.name != "summary.json":
                stale.unlink()
        month_counts: dict[str, int] = {}
        for month, chunk in chunks.items():
            packed = _pack_series(
                chunk,
                [
                    "x_close", "y_close", "y_theoretical", "zscore",
                    "pnl", "cum_funding", "peak",
                ],
            )
            (pair_data_dir / f"{month}.json").write_text(
                dumps(_json_safe(packed), ensure_ascii=False, separators=(",", ":"), allow_nan=False),
                encoding="utf-8",
            )
            month_counts[month] = len(chunk)
        (pair_data_dir / "summary.json").write_text(
            dumps(_json_safe(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        month_manifest[pair_id] = {"months": month_counts, "total_points": len(points)}

        # ---- Pair 独立页面 ----
        # 默认窗口：优先定位到首笔交易附近；否则首次有效 Z-score 之后；
        # 否则用最大回撤窗口；都没有则取时间轴起点后 14 天。
        day_ms = 24 * 60 * 60 * 1000
        timeline_start = timeline_ts[0] if timeline_ts else None
        timeline_end = timeline_ts[-1] if timeline_ts else None
        default_window: dict | None = None
        if pair_trades:
            first_open = min(int(trade["open_ts"]) for trade in pair_trades)
            default_window = {"start": first_open - 3 * day_ms, "end": first_open + 3 * day_ms}
        else:
            valid_z = [row["ts"] for row in z_rows if row.get("zscore") is not None]
            if valid_z:
                first_z = min(valid_z)
                default_window = {"start": first_z - 2 * day_ms, "end": first_z + 2 * day_ms}
            elif dd_start is not None:
                default_window = {"start": dd_start, "end": dd_end}
        if default_window is None and timeline_start is not None and timeline_end is not None:
            default_window = {
                "start": timeline_start,
                "end": min(timeline_end, timeline_start + 14 * day_ms),
            }
        if default_window is not None and timeline_start is not None and timeline_end is not None:
            default_window["start"] = max(int(default_window["start"]), timeline_start)
            default_window["end"] = min(int(default_window["end"]), timeline_end)

        pair_config = {
            "pair_id": pair_id,
            "summary": summary,
            "initial_equity": initial_equity,
            "default_window": default_window,
            "range": {
                "start": timeline_start,
                "end": timeline_end,
            },
            "trades": pair_trades,
            "events": pair_events,
            "thresholds": thresholds,
            "model_lookback_bars": estimator_review_parameters["model_lookback_bars"],
            "model_update_interval_bars": estimator_review_parameters["model_update_interval_bars"],
            "regression_method": estimator_review_parameters["regression_method"],
            "exit_z": estimator_review_parameters["exit_z"],
        }
        pair_html = (
            PAIR_TEMPLATE
            .replace("__PAIR_ID__", str(pair_id))
            .replace(
                "__PAIR_CONFIG__",
                dumps(_json_safe(pair_config), ensure_ascii=False, separators=(",", ":"), allow_nan=False),
            )
            .replace("__TITLE__", escape(run_dir.name))
        )
        (pairs_html_dir / f"{pair_id}.html").write_text(pair_html, encoding="utf-8")

    # ---- 共享资产与组合数据文件 ----
    (assets_dir / "review.css").write_text(REVIEW_CSS, encoding="utf-8")
    (assets_dir / "review.js").write_text(REVIEW_JS, encoding="utf-8")

    # 组合数据已经嵌入总览 HTML；清理旧版遗留的重复外部副本。
    portfolio_data_path = data_dir / "portfolio.json"
    if portfolio_data_path.exists():
        portfolio_data_path.unlink()
    (data_dir / "manifest.json").write_text(
        dumps(
            _json_safe({
                "version": 2,
                "run_name": run_dir.name,
                "pairs": month_manifest,
            }),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---- 组合总览页 ----
    overview_html = (
        OVERVIEW_TEMPLATE
        .replace("__TITLE__", escape(run_dir.name))
        .replace(
            "__PAYLOAD__",
            dumps(_json_safe(overview_payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        )
    )
    output.write_text(overview_html, encoding="utf-8")
    logger.info(
        "trade_review v2 generated overview={} pairs={} data_dir={} elapsed_minutes={:.1f}",
        output,
        len(pair_defs),
        data_dir,
        (perf_counter() - review_started) / 60.0,
    )
    return output


def _overview_metrics(metrics: dict, equity: pl.DataFrame) -> dict:
    """在回测 metrics 基础上补充 CAGR 等总览指标。"""
    result = dict(metrics)
    result.setdefault("cagr", None)
    initial = result.get("initial_equity")
    final = result.get("final_equity")
    if initial and final and not equity.is_empty():
        try:
            start = int(equity["ts"].min())
            end = int(equity["ts"].max())
            days = (end - start) / 86400000.0
            if days > 0 and float(initial) > 0:
                result["cagr"] = (float(final) / float(initial)) ** (365.0 / days) - 1
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return result


def _complete_trade_metric_overrides(trades: list[dict]) -> dict:
    closed = [row for row in trades or [] if row.get("net_pnl") is not None]
    wins = [float(row["net_pnl"]) for row in closed if float(row["net_pnl"]) > 0]
    losses = [-float(row["net_pnl"]) for row in closed if float(row["net_pnl"]) < 0]
    holdings = [
        float(row["holding_minutes"])
        for row in closed if row.get("holding_minutes") is not None
    ]
    return {
        "trade_count": len(trades or []),
        "win_rate": len(wins) / len(closed) if closed else None,
        "profit_loss_ratio": (
            (sum(wins) / len(wins)) / (sum(losses) / len(losses))
            if wins and losses else None
        ),
        "average_holding_minutes": sum(holdings) / len(holdings) if holdings else None,
    }


def _pack_series(points: list[dict], value_keys: list[str]) -> dict:
    """列式紧凑打包时间序列：ts 用相邻差分（第一项为 0），数值保留 6 位小数。

    对象数组（每点重复一遍键名）打包成 {t0, t:[相邻差分], key:[值]}，
    体积可降 10 倍以上；前端按 t0 + 累加差分还原 ts。
    """
    t0: int | None = None
    prev_ts: int | None = None
    ts_deltas: list[int] = []
    columns: dict[str, list] = {key: [] for key in value_keys}
    for point in points or []:
        ts = point.get("ts")
        if ts is None:
            continue
        ts = int(ts)
        if t0 is None:
            t0 = ts
            ts_deltas.append(0)
        else:
            ts_deltas.append(ts - prev_ts if prev_ts is not None else 0)
        prev_ts = ts
        for key in value_keys:
            value = point.get(key)
            columns[key].append(round(float(value), 6) if value is not None else None)
    result: dict = {"t0": t0, "t": ts_deltas}
    result.update(columns)
    return result


def _month_key(ts: int) -> str:
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m")


def _zscore_forward_rows(
    signal: pl.DataFrame,
    pair_id: str,
    pair_curve: pl.DataFrame | None = None,
) -> list[dict]:
    """取该 pair 的信号参数（按 ts 升序，用于分钟级前向填充）。"""
    prefix = f"{pair_id}_"
    column = f"{prefix}zscore"
    if signal.is_empty() or column not in signal.columns:
        return []
    selected_columns = ["ts", column]
    signal_rows = signal.select(selected_columns).sort("ts")
    state_rows = pl.DataFrame()
    if (
        pair_curve is not None
        and not pair_curve.is_empty()
        and {"ts", "pair_id"}.issubset(pair_curve.columns)
    ):
        state_columns = [
            name for name in ("ts", "alpha", "beta")
            if name in pair_curve.columns
        ]
        if len(state_columns) > 1:
            state_rows = (
                pair_curve
                .filter(pl.col("pair_id") == pair_id)
                .select(state_columns)
                .sort("ts")
                .unique(subset=["ts"], keep="last")
            )
    if not state_rows.is_empty():
        signal_rows = signal_rows.join_asof(state_rows, on="ts", strategy="backward")
    rows = []
    # signal_curve can contain hundreds of fields for all configured pairs.
    # Restrict conversion to the current pair to avoid copying the full frame
    # once per pair while generating the report.
    for row in signal_rows.to_dicts():
        value = _float(row.get(column))
        rows.append({
            "ts": int(row["ts"]),
            "zscore": value,
            "alpha": _float(row.get("alpha")),
            "spread_beta": _float(row.get("beta")),
        })
    return rows


def _pair_contribution_points(
    pair_id: str,
    x_symbol: str,
    y_symbol: str,
    trades_rows: list[dict],
    funding_rows: list[dict],
    timeline_ts: list[int],
    x_prices: dict[int, float],
    y_prices: dict[int, float],
    z_rows: list[dict],
    initial_equity: float = 0.0,
    regression_method: str | None = None,
) -> list[dict]:
    """按 pair_id 的成交记录独立重建两腿持仓，输出每分钟贡献点。

    PairPnL_t = 累计成交现金流_t + qx*Px + qy*Py - 累计资金费率_t
    - 成交现金流使用实际成交价（已含滑点影响），手续费在成交时扣除；
    - 资金费率按该 Pair 在结算时刻对某 symbol 的持仓（带符号数量）乘以
      funding_rate × mark_price 重算：pair_funding = q * mark * rate，
      与回测引擎 payment = qty * mark * rate 的口径一致；共享同一 symbol 的
      多个 Pair 各自按自己的持仓计算，天然不会重复归属，且方向（多头付/空头收）正确；
    - 结算顺序与回测引擎一致：同一分钟内先按上一分钟末的持仓结算资金费率，
      再应用本分钟的成交（exchange.__call__ 中先 _apply_funding 后撮合）。
    """
    regression_method = str(regression_method or "").lower()
    pair_trades = sorted(
        (t for t in trades_rows if t.get("pair_id") == pair_id),
        key=lambda t: (t["ts"], t.get("group_id") or ""),
    )
    all_trades = sorted(trades_rows, key=lambda t: (t["ts"], t.get("group_id") or ""))
    funding = sorted(funding_rows, key=lambda f: (f["ts"], f["symbol"]))

    pi = ai = fi = zi = 0
    qx = qy = 0.0
    cashflow = 0.0
    total_q: dict[str, float] = {}
    cum_funding = 0.0
    last_z: float | None = None
    last_alpha: float | None = None
    last_spread_beta: float | None = None
    last_px: float | None = None
    last_py: float | None = None
    peak: float | None = None

    points = []
    for ts in timeline_ts:
        # 1) 先结算资金费率：使用进入本分钟时的持仓（上一分钟末）
        while fi < len(funding) and funding[fi]["ts"] <= ts:
            f = funding[fi]
            sym = f["symbol"]
            rate = _float(f.get("funding_rate"))
            mark = _float(f.get("mark_price"))
            if rate is None or mark is None:
                # 旧数据回退：按带符号持仓比例分摊 payment。
                # payment = total_q × mark × rate（净额），pair 分摊 = payment × pq / total_q
                # = pq × mark × rate，与按持仓重算等价；方向相反的共享持仓不会超分。
                tq = total_q.get(sym, 0.0)
                pq = qx if sym == x_symbol else (qy if sym == y_symbol else 0.0)
                if abs(tq) > 1e-12:
                    cum_funding += float(f["payment"]) * (pq / tq)
            else:
                q = qx if sym == x_symbol else (qy if sym == y_symbol else 0.0)
                cum_funding += q * mark * rate
            fi += 1
        # 2) 再应用本分钟的成交（更新持仓与现金流）
        while ai < len(all_trades) and all_trades[ai]["ts"] <= ts:
            t = all_trades[ai]
            sym = t["symbol"]
            qty = float(t["quantity"])
            total_q[sym] = total_q.get(sym, 0.0) + (qty if t["side"] == "buy" else -qty)
            ai += 1
        while pi < len(pair_trades) and pair_trades[pi]["ts"] <= ts:
            t = pair_trades[pi]
            sym = t["symbol"]
            qty = float(t["quantity"])
            delta = qty if t["side"] == "buy" else -qty
            if sym == x_symbol:
                qx += delta
            elif sym == y_symbol:
                qy += delta
            notional = float(t["notional"])
            fee = float(t["fee"])
            cashflow += (notional - fee) if t["side"] == "sell" else (-notional - fee)
            pi += 1
        while zi < len(z_rows) and z_rows[zi]["ts"] <= ts:
            if z_rows[zi]["zscore"] is not None:
                last_z = z_rows[zi]["zscore"]
            if z_rows[zi].get("alpha") is not None:
                last_alpha = float(z_rows[zi]["alpha"])
            if z_rows[zi].get("spread_beta") is not None:
                last_spread_beta = float(z_rows[zi]["spread_beta"])
            zi += 1
        px = x_prices.get(ts)
        py = y_prices.get(ts)
        if px is not None:
            last_px = float(px)
        if py is not None:
            last_py = float(py)
        # A missing quote must not value an existing leg at zero.  Carry the
        # most recent valid mark for PnL while preserving null chart values so
        # the data gap remains visible to the reviewer.
        marked_x = last_px if last_px is not None else 0.0
        marked_y = last_py if last_py is not None else 0.0
        pnl = cashflow + qx * marked_x + qy * marked_y - cum_funding
        theoretical_y: float | None = None
        if (
            regression_method in {"log_price", "price"}
            and px is not None
            and float(px) > 0
            and last_alpha is not None
            and last_spread_beta is not None
        ):
            try:
                if regression_method == "log_price":
                    theoretical_y = exp(last_alpha + last_spread_beta * log(float(px)))
                else:
                    theoretical_y = last_alpha + last_spread_beta * float(px)
                if theoretical_y is not None and (
                    not isfinite(theoretical_y) or theoretical_y <= 0.0
                ):
                    theoretical_y = None
            except (OverflowError, TypeError, ValueError):
                theoretical_y = None
        # 全局累计峰值（贡献权益口径），供前端回撤图使用：dd = equity / peak - 1
        equity_contrib = initial_equity + pnl
        if peak is None or equity_contrib > peak:
            peak = equity_contrib
        points.append(
            {
                "ts": ts,
                "x_close": px,
                "y_close": py,
                "y_theoretical": theoretical_y,
                "zscore": last_z,
                "pnl": pnl,
                "cum_funding": cum_funding,
                "peak": peak,
            }
        )
    return points


def _chunk_points_by_month(points: list[dict]) -> dict[str, list[dict]]:
    chunks: dict[str, list[dict]] = {}
    for point in points:
        chunks.setdefault(_month_key(point["ts"]), []).append(point)
    return chunks


def _pair_curve_drawdown(points: list[dict], initial_equity: float) -> tuple[float, int | None, int | None]:
    """Pair 贡献曲线的最大回撤与窗口（trough 前后各外扩 3 天）。"""
    peak = None
    max_dd = 0.0
    trough_ts = None
    for point in points:
        eq = initial_equity + point["pnl"]
        if peak is None or eq > peak:
            peak = eq
        if peak and peak > 0:
            dd = eq / peak - 1
            if dd < max_dd:
                max_dd = dd
                trough_ts = point["ts"]
    if trough_ts is None or peak is None:
        return 0.0, None, None
    pad = 3 * 24 * 60 * 60 * 1000
    return max_dd, trough_ts - pad, trough_ts + pad


def discover_report_runs(results_root: Path) -> list[dict]:
    """Return backtest folders and lightweight metadata for the report index."""
    results_root = Path(results_root)
    if not results_root.is_absolute():
        results_root = ROOT / results_root
    results_root = results_root.resolve()
    if not results_root.is_dir():
        return []

    runs: list[dict] = []
    for run_dir in results_root.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        config_path = run_dir / "config.yaml"
        metrics_path = run_dir / "metrics.json"
        report_path = run_dir / "trade_review.html"
        if not (config_path.exists() or metrics_path.exists() or report_path.exists()):
            continue

        metrics = {}
        if metrics_path.exists():
            try:
                metrics = loads(metrics_path.read_text(encoding="utf-8")) or {}
            except (OSError, ValueError, TypeError):
                metrics = {}

        config = {}
        if config_path.exists():
            try:
                config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            except (OSError, ValueError, TypeError, yaml.YAMLError):
                config = {}
        setup = (config.get("setups") or {}).get(config.get("active_setup"), {}) or {}
        pipeline = setup.get("pipeline") or {}
        estimator = pipeline.get("estimator") or {}
        signal = pipeline.get("signal") or {}
        modified_at = run_dir.stat().st_mtime
        runs.append(
            {
                "run_id": run_dir.name,
                "report_ready": report_path.is_file(),
                "modified_at": modified_at,
                "modified_label": datetime.fromtimestamp(modified_at).strftime("%Y-%m-%d %H:%M"),
                "method": estimator.get("method"),
                "regression_method": estimator.get("regression_method"),
                "model_lookback_bars": _float(estimator.get("model_lookback_bars")),
                "model_update_interval_bars": _float(estimator.get("model_update_interval_bars")),
                "position_update_policy": estimator.get("position_update_policy"),
                "entry_z": _float(signal.get("entry_z")),
                "exit_z": _float(signal.get("exit_z")),
                "initial_equity": _float(metrics.get("initial_equity")),
                "final_equity": _float(metrics.get("final_equity")),
                "total_return": _float(metrics.get("total_return")),
                "sharpe": _float(metrics.get("sharpe")),
                "max_drawdown": _float(metrics.get("max_drawdown")),
                "trade_count": _float(metrics.get("trade_count")),
            }
        )
    return sorted(runs, key=lambda item: (item["modified_at"], item["run_id"]), reverse=True)


def _report_center_html(runs: list[dict], results_root: Path) -> str:
    payload = dumps(_json_safe(runs), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    payload = payload.replace("</", "<\\/")
    root_label = escape(str(Path(results_root).resolve()))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>BTC-COIN 回测报告中心</title>
  <style>
    :root {{ color-scheme:light; --bg:#f5f7fb; --panel:#fff; --ink:#17202a; --muted:#667085;
      --line:#d7dde8; --blue:#2563eb; --green:#15803d; --red:#c81e1e; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font-family:"Microsoft YaHei","Segoe UI",Arial,sans-serif; }}
    header {{ background:#fff; border-bottom:1px solid var(--line); padding:18px 28px 14px; position:sticky; top:0; z-index:2; }}
    h1 {{ margin:0 0 5px; font-size:22px; letter-spacing:0; }}
    .root {{ color:var(--muted); font-size:12px; overflow-wrap:anywhere; }}
    main {{ max-width:1500px; margin:0 auto; padding:18px 28px 30px; }}
    .toolbar {{ display:grid; grid-template-columns:minmax(240px,1fr) minmax(280px,2fr) auto auto; gap:10px; align-items:center;
      background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:12px; margin-bottom:14px; }}
    input, select, button {{ height:36px; border:1px solid var(--line); border-radius:6px; background:#fff; color:var(--ink);
      padding:0 10px; font:inherit; font-size:13px; min-width:0; }}
    button {{ cursor:pointer; white-space:nowrap; }}
    button.primary {{ background:var(--blue); border-color:var(--blue); color:#fff; }}
    button:disabled {{ cursor:not-allowed; opacity:.45; }}
    .summary {{ color:var(--muted); font-size:13px; margin:0 0 10px; }}
    .table-wrap {{ overflow:auto; background:var(--panel); border:1px solid var(--line); border-radius:6px; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th,td {{ padding:9px 10px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }}
    th:first-child,td:first-child {{ text-align:left; }}
    th {{ position:sticky; top:0; background:#fafbfc; color:var(--muted); font-weight:600; }}
    tbody tr.ready {{ cursor:pointer; }}
    tbody tr.ready:hover, tbody tr.selected {{ background:#eff6ff; }}
    tbody tr.unavailable {{ color:#98a2b3; }}
    .status {{ display:inline-block; min-width:54px; text-align:center; border-radius:4px; padding:2px 6px; font-size:12px; }}
    .status.ready {{ color:var(--green); background:#ecfdf3; }}
    .status.missing {{ color:#8a6100; background:#fff7db; }}
    .pos {{ color:var(--green); }} .neg {{ color:var(--red); }}
    .empty {{ padding:36px; color:var(--muted); text-align:center; }}
    @media(max-width:900px) {{ header,main {{ padding-left:14px; padding-right:14px; }}
      .toolbar {{ grid-template-columns:1fr; }} input,select,button {{ width:100%; }} }}
  </style>
</head>
<body>
  <header><h1>BTC-COIN 回测报告中心</h1><div class="root">结果目录：{root_label}</div></header>
  <main>
    <div class="toolbar">
      <input id="search" type="search" placeholder="搜索文件夹、模型或回归类型" aria-label="搜索回测" />
      <select id="runSelect" aria-label="选择回测结果"></select>
      <button class="primary" id="openRun">打开回测报告</button>
      <button id="refreshRuns">刷新列表</button>
    </div>
    <div class="summary" id="summary"></div>
    <div class="table-wrap"><table>
      <thead><tr><th>回测文件夹</th><th>状态</th><th>模型</th><th>数据</th><th>窗口</th><th>更新</th>
        <th>开仓 / 平仓 Z</th><th>最终权益</th><th>总收益</th><th>Sharpe</th><th>最大回撤</th><th>交易数</th><th>更新时间</th></tr></thead>
      <tbody id="runRows"></tbody>
    </table></div>
  </main>
  <script>
    const allRuns = {payload};
    const search = document.getElementById("search");
    const select = document.getElementById("runSelect");
    const rows = document.getElementById("runRows");
    const openButton = document.getElementById("openRun");
    let visibleRuns = [];
    const esc = value => String(value ?? "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[ch]));
    const num = (value, digits=2) => value == null ? "-" : Number(value).toLocaleString("en-US", {{maximumFractionDigits:digits}});
    const pct = value => value == null ? "-" : (Number(value)*100).toFixed(2)+"%";
    const signed = value => value == null ? "" : (value > 0 ? "pos" : (value < 0 ? "neg" : ""));
    const bars = value => {{
      if (value == null) return "-";
      if (value % 1440 === 0) return num(value/1440,0)+"D";
      if (value % 60 === 0) return num(value/60,0)+"H";
      return num(value,0)+"m";
    }};
    const runUrl = run => "/runs/"+encodeURIComponent(run.run_id)+"/trade_review.html";
    function openSelected() {{
      const run = allRuns.find(item => item.run_id === select.value);
      if (run?.report_ready) window.open(runUrl(run), "_blank", "noopener,noreferrer");
    }}
    function render() {{
      const query = search.value.trim().toLowerCase();
      visibleRuns = allRuns.filter(run => !query || [run.run_id,run.method,run.regression_method].some(v => String(v ?? "").toLowerCase().includes(query)));
      const previous = select.value;
      const readyRuns = visibleRuns.filter(run => run.report_ready);
      select.innerHTML = readyRuns.length ? readyRuns.map(run => `<option value="${{esc(run.run_id)}}">${{esc(run.run_id)}} · ${{esc(run.method || "未知模型")}}</option>`).join("") : '<option value="">没有可打开的报告</option>';
      if (readyRuns.some(run => run.run_id === previous)) select.value = previous;
      openButton.disabled = !readyRuns.length;
      document.getElementById("summary").textContent = `共 ${{allRuns.length}} 个回测文件夹，${{allRuns.filter(r => r.report_ready).length}} 个已有复盘报告；当前显示 ${{visibleRuns.length}} 个。`;
      rows.innerHTML = visibleRuns.length ? visibleRuns.map(run => `
        <tr class="${{run.report_ready ? "ready" : "unavailable"}}" data-run="${{esc(run.run_id)}}">
          <td>${{esc(run.run_id)}}</td><td><span class="status ${{run.report_ready ? "ready" : "missing"}}">${{run.report_ready ? "可查看" : "未生成"}}</span></td>
          <td>${{esc(run.method || "-")}}</td><td>${{esc(run.regression_method || "-")}}</td>
          <td>${{bars(run.model_lookback_bars)}}</td><td>${{bars(run.model_update_interval_bars)}}</td>
          <td>${{num(run.entry_z,2)}} / ${{num(run.exit_z,2)}}</td><td>${{num(run.final_equity,0)}}</td>
          <td class="${{signed(run.total_return)}}">${{pct(run.total_return)}}</td><td>${{num(run.sharpe,3)}}</td>
          <td class="${{signed(run.max_drawdown)}}">${{pct(run.max_drawdown)}}</td><td>${{num(run.trade_count,0)}}</td><td>${{esc(run.modified_label)}}</td>
        </tr>`).join("") : '<tr><td class="empty" colspan="13">没有匹配的回测结果</td></tr>';
      rows.querySelectorAll("tr.ready").forEach(row => row.addEventListener("click", () => {{
        select.value = row.dataset.run; rows.querySelectorAll("tr.selected").forEach(item => item.classList.remove("selected")); row.classList.add("selected");
      }}));
      rows.querySelectorAll("tr.ready").forEach(row => row.addEventListener("dblclick", openSelected));
    }}
    search.addEventListener("input", render);
    select.addEventListener("change", () => {{
      rows.querySelectorAll("tr").forEach(row => row.classList.toggle("selected", row.dataset.run === select.value));
    }});
    openButton.addEventListener("click", openSelected);
    document.getElementById("refreshRuns").addEventListener("click", () => window.location.reload());
    render();
  </script>
</body>
</html>"""


def _resolved_run_dir(results_root: Path, run_id: str) -> Path | None:
    results_root = Path(results_root).resolve()
    candidate = (results_root / run_id).resolve()
    try:
        relative = candidate.relative_to(results_root)
    except ValueError:
        return None
    if len(relative.parts) != 1 or not candidate.is_dir():
        return None
    return candidate


def _inject_report_center_link(html: bytes) -> bytes:
    """Add a report-center link to reports generated before the center existed."""
    if b'id="runIndexLink"' in html or b"id='runIndexLink'" in html:
        return html
    script = br"""
<script>
(function () {
  const toolbar = document.querySelector(".buttons");
  if (!toolbar || document.getElementById("runIndexLink")) return;
  const link = document.createElement("a");
  link.id = "runIndexLink";
  link.href = "/";
  link.textContent = "\u5207\u6362\u56de\u6d4b";
  link.className = "back-link";
  Object.assign(link.style, {
    marginBottom: "0", border: "1px solid #d7dde8", borderRadius: "6px",
    padding: "6px 12px", background: "#fff", textDecoration: "none"
  });
  toolbar.insertBefore(link, toolbar.firstChild);
})();
</script>
"""
    marker = b"</body>"
    return html.replace(marker, script + marker, 1) if marker in html else html + script


def serve_report_center(
    results_root: Path,
    port: int = 8000,
    open_browser: bool = True,
    initial_run_dir: Path | None = None,
) -> None:
    """Serve a selectable collection of backtest reports on localhost."""
    import http.server
    import threading
    import webbrowser
    from urllib.parse import quote, unquote, urlsplit

    results_root = Path(results_root)
    if not results_root.is_absolute():
        results_root = ROOT / results_root
    results_root = results_root.resolve()
    if not results_root.is_dir():
        raise FileNotFoundError(f"backtest results directory not found: {results_root}")

    class ReportCenterHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(results_root), **kwargs)

        def _send_content(self, content: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path in {"/", "/index.html"}:
                body = _report_center_html(discover_report_runs(results_root), results_root).encode("utf-8")
                self._send_content(body, "text/html; charset=utf-8")
                return
            if parsed.path == "/api/runs":
                body = dumps(_json_safe(discover_report_runs(results_root)), ensure_ascii=False, allow_nan=False).encode("utf-8")
                self._send_content(body, "application/json; charset=utf-8")
                return
            if not parsed.path.startswith("/runs/"):
                self.send_error(404, "Unknown report route")
                return

            raw_tail = parsed.path[len("/runs/"):]
            raw_run_id, separator, raw_asset = raw_tail.partition("/")
            run_id = unquote(raw_run_id)
            run_dir = _resolved_run_dir(results_root, run_id)
            if run_dir is None or not (run_dir / "trade_review.html").is_file():
                self.send_error(404, "Backtest report not found")
                return
            if not separator or not raw_asset:
                self.send_response(302)
                self.send_header("Location", f"/runs/{quote(run_id, safe='')}/trade_review.html")
                self.end_headers()
                return

            asset_path = (run_dir / unquote(raw_asset)).resolve()
            try:
                asset_path.relative_to(run_dir)
            except ValueError:
                self.send_error(404, "Invalid report path")
                return
            if asset_path.is_file() and asset_path.suffix.lower() == ".html":
                try:
                    body = _inject_report_center_link(asset_path.read_bytes())
                except OSError:
                    self.send_error(404, "Report page not found")
                    return
                self._send_content(body, "text/html; charset=utf-8")
                return
            self.path = f"/{quote(run_id, safe='')}/{raw_asset}"
            if parsed.query:
                self.path += f"?{parsed.query}"
            super().do_GET()

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), ReportCenterHandler)
    url = f"http://127.0.0.1:{port}/"
    if initial_run_dir is not None:
        initial_run_dir = Path(initial_run_dir)
        if not initial_run_dir.is_absolute():
            initial_run_dir = ROOT / initial_run_dir
        initial_run_dir = initial_run_dir.resolve()
        if initial_run_dir.parent == results_root and (initial_run_dir / "trade_review.html").is_file():
            url = f"http://127.0.0.1:{port}/runs/{quote(initial_run_dir.name, safe='')}/trade_review.html"
    print(f"Trade review center: {url}")
    print(f"Results directory: {results_root}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


def serve_report(run_dir: Path, port: int = 8000, open_browser: bool = True) -> None:
    """Serve one report initially while keeping sibling reports selectable."""
    run_dir = Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir = run_dir.resolve()
    if not (run_dir / "trade_review.html").exists():
        raise FileNotFoundError(
            f"{run_dir} has no trade_review.html; run the trade_review export first"
        )
    serve_report_center(run_dir.parent, port, open_browser, initial_run_dir=run_dir)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--max-points",
        type=int,
        default=2000,
        help="兼容旧命令；交易复盘保留完整时间序列，不做降采样",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=20,
        help="兼容旧命令；不再遗漏配置中的 Pair",
    )
    parser.add_argument("--serve", action="store_true", help="导出后启动本地 HTTP 服务并自动打开浏览器")
    parser.add_argument("--port", type=int, default=8000, help="本地服务端口（仅监听 127.0.0.1）")
    args = parser.parse_args()

    run_dir = args.run_dir or latest_backtest_dir()
    output = export_trade_review_html(run_dir, args.output, args.max_points, args.max_pairs)
    print(output)
    if args.serve:
        serve_report(run_dir, args.port, open_browser=True)


if __name__ == "__main__":
    main()
