from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from core.modules.data.loader import load_csv_data
from core.modules.reporting.utils import reporting_frame


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def pair_defs_from_config(config_path: Path) -> list[dict]:
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
        pairs.append(
            {
                "pair_id": item.get("pair_id") or f"{x_symbol}_{y_symbol}",
                "x_symbol": x_symbol,
                "y_symbol": y_symbol,
            }
        )

    if pairs:
        return pairs

    pair = setup.get("pair", raw.get("pair", {}))
    x_symbol = pair.get("x_symbol", pair.get("long_symbol"))
    y_symbol = pair.get("y_symbol", pair.get("short_symbol"))
    if x_symbol and y_symbol:
        pairs.append(
            {
                "pair_id": pair.get("pair_id") or f"{x_symbol}_{y_symbol}",
                "x_symbol": x_symbol,
                "y_symbol": y_symbol,
            }
        )
    return pairs


def required_position_columns(pair_defs: list[dict]) -> list[str]:
    columns = ["ts"]
    seen = {"ts"}
    for pair in pair_defs or []:
        for symbol in (pair.get("x_symbol"), pair.get("y_symbol")):
            if not symbol:
                continue
            column = f"{symbol}_position_value"
            if column not in seen:
                seen.add(column)
                columns.append(column)
    return columns


def build_pair_summary(
    trades: Any,
    position_curve: Any = None,
    funding_payments: Any = None,
    pair_defs: list[dict] | None = None,
    initial_equity: float | None = None,
    price_source: Path | str | None = None,
    timeline: Any = None,
    final_position_valuation: dict | None = None,
) -> list[dict]:
    pair_defs = pair_defs or []
    pair_symbols = {
        str(pair["pair_id"]): (pair.get("x_symbol"), pair.get("y_symbol"))
        for pair in pair_defs
        if pair.get("pair_id")
    }

    trade_df = _normalize_trades(_to_frame(trades))
    funding_df = _normalize_funding(_to_frame(funding_payments))
    position_df = _to_frame(position_curve)
    pair_ids = set(pair_symbols)
    if not trade_df.is_empty() and "pair_id" in trade_df.columns:
        pair_ids.update(str(pid) for pid in trade_df["pair_id"].drop_nulls().unique().to_list())

    if not pair_ids:
        return []

    if trade_df.is_empty():
        return [
            _empty_pair_row(pair_id, pair_symbols.get(pair_id), initial_equity)
            for pair_id in sorted(pair_ids)
        ]

    trade_df = trade_df.with_columns(
        pl.when(pl.col("side") == "sell")
        .then(pl.col("notional") - pl.col("fee"))
        .otherwise(-pl.col("notional") - pl.col("fee"))
        .alias("cashflow"),
        pl.when(pl.col("action") == "open").then(pl.col("notional")).otherwise(0.0).alias("open_notional"),
        pl.when(pl.col("action") == "close").then(pl.col("notional")).otherwise(0.0).alias("close_notional"),
    )
    if "position_id" in trade_df.columns and "group_id" in trade_df.columns:
        trade_df = trade_df.with_columns(
            pl.coalesce([pl.col("position_id"), pl.col("group_id")]).alias("position_key")
        )
    elif "position_id" in trade_df.columns:
        trade_df = trade_df.with_columns(pl.col("position_id").alias("position_key"))
    elif "group_id" in trade_df.columns:
        trade_df = trade_df.with_columns(pl.col("group_id").alias("position_key"))
    else:
        trade_df = trade_df.with_columns(pl.col("pair_id").alias("position_key"))

    funding_by_position, funding_by_pair, funding_events = _funding_allocations(
        trade_df, funding_df
    )

    positions = (
        trade_df.drop_nulls(["pair_id", "position_key"])
        .group_by(["pair_id", "position_key"])
        .agg(
            pl.col("ts").filter(pl.col("action") == "open").min().alias("open_ts"),
            pl.col("ts").filter(pl.col("action") == "close").max().alias("close_ts"),
            pl.col("cashflow").sum().alias("cashflow_pnl"),
            pl.col("open_notional").sum().alias("open_gross_notional"),
            pl.col("close_notional").sum().alias("close_gross_notional"),
            pl.col("fee").sum().alias("fee"),
            pl.col("slippage").sum().alias("slippage"),
        )
        .with_columns(
            (pl.col("open_ts").is_not_null() & pl.col("close_ts").is_not_null()).alias("is_closed"),
            ((pl.col("close_ts") - pl.col("open_ts")) / 60000.0).alias("holding_minutes"),
        )
    )
    funding_position_rows = [
        {"pair_id": key[0], "position_key": key[1], "funding_fee": value}
        for key, value in funding_by_position.items()
    ]
    if funding_position_rows:
        positions = positions.join(
            pl.DataFrame(funding_position_rows, infer_schema_length=None),
            on=["pair_id", "position_key"],
            how="left",
        )
    else:
        positions = positions.with_columns(pl.lit(0.0).alias("funding_fee"))
    positions = positions.with_columns(
        pl.col("funding_fee").cast(pl.Float64, strict=False).fill_null(0.0),
    ).with_columns(
        (pl.col("cashflow_pnl") - pl.col("funding_fee")).alias("net_pnl")
    )

    cash_by_pair = _rows_by_pair(
        trade_df.group_by("pair_id").agg(
            pl.col("cashflow").sum().alias("cashflow_pnl"),
            pl.col("notional").sum().alias("total_trade_notional"),
            pl.col("open_notional").sum().alias("total_open_gross_notional"),
            pl.col("fee").sum().alias("total_fee"),
            pl.col("slippage").sum().alias("total_slippage"),
            pl.col("ts").min().alias("first_trade_ts"),
            pl.col("ts").max().alias("last_trade_ts"),
        )
    )
    position_by_pair = _rows_by_pair(
        positions.group_by("pair_id").agg(
            pl.len().alias("total_positions"),
            pl.col("is_closed").sum().alias("closed_positions"),
            (~pl.col("is_closed")).sum().alias("open_positions"),
            pl.col("net_pnl").filter(pl.col("is_closed")).sum().alias("realized_pnl"),
            pl.col("open_gross_notional").filter(pl.col("is_closed")).sum().alias("closed_open_gross_notional"),
            (pl.col("net_pnl").filter(pl.col("is_closed")) > 0).sum().alias("wins"),
            (pl.col("net_pnl").filter(pl.col("is_closed")) < 0).sum().alias("losses"),
            pl.col("net_pnl").filter(pl.col("is_closed") & (pl.col("net_pnl") > 0)).mean().alias("avg_win"),
            (-pl.col("net_pnl")).filter(pl.col("is_closed") & (pl.col("net_pnl") < 0)).mean().alias("avg_loss"),
            pl.col("holding_minutes").filter(pl.col("is_closed")).mean().alias("avg_holding_minutes"),
            pl.col("holding_minutes").filter(pl.col("is_closed")).median().alias("median_holding_minutes"),
            pl.col("holding_minutes").filter(pl.col("is_closed")).max().alias("max_holding_minutes"),
        )
    )
    reconstructed_curves = _reconstructed_pair_curves(
        trade_df,
        position_df,
        pair_symbols,
        funding_events,
        price_source=price_source,
        timeline=timeline,
        final_position_valuation=final_position_valuation,
    )
    final_position_values = _final_pair_position_values(
        position_df,
        pair_symbols,
        reconstructed_curves=reconstructed_curves,
        final_position_valuation=final_position_valuation,
    )
    open_holding = _open_holding_stats(positions, position_df)
    drawdowns = _pair_drawdowns(
        trade_df, position_df, pair_symbols, initial_equity, funding_events,
        reconstructed_curves=reconstructed_curves,
    )

    rows = []
    for pair_id in sorted(pair_ids):
        symbols = pair_symbols.get(pair_id)
        cash = cash_by_pair.get(pair_id, {})
        stats = position_by_pair.get(pair_id, {})
        final_position_value = final_position_values.get(pair_id, 0.0)
        cashflow_pnl = _float(cash.get("cashflow_pnl"), 0.0)
        funding_fee = _float(funding_by_pair.get(pair_id), 0.0)
        realized_pnl = _float(stats.get("realized_pnl"), 0.0)
        unrealized_pnl = cashflow_pnl - funding_fee - realized_pnl + final_position_value
        total_pnl = realized_pnl + unrealized_pnl
        closed_positions = int(_float(stats.get("closed_positions"), 0.0))
        wins = int(_float(stats.get("wins"), 0.0))
        losses = int(_float(stats.get("losses"), 0.0))
        total_open_gross = _float(cash.get("total_open_gross_notional"), 0.0)
        closed_open_gross = _float(stats.get("closed_open_gross_notional"), 0.0)
        avg_loss = _float(stats.get("avg_loss"), None)
        avg_win = _float(stats.get("avg_win"), None)

        row = {
            "pair_id": pair_id,
            "x_symbol": symbols[0] if symbols else None,
            "y_symbol": symbols[1] if symbols else None,
            "total_positions": int(_float(stats.get("total_positions"), 0.0)),
            "closed_positions": closed_positions,
            "open_positions": int(_float(stats.get("open_positions"), 0.0)),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / closed_positions if closed_positions else None,
            "profit_loss_ratio": avg_win / avg_loss if avg_win is not None and avg_loss else None,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": total_pnl,
            "total_return_on_open_gross": total_pnl / total_open_gross if total_open_gross else None,
            "realized_return_on_closed_open_gross": realized_pnl / closed_open_gross if closed_open_gross else None,
            "contribution_to_initial_equity": total_pnl / initial_equity if initial_equity else None,
            "total_open_gross_notional": total_open_gross,
            "total_trade_notional": _float(cash.get("total_trade_notional"), 0.0),
            "total_fee": _float(cash.get("total_fee"), 0.0),
            "total_slippage": _float(cash.get("total_slippage"), 0.0),
            "funding_fee": funding_fee,
            "avg_holding_days": _minutes_to_days(stats.get("avg_holding_minutes")),
            "median_holding_days": _minutes_to_days(stats.get("median_holding_minutes")),
            "max_holding_days": _minutes_to_days(stats.get("max_holding_minutes")),
            "open_avg_holding_days": open_holding.get(pair_id, {}).get("avg_open_days"),
            "open_max_holding_days": open_holding.get(pair_id, {}).get("max_open_days"),
            "final_position_value": final_position_value,
            "pair_max_drawdown": drawdowns.get(pair_id, {}).get("max_drawdown"),
            "pair_max_drawdown_pct_initial_equity": drawdowns.get(pair_id, {}).get("max_drawdown_pct_initial_equity"),
            "first_trade_ts": _int_or_none(cash.get("first_trade_ts")),
            "last_trade_ts": _int_or_none(cash.get("last_trade_ts")),
        }
        rows.append(row)

    return sorted(rows, key=lambda row: _sort_value(row.get("total_pnl")), reverse=True)


def compact_pair_summary_rows(pair_summary: list[dict], formatted: bool = False) -> list[dict]:
    rows = []
    for row in pair_summary or []:
        longest_holding_days = _max_optional(row.get("max_holding_days"), row.get("open_max_holding_days"))
        if formatted:
            rows.append(
                {
                    "交易对": row.get("pair_id"),
                    "交易笔数": _format_int(row.get("total_positions")),
                    "胜率": _format_percent(row.get("win_rate")),
                    "收益率": _format_percent(row.get("contribution_to_initial_equity")),
                    "总盈亏": _format_number(row.get("total_pnl"), 0),
                    "最大回撤率": _format_percent(row.get("pair_max_drawdown_pct_initial_equity")),
                    "平均持仓天": _format_number(row.get("avg_holding_days"), 1),
                    "最长持仓天": _format_number(longest_holding_days, 1),
                }
            )
            continue
        rows.append(
            {
                "交易对": row.get("pair_id"),
                "交易笔数": int(_float(row.get("total_positions"), 0.0)),
                "胜率": _float(row.get("win_rate"), None),
                "收益率": _float(row.get("contribution_to_initial_equity"), None),
                "总盈亏": _float(row.get("total_pnl"), None),
                "最大回撤率": _float(row.get("pair_max_drawdown_pct_initial_equity"), None),
                "平均持仓天": _float(row.get("avg_holding_days"), None),
                "最长持仓天": _float(longest_holding_days, None),
            }
        )
    return rows


def compact_pair_summary_payload(pair_summary: list[dict]) -> list[dict]:
    rows = []
    for row in pair_summary or []:
        rows.append(
            {
                "pair_id": row.get("pair_id"),
                "total_positions": int(_float(row.get("total_positions"), 0.0)),
                "win_rate": _float(row.get("win_rate"), None),
                "contribution_to_initial_equity": _float(row.get("contribution_to_initial_equity"), None),
                "total_pnl": _float(row.get("total_pnl"), None),
                "pair_max_drawdown": _float(row.get("pair_max_drawdown"), None),
                "pair_max_drawdown_pct_initial_equity": _float(row.get("pair_max_drawdown_pct_initial_equity"), None),
                "avg_holding_days": _float(row.get("avg_holding_days"), None),
                "max_holding_days": _float(row.get("max_holding_days"), None),
                "open_max_holding_days": _float(row.get("open_max_holding_days"), None),
            }
        )
    return rows


def compact_pair_summary_markdown(pair_summary: list[dict]) -> str:
    rows = compact_pair_summary_rows(pair_summary, formatted=True)
    lines = [
        "| 交易对 | 交易笔数 | 胜率 | 收益率 | 总盈亏 | 最大回撤率 | 平均持仓天 | 最长持仓天 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {交易对} | {交易笔数} | {胜率} | {收益率} | {总盈亏} | {最大回撤率} | {平均持仓天} | {最长持仓天} |".format(
                **row
            )
        )
    return "\n".join(lines)


def write_compact_pair_summary_csv(path: Path, pair_summary: list[dict]) -> None:
    rows = compact_pair_summary_rows(pair_summary)
    frame = pl.DataFrame(rows, infer_schema_length=None)
    if frame.is_empty():
        path.write_text("", encoding="utf-8-sig")
        return
    path.write_text(frame.write_csv(), encoding="utf-8-sig")


def _empty_pair_row(pair_id: str, symbols: tuple | None, initial_equity: float | None) -> dict:
    return {
        "pair_id": pair_id,
        "x_symbol": symbols[0] if symbols else None,
        "y_symbol": symbols[1] if symbols else None,
        "total_positions": 0,
        "closed_positions": 0,
        "open_positions": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": None,
        "profit_loss_ratio": None,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "total_pnl": 0.0,
        "total_return_on_open_gross": None,
        "realized_return_on_closed_open_gross": None,
        "contribution_to_initial_equity": 0.0 if initial_equity else None,
        "total_open_gross_notional": 0.0,
        "total_trade_notional": 0.0,
        "total_fee": 0.0,
        "total_slippage": 0.0,
        "funding_fee": 0.0,
        "avg_holding_days": None,
        "median_holding_days": None,
        "max_holding_days": None,
        "open_avg_holding_days": None,
        "open_max_holding_days": None,
        "final_position_value": 0.0,
        "pair_max_drawdown": None,
        "pair_max_drawdown_pct_initial_equity": None,
        "first_trade_ts": None,
        "last_trade_ts": None,
    }


def _to_frame(data: Any) -> pl.DataFrame:
    if data is None:
        return pl.DataFrame()
    if isinstance(data, pl.DataFrame):
        return data
    return reporting_frame(data)


def _normalize_trades(trades: pl.DataFrame) -> pl.DataFrame:
    if trades.is_empty():
        return trades
    for column in [
        "pair_id", "position_id", "group_id", "action", "side", "symbol", "exchange"
    ]:
        if column not in trades.columns:
            trades = trades.with_columns(pl.lit(None).alias(column))
    for column in ["quantity", "price", "notional", "fee", "slippage", "ts"]:
        if column not in trades.columns:
            trades = trades.with_columns(pl.lit(0.0 if column != "ts" else None).alias(column))
    return trades.with_columns(
        pl.col("pair_id").cast(pl.Utf8, strict=False),
        pl.col("position_id").cast(pl.Utf8, strict=False),
        pl.col("group_id").cast(pl.Utf8, strict=False),
        pl.col("symbol").cast(pl.Utf8, strict=False),
        pl.col("exchange").cast(pl.Utf8, strict=False),
        pl.col("action").cast(pl.Utf8, strict=False).str.to_lowercase(),
        pl.col("side").cast(pl.Utf8, strict=False).str.to_lowercase(),
        pl.col("notional").cast(pl.Float64, strict=False).abs().fill_null(0.0),
        pl.col("fee").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("slippage").cast(pl.Float64, strict=False).fill_null(0.0),
        pl.col("ts").cast(pl.Int64, strict=False),
        pl.col("quantity").cast(pl.Float64, strict=False).abs().fill_null(0.0),
    )


def _normalize_funding(funding: pl.DataFrame) -> pl.DataFrame:
    if funding.is_empty():
        return funding
    for column in ["exchange", "symbol", "ts", "funding_rate", "mark_price", "payment"]:
        if column not in funding.columns:
            funding = funding.with_columns(pl.lit(None).alias(column))
    return funding.with_columns(
        pl.col("exchange").cast(pl.Utf8, strict=False),
        pl.col("symbol").cast(pl.Utf8, strict=False),
        pl.col("ts").cast(pl.Int64, strict=False),
        pl.col("funding_rate").cast(pl.Float64, strict=False),
        pl.col("mark_price").cast(pl.Float64, strict=False),
        pl.col("payment").cast(pl.Float64, strict=False),
    ).drop_nulls(["symbol", "ts"])


def _funding_allocations(
    trades: pl.DataFrame,
    funding: pl.DataFrame,
) -> tuple[dict[tuple[str, str], float], dict[str, float], list[dict]]:
    """Allocate each exchange funding payment to the positions that created it.

    Funding is settled before fills with the same timestamp, matching the
    exchange simulator.  New records use quantity * mark * rate.  Legacy
    records without mark/rate are split from the recorded payment by signed
    position quantity so their allocations still sum to the portfolio charge.
    """
    if trades.is_empty() or funding.is_empty():
        return {}, {}, []

    trade_rows = sorted(
        trades.to_dicts(), key=lambda row: (int(row.get("ts") or 0), str(row.get("group_id") or ""))
    )
    funding_rows = sorted(
        funding.to_dicts(), key=lambda row: (int(row.get("ts") or 0), str(row.get("symbol") or ""))
    )
    quantities: dict[tuple[str, str, str | None, str], float] = {}
    by_position: dict[tuple[str, str], float] = {}
    by_pair: dict[str, float] = {}
    pair_events: list[dict] = []
    trade_index = 0

    for event in funding_rows:
        event_ts = int(event.get("ts") or 0)
        while trade_index < len(trade_rows) and int(trade_rows[trade_index].get("ts") or 0) < event_ts:
            trade = trade_rows[trade_index]
            pair_id = str(trade.get("pair_id") or "")
            position_key = str(trade.get("position_key") or trade.get("group_id") or pair_id)
            symbol = str(trade.get("symbol") or "")
            exchange = trade.get("exchange")
            quantity = _float(trade.get("quantity"), 0.0)
            signed = quantity if str(trade.get("side") or "").lower() == "buy" else -quantity
            key = (pair_id, position_key, exchange, symbol)
            quantities[key] = quantities.get(key, 0.0) + signed
            trade_index += 1

        symbol = str(event.get("symbol") or "")
        exchange = event.get("exchange")
        active = {
            key: quantity
            for key, quantity in quantities.items()
            if key[3] == symbol
            and (exchange is None or key[2] is None or key[2] == exchange)
            and abs(quantity) > 1e-12
        }
        if not active:
            continue

        rate = _float(event.get("funding_rate"), None)
        mark = _float(event.get("mark_price"), None)
        recorded_payment = _float(event.get("payment"), None)
        if rate is not None and mark is not None:
            allocations = {key: quantity * mark * rate for key, quantity in active.items()}
        else:
            total_quantity = sum(active.values())
            if recorded_payment is None or abs(total_quantity) <= 1e-12:
                continue
            allocations = {
                key: recorded_payment * quantity / total_quantity
                for key, quantity in active.items()
            }

        for key, payment in allocations.items():
            pair_id, position_key, _, _ = key
            position_id = (pair_id, position_key)
            by_position[position_id] = by_position.get(position_id, 0.0) + payment
            by_pair[pair_id] = by_pair.get(pair_id, 0.0) + payment
            pair_events.append({"pair_id": pair_id, "ts": event_ts, "payment": payment})

    return by_position, by_pair, pair_events


def _rows_by_pair(frame: pl.DataFrame) -> dict[str, dict]:
    if frame.is_empty() or "pair_id" not in frame.columns:
        return {}
    return {str(row["pair_id"]): row for row in frame.to_dicts()}


def _final_pair_position_values(
    position_curve: pl.DataFrame,
    pair_symbols: dict[str, tuple],
    reconstructed_curves: dict[str, pl.DataFrame] | None = None,
    final_position_valuation: dict | None = None,
) -> dict[str, float]:
    """Return final signed position value per pair.

    Newer runs intentionally omit per-symbol columns from ``position_curve``.
    In that format the value is reconstructed from fills and raw close prices;
    the legacy columns remain the first choice for backwards compatibility.
    """
    values = {pair_id: 0.0 for pair_id in pair_symbols}
    if not position_curve.is_empty():
        last = position_curve.tail(1)
        for pair_id, symbols in pair_symbols.items():
            columns = [f"{symbol}_position_value" for symbol in symbols]
            available = [column for column in columns if column in last.columns]
            if available:
                values[pair_id] = sum(_float(last[column][0], 0.0) for column in available)

    for pair_id, curve in (reconstructed_curves or {}).items():
        if curve.is_empty() or "position_value" not in curve.columns:
            continue
        values[pair_id] = _float(curve.tail(1)["position_value"][0], values.get(pair_id, 0.0))

    # Last-resort support for reports that have neither symbol columns nor raw
    # data paths.  ``final_position_valuation`` contains aggregate symbol
    # quantities/marks, so it is only used when a pair has a single unambiguous
    # symbol contribution; raw-price reconstruction is preferred above.
    if final_position_valuation:
        marks = {}
        for exchange_positions in (final_position_valuation.get("positions", {}) or {}).values():
            for symbol, item in (exchange_positions or {}).items():
                marks[symbol] = _float((item or {}).get("mark_price"), None)
        for pair_id, symbols in pair_symbols.items():
            if pair_id in (reconstructed_curves or {}):
                continue
            # No quantity is available at this layer; retain the legacy zero
            # rather than treating aggregate symbol notional as pair PnL.
            if not any(symbol in marks for symbol in symbols):
                values[pair_id] = 0.0
    return values


def _reconstructed_pair_curves(
    trades: pl.DataFrame,
    position_curve: pl.DataFrame,
    pair_symbols: dict[str, tuple],
    funding_events: list[dict],
    *,
    price_source: Path | str | None = None,
    timeline: Any = None,
    final_position_valuation: dict | None = None,
) -> dict[str, pl.DataFrame]:
    """Rebuild pair valuation paths when symbol columns were compacted away."""
    if trades.is_empty() or not pair_symbols:
        return {}
    config = _read_report_config(price_source)
    price_cache: dict[str, pl.DataFrame] = {}
    timeline_frame = _timeline_frame(position_curve, timeline)
    results: dict[str, pl.DataFrame] = {}
    for pair_id, symbols in pair_symbols.items():
        if all(f"{symbol}_position_value" in position_curve.columns for symbol in symbols):
            continue
        pair_trades = trades.filter(pl.col("pair_id") == pair_id)
        if pair_trades.is_empty():
            continue
        x_prices = _load_close_frame(config, symbols[0], price_cache)
        y_prices = _load_close_frame(config, symbols[1], price_cache)
        pair_timeline = timeline_frame
        if pair_timeline.is_empty():
            ts_frames = [frame.select("ts") for frame in (x_prices, y_prices) if not frame.is_empty()]
            ts_frames.append(pair_trades.select("ts"))
            funding_ts = [
                {"ts": int(row["ts"])}
                for row in funding_events
                if row.get("pair_id") == pair_id and row.get("ts") is not None
            ]
            if funding_ts:
                ts_frames.append(pl.DataFrame(funding_ts))
            if ts_frames:
                pair_timeline = pl.concat(ts_frames, how="vertical_relaxed").unique().sort("ts")
        if pair_timeline.is_empty():
            continue

        price_timeline = _asof_close(pair_timeline, x_prices, "x_close")
        price_timeline = _asof_close(price_timeline, y_prices, "y_close")
        events = pair_trades.select(
            "ts",
            pl.when(pl.col("symbol") == symbols[0])
            .then(pl.when(pl.col("side") == "buy").then(pl.col("quantity")).otherwise(-pl.col("quantity")))
            .otherwise(0.0)
            .alias("x_delta"),
            pl.when(pl.col("symbol") == symbols[1])
            .then(pl.when(pl.col("side") == "buy").then(pl.col("quantity")).otherwise(-pl.col("quantity")))
            .otherwise(0.0)
            .alias("y_delta"),
            pl.col("cashflow"),
        ).group_by("ts").agg(
            pl.col("x_delta").sum(),
            pl.col("y_delta").sum(),
            pl.col("cashflow").sum(),
        )
        funding = pl.DataFrame(
            [
                {"ts": int(row["ts"]), "funding_cashflow": -float(row["payment"])}
                for row in funding_events
                if row.get("pair_id") == pair_id and row.get("ts") is not None
            ],
            schema={"ts": pl.Int64, "funding_cashflow": pl.Float64},
        )
        curve = (
            price_timeline.join(events, on="ts", how="left")
            .join(funding, on="ts", how="left")
            .with_columns(
                pl.col("x_delta").fill_null(0.0),
                pl.col("y_delta").fill_null(0.0),
                pl.col("cashflow").fill_null(0.0),
                pl.col("funding_cashflow").fill_null(0.0),
                pl.col("x_close").forward_fill(),
                pl.col("y_close").forward_fill(),
            )
            .with_columns(
                pl.col("x_delta").cum_sum().alias("x_quantity"),
                pl.col("y_delta").cum_sum().alias("y_quantity"),
                pl.col("cashflow").cum_sum().alias("cum_cashflow"),
                pl.col("funding_cashflow").cum_sum().alias("cum_funding"),
            )
            .with_columns(
                (
                    pl.col("x_quantity") * pl.col("x_close").fill_null(0.0)
                    + pl.col("y_quantity") * pl.col("y_close").fill_null(0.0)
                ).alias("position_value"),
            )
            .with_columns(
                (pl.col("cum_cashflow") + pl.col("position_value") + pl.col("cum_funding")).alias("pair_pnl")
            )
            .select("ts", "position_value", "pair_pnl")
        )
        results[pair_id] = curve
    return results


def _read_report_config(source: Path | str | None) -> dict:
    if source is None:
        return {}
    path = Path(source)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _timeline_frame(position_curve: pl.DataFrame, timeline: Any) -> pl.DataFrame:
    if not position_curve.is_empty() and "ts" in position_curve.columns:
        return position_curve.select(pl.col("ts").cast(pl.Int64, strict=False)).drop_nulls().unique().sort("ts")
    if timeline is None:
        return pl.DataFrame(schema={"ts": pl.Int64})
    frame = _to_frame(timeline)
    if frame.is_empty() or "ts" not in frame.columns:
        return pl.DataFrame(schema={"ts": pl.Int64})
    return frame.select(pl.col("ts").cast(pl.Int64, strict=False)).drop_nulls().unique().sort("ts")


def _symbol_data_path(config: dict, symbol: str) -> Path | None:
    path = ((config.get("data", {}).get("symbols", {}).get(symbol, {}) or {}).get("path"))
    candidates = []
    if path:
        candidate = Path(path)
        candidates.append(candidate if candidate.is_absolute() else PROJECT_ROOT / candidate)
    candidates.extend([
        PROJECT_ROOT / "data" / symbol / f"{symbol}-1m.csv",
        PROJECT_ROOT / "data" / symbol / f"{symbol}_1m.csv",
    ])
    return next((candidate for candidate in candidates if candidate.exists()), None)


def _load_close_frame(config: dict, symbol: str, cache: dict[str, pl.DataFrame]) -> pl.DataFrame:
    if symbol in cache:
        return cache[symbol]
    path = _symbol_data_path(config, symbol)
    if path is None:
        frame = pl.DataFrame(schema={"ts": pl.Int64, "close": pl.Float64})
    else:
        try:
            frame = load_csv_data(path).select("ts", "close")
        except Exception:
            frame = pl.DataFrame(schema={"ts": pl.Int64, "close": pl.Float64})
    cache[symbol] = frame
    return frame


def _asof_close(timeline: pl.DataFrame, prices: pl.DataFrame, name: str) -> pl.DataFrame:
    if prices.is_empty():
        return timeline.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))
    return timeline.join_asof(
        prices.rename({"close": name}).sort("ts"), on="ts", strategy="backward"
    )


def _open_holding_stats(positions: pl.DataFrame, position_curve: pl.DataFrame) -> dict[str, dict]:
    if positions.is_empty():
        return {}
    if not position_curve.is_empty() and "ts" in position_curve.columns:
        last_ts = _int_or_none(position_curve["ts"].max())
    else:
        last_ts = _int_or_none(positions["close_ts"].max()) or _int_or_none(positions["open_ts"].max())
    if last_ts is None:
        return {}
    open_positions = positions.filter(pl.col("open_ts").is_not_null() & ~pl.col("is_closed"))
    if open_positions.is_empty():
        return {}
    rows = {}
    for pair_id in open_positions["pair_id"].drop_nulls().unique().to_list():
        pair_rows = open_positions.filter(pl.col("pair_id") == pair_id).to_dicts()
        days = [
            (last_ts - int(row["open_ts"])) / 60000.0 / 1440.0
            for row in pair_rows
            if row.get("open_ts") is not None
        ]
        if days:
            rows[str(pair_id)] = {
                "avg_open_days": sum(days) / len(days),
                "max_open_days": max(days),
            }
    return rows


def _pair_drawdowns(
    trades: pl.DataFrame,
    position_curve: pl.DataFrame,
    pair_symbols: dict[str, tuple],
    initial_equity: float | None,
    funding_events: list[dict] | None = None,
    reconstructed_curves: dict[str, pl.DataFrame] | None = None,
) -> dict[str, dict]:
    if trades.is_empty():
        return {}
    result = {}
    for pair_id, symbols in pair_symbols.items():
        reconstructed = (reconstructed_curves or {}).get(pair_id)
        if reconstructed is not None and not reconstructed.is_empty():
            curve = reconstructed
        elif position_curve.is_empty() or "ts" not in position_curve.columns:
            continue
        else:
            value_exprs = []
            for symbol in symbols:
                column = f"{symbol}_position_value"
                if column in position_curve.columns:
                    value_exprs.append(pl.col(column).cast(pl.Float64, strict=False).fill_null(0.0))
            if not value_exprs:
                continue
            pair_position = position_curve.select(
                "ts",
                sum(value_exprs, start=pl.lit(0.0)).alias("position_value"),
            )
            pair_cash = (
                trades.filter(pl.col("pair_id") == pair_id)
                .group_by("ts")
                .agg(pl.col("cashflow").sum().alias("cashflow"))
                .sort("ts")
            )
            pair_funding_rows = [
                {"ts": row["ts"], "cashflow": -float(row["payment"])}
                for row in (funding_events or [])
                if row.get("pair_id") == pair_id
            ]
            if pair_funding_rows:
                pair_cash = pl.concat(
                    [pair_cash, pl.DataFrame(pair_funding_rows, infer_schema_length=None)],
                    how="diagonal_relaxed",
                ).group_by("ts").agg(pl.col("cashflow").sum()).sort("ts")
            curve = (
                pair_position.join(pair_cash, on="ts", how="left")
                .with_columns(pl.col("cashflow").fill_null(0.0).cum_sum().alias("cum_cashflow"))
                .with_columns((pl.col("cum_cashflow") + pl.col("position_value")).alias("pair_pnl"))
            )
        if curve.is_empty():
            continue
        drawdown = (
            curve.with_columns(pl.col("pair_pnl").cum_max().alias("peak"))
            .with_columns((pl.col("pair_pnl") - pl.col("peak")).alias("drawdown"))
            .select(pl.col("drawdown").min())
            .item()
        )
        drawdown = _float(drawdown, None)
        result[pair_id] = {
            "max_drawdown": drawdown,
            "max_drawdown_pct_initial_equity": drawdown / initial_equity if drawdown is not None and initial_equity else None,
        }
    return result


def _minutes_to_days(value) -> float | None:
    minutes = _float(value, None)
    if minutes is None:
        return None
    return minutes / 1440.0


def _float(value, default=None):
    try:
        if value is None:
            return default
        value = float(value)
        if not isfinite(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def _max_optional(*values):
    cleaned = [_float(value, None) for value in values]
    cleaned = [value for value in cleaned if value is not None]
    return max(cleaned) if cleaned else None


def _format_int(value) -> str:
    number = _float(value, None)
    if number is None:
        return "-"
    return f"{int(number):,}"


def _format_number(value, digits: int = 0) -> str:
    number = _float(value, None)
    if number is None:
        return "-"
    return f"{number:,.{digits}f}"


def _format_percent(value) -> str:
    number = _float(value, None)
    if number is None:
        return "-"
    return f"{number * 100:.2f}%"


def _int_or_none(value):
    try:
        if value is None:
            return None
        value = int(value)
        return value
    except (TypeError, ValueError):
        return None


def _sort_value(value):
    value = _float(value, None)
    return value if value is not None else float("-inf")
