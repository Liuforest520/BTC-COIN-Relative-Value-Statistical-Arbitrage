"""Equity curve charts for one backtest run: portfolio + one image per traded pair.

Output layout (inside the run directory)::

    curve_charts/
        portfolio_curve.png          overall equity + drawdown + headline metrics
        pairs/
            <pair_id>_curve.png      one per pair that actually traded

Only pairs with at least one fill get a chart.  Each pair image carries that
pair's own metrics (sharpe, calmar, trades, win rate, max drawdown, ...) so the
file is self-describing without opening the review page.

The pair curve uses the same accounting as the review page::

    pair_pnl(t) = cumulative fill cashflow(t)
                  + qx(t) * Px(t) + qy(t) * Py(t)
                  - cumulative funding(t)

so the sum of the pair curves reconciles with the portfolio equity change.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from math import isfinite, sqrt
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from core.modules.logger import logger

CHART_DIR_NAME = "curve_charts"
PAIR_CHART_DIR_NAME = "pairs"
PORTFOLIO_CHART_NAME = "portfolio_curve.png"

# One sample every 15 minutes keeps the images small while still showing the
# intraday shape of a pair that is only open for a few hours.
DEFAULT_SAMPLE_MINUTES = 15
# Padding around a pair's active window so the entry/exit is visible in context.
WINDOW_PAD_MS = 12 * 60 * 60 * 1000
PRICE_CACHE_SIZE = 6
PRICE_CACHE_DIR = Path("tmp") / "curve_chart_price_cache"
MILLISECONDS_PER_YEAR = 365 * 24 * 60 * 60 * 1000

CJK_FONT_CANDIDATES = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
    "WenQuanYi Zen Hei",
)


@dataclass
class Fill:
    ts: int
    symbol: str
    signed_qty: float
    cashflow: float
    fee: float
    notional: float
    position_id: str
    action: str


@dataclass
class PairTrades:
    pair_id: str
    x_symbol: str
    y_symbol: str
    fills: list[Fill]
    positions: dict[str, dict]

    @property
    def has_trades(self) -> bool:
        return bool(self.fills)


def export_curve_charts(
    result,
    config_path: str | Path,
    output_dir: str | Path,
    initial_equity: float | None = None,
    sample_minutes: int = DEFAULT_SAMPLE_MINUTES,
) -> dict:
    """Write ``curve_charts/`` for one run and return what was written."""
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    chart_dir = output_dir / CHART_DIR_NAME
    pair_chart_dir = chart_dir / PAIR_CHART_DIR_NAME
    pair_chart_dir.mkdir(parents=True, exist_ok=True)

    plt, labels, cjk = _matplotlib_setup()
    written: dict = {
        "dir": chart_dir,
        "portfolio": None,
        "pairs": [],
        "sample_minutes": int(sample_minutes),
        "cjk_labels": cjk,
    }

    metrics = dict(getattr(result, "metrics", {}) or {})
    equity_ts, equity_values = _equity_series(result)
    if initial_equity is None:
        initial_equity = metrics.get("initial_equity")
    if initial_equity is None and len(equity_values):
        initial_equity = float(equity_values[0])

    if len(equity_ts) >= 2:
        path = chart_dir / PORTFOLIO_CHART_NAME
        _plot_portfolio(plt, labels, path, equity_ts, equity_values, metrics, output_dir.name, result)
        written["portfolio"] = path
    else:
        logger.warning("curve charts: portfolio equity curve is empty, skipped")

    pair_defs, symbol_paths = _config_pair_defs(config_path)
    pair_trades = _collect_pair_trades(result, pair_defs)
    if not pair_trades:
        logger.info("curve charts: no pair had fills, only the portfolio chart was written")
        return written

    funding_events = _funding_events(result)
    timeline = (int(equity_ts[0]), int(equity_ts[-1])) if len(equity_ts) else None
    step_ms = max(1, int(sample_minutes)) * 60_000

    for trades in sorted(pair_trades, key=lambda item: item.pair_id):
        if not trades.has_trades:
            continue
        try:
            points = _pair_points(trades, funding_events, symbol_paths, timeline, step_ms)
        except Exception as exc:  # a single bad pair must not abort the export
            logger.warning("curve charts: pair {} failed: {}", trades.pair_id, exc)
            continue
        if points is None or len(points["ts"]) < 2:
            continue
        path = pair_chart_dir / f"{_safe_name(trades.pair_id)}_curve.png"
        stats = _pair_metrics(trades, points, initial_equity)
        _plot_pair(plt, labels, path, trades, points, stats, initial_equity)
        written["pairs"].append({"pair_id": trades.pair_id, "path": path, "metrics": stats})

    logger.info(
        "curve charts: {} portfolio + {} pair images in {}",
        int(written["portfolio"] is not None),
        len(written["pairs"]),
        chart_dir,
    )
    _write_pair_metrics_csv(pair_chart_dir, written["pairs"])
    return written


PAIR_METRICS_COLUMNS = (
    "pair_id",
    "x_symbol",
    "y_symbol",
    "net_pnl",
    "contribution",
    "sharpe",
    "calmar",
    "cagr",
    "max_drawdown",
    "closed_positions",
    "open_positions",
    "win_rate",
    "profit_loss_ratio",
    "avg_holding_minutes",
    "fee",
    "funding",
    "turnover",
    "window_start",
    "window_end",
    "samples",
)


def _write_pair_metrics_csv(pair_chart_dir: Path, entries: list[dict]) -> None:
    """Write the numbers that are drawn on the pair images, in one table."""
    if not entries:
        return
    rows = []
    for entry in entries:
        stats = entry["metrics"]
        rows.append({column: stats.get(column) for column in PAIR_METRICS_COLUMNS})
    try:
        frame = pl.DataFrame(rows, infer_schema_length=None)
        frame.write_csv(pair_chart_dir / "pair_chart_metrics.csv")
    except Exception as exc:
        logger.warning("curve charts: cannot write pair metric table: {}", exc)


# --------------------------------------------------------------------------- #
# data preparation
# --------------------------------------------------------------------------- #

def _equity_series(result) -> tuple[np.ndarray, np.ndarray]:
    equity = getattr(result, "equity_curve", None)
    if not equity:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)

    if isinstance(equity, dict):
        timestamps = equity.get("ts") or []
        values = equity.get("equity") or []
    else:
        timestamps = [row.get("ts") for row in equity]
        values = [row.get("equity") for row in equity]

    ts_out: list[int] = []
    value_out: list[float] = []
    for ts, value in zip(timestamps, values):
        if ts is None or value is None:
            continue
        try:
            ts_int = int(ts)
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if not isfinite(value_float):
            continue
        ts_out.append(ts_int)
        value_out.append(value_float)
    return np.asarray(ts_out, dtype=np.int64), np.asarray(value_out, dtype=np.float64)


def _config_pair_defs(config_path: Path) -> tuple[list[dict], dict[str, Path]]:
    """Return the configured pairs and the data path of every symbol."""
    if not config_path.exists():
        return [], {}
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("curve charts: cannot read {}: {}", config_path, exc)
        return [], {}

    symbols: dict[str, Path] = {}
    for symbol, info in (raw.get("data", {}).get("symbols") or {}).items():
        path = (info or {}).get("path")
        if path:
            symbols[str(symbol)] = Path(str(path))
        else:
            for candidate in (Path("data") / str(symbol) / f"{symbol}-1m.csv",):
                if candidate.exists():
                    symbols[str(symbol)] = candidate
                    break

    setup = (raw.get("setups") or {}).get(raw.get("active_setup")) or {}
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
                "pair_id": str(item.get("pair_id") or f"{x_symbol}_{y_symbol}"),
                "x_symbol": str(x_symbol),
                "y_symbol": str(y_symbol),
            }
        )
    return pairs, symbols


def _collect_pair_trades(result, pair_defs: list[dict]) -> list[PairTrades]:
    by_id = {item["pair_id"]: item for item in pair_defs}
    buckets: dict[str, PairTrades] = {}

    for trade in getattr(result, "trades", None) or []:
        pair_id = getattr(trade, "pair_id", None)
        symbol = getattr(trade, "symbol", None)
        if not pair_id or not symbol:
            continue
        pair_id = str(pair_id)
        symbol = str(symbol)
        definition = by_id.get(pair_id)
        if definition is None:
            definition = {"pair_id": pair_id, "x_symbol": symbol, "y_symbol": symbol}
            by_id[pair_id] = definition
        bucket = buckets.get(pair_id)
        if bucket is None:
            bucket = PairTrades(
                pair_id=pair_id,
                x_symbol=definition["x_symbol"],
                y_symbol=definition["y_symbol"],
                fills=[],
                positions={},
            )
            buckets[pair_id] = bucket

        try:
            ts = int(getattr(trade, "ts", 0) or 0)
            quantity = abs(float(getattr(trade, "quantity", 0.0) or 0.0))
            price = float(getattr(trade, "price", 0.0) or 0.0)
            fee = float(getattr(trade, "fee", 0.0) or 0.0)
            notional = getattr(trade, "notional", None)
            notional = abs(float(notional)) if notional is not None else abs(quantity * price)
        except (TypeError, ValueError):
            continue
        side = str(getattr(trade, "side", "") or "").lower()
        action = str(getattr(trade, "action", "") or "").lower()
        signing = 1.0 if side == "buy" else -1.0
        cashflow = notional - fee if side == "sell" else -(notional + fee)

        bucket.fills.append(
            Fill(
                ts=ts,
                symbol=symbol,
                signed_qty=signing * quantity,
                cashflow=cashflow,
                fee=fee,
                notional=notional,
                position_id=str(getattr(trade, "position_id", "") or ""),
                action=action,
            )
        )
        position_id = str(getattr(trade, "position_id", "") or "")
        if not position_id:
            continue
        record = bucket.positions.setdefault(
            position_id, {"cashflow": 0.0, "fee": 0.0, "open_ts": None, "close_ts": None}
        )
        record["cashflow"] += cashflow
        record["fee"] += fee
        if action == "open":
            record["open_ts"] = ts if record["open_ts"] is None else min(record["open_ts"], ts)
        elif action == "close":
            record["close_ts"] = ts if record["close_ts"] is None else max(record["close_ts"], ts)

    for bucket in buckets.values():
        bucket.fills.sort(key=lambda fill: fill.ts)
    return list(buckets.values())


def _funding_events(result) -> list[dict]:
    events = []
    for payment in getattr(result, "funding_payments", None) or []:
        symbol = str(getattr(payment, "symbol", "") or "")
        if not symbol:
            continue
        try:
            events.append(
                {
                    "ts": int(getattr(payment, "ts", 0) or 0),
                    "symbol": symbol,
                    "rate": _optional_float(getattr(payment, "funding_rate", None)),
                    "mark": _optional_float(getattr(payment, "mark_price", None)),
                    "payment": _optional_float(getattr(payment, "payment", None)),
                }
            )
        except (TypeError, ValueError):
            continue
    events.sort(key=lambda item: item["ts"])
    return events


@lru_cache(maxsize=PRICE_CACHE_SIZE)
def _load_prices(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load ``(ts, close)`` for one symbol CSV, cached across pairs and runs.

    Parsing a 1m CSV costs a couple of seconds, and one run needs two of them
    per traded pair; a sweep would re-parse the same files over and over.  The
    arrays are therefore also cached on disk under ``tmp/`` and re-used while
    the source file's size/mtime are unchanged.
    """
    cached = _read_price_cache(path)
    if cached is not None:
        return cached

    from core.modules.data import load_csv_data

    frame = load_csv_data(Path(path))
    if frame.is_empty() or "ts" not in frame.columns or "close" not in frame.columns:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    frame = frame.select(["ts", "close"]).sort("ts")
    ts = frame["ts"].cast(pl.Int64, strict=False).to_numpy()
    close = frame["close"].cast(pl.Float64, strict=False).to_numpy()
    mask = np.isfinite(close.astype(float)) & (close > 0)
    ts = ts[mask].astype(np.int64)
    close = close[mask].astype(np.float64)
    _write_price_cache(path, ts, close)
    return ts, close


def _price_cache_file(path: str) -> Path | None:
    try:
        source = Path(path)
        stat = source.stat()
    except OSError:
        return None
    # A stable digest: Python's str hash is salted per process, so it would
    # never hit across runs.
    digest = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:12]
    key = f"{digest}_{stat.st_size}_{int(stat.st_mtime)}"
    return PRICE_CACHE_DIR / f"{_safe_name(source.stem)}_{key}.npz"


def _read_price_cache(path: str) -> tuple[np.ndarray, np.ndarray] | None:
    cache_file = _price_cache_file(path)
    if cache_file is None or not cache_file.exists():
        return None
    try:
        with np.load(cache_file) as data:
            return (
                data["ts"].astype(np.int64),
                data["close"].astype(np.float64),
            )
    except Exception:
        return None


def _write_price_cache(path: str, ts: np.ndarray, close: np.ndarray) -> None:
    cache_file = _price_cache_file(path)
    if cache_file is None:
        return
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, ts=ts, close=close)
    except Exception as exc:  # a cache miss must never break the export
        logger.debug("curve charts: cannot write price cache {}: {}", cache_file, exc)


def _price_at(prices: tuple[np.ndarray, np.ndarray], ts: int, fallback: float | None) -> float | None:
    ts_array, close_array = prices
    if len(ts_array) == 0:
        return fallback
    index = int(np.searchsorted(ts_array, ts, side="right")) - 1
    if index < 0:
        return fallback
    return float(close_array[index])


def _pair_points(
    trades: PairTrades,
    funding_events: list[dict],
    symbol_paths: dict[str, Path],
    timeline: tuple[int, int] | None,
    step_ms: int,
) -> dict | None:
    fills = trades.fills
    if not fills:
        return None

    start = fills[0].ts - WINDOW_PAD_MS
    end = fills[-1].ts + WINDOW_PAD_MS
    if timeline is not None:
        start = max(start, int(timeline[0]))
        end = min(end, int(timeline[1]))
    if end <= start:
        start, end = fills[0].ts, fills[-1].ts

    grid = list(range(start, end + 1, step_ms))
    if grid[-1] != end:
        grid.append(end)
    for fill in fills:  # make sure every fill is itself a sample
        grid.append(fill.ts)
    grid = sorted(set(grid))

    prices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for symbol in (trades.x_symbol, trades.y_symbol):
        path = symbol_paths.get(symbol)
        prices[symbol] = _load_prices(str(path)) if path else (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
        )

    symbol_quantities: dict[str, float] = {trades.x_symbol: 0.0, trades.y_symbol: 0.0}
    last_price: dict[str, float | None] = {trades.x_symbol: None, trades.y_symbol: None}
    cum_cashflow = 0.0
    cum_funding = 0.0

    funding_by_pair = _pair_funding_events([trades], funding_events).get(trades.pair_id, [])
    funding_index = 0
    fill_index = 0
    total_funding = 0.0

    ts_out: list[int] = []
    pnl_out: list[float] = []
    px_out: list[float] = []
    py_out: list[float] = []

    for ts in grid:
        while fill_index < len(fills) and fills[fill_index].ts <= ts:
            fill = fills[fill_index]
            cum_cashflow += fill.cashflow
            if fill.symbol in symbol_quantities:
                symbol_quantities[fill.symbol] += fill.signed_qty
            fill_index += 1
        while funding_index < len(funding_by_pair) and funding_by_pair[funding_index]["ts"] <= ts:
            cum_funding += funding_by_pair[funding_index]["payment"]
            funding_index += 1

        px = _price_at(prices[trades.x_symbol], ts, last_price[trades.x_symbol])
        py = _price_at(prices[trades.y_symbol], ts, last_price[trades.y_symbol])
        if px is not None:
            last_price[trades.x_symbol] = px
        if py is not None:
            last_price[trades.y_symbol] = py

        pnl = cum_cashflow - cum_funding
        if px is not None:
            pnl += symbol_quantities[trades.x_symbol] * px
        if py is not None:
            pnl += symbol_quantities[trades.y_symbol] * py

        ts_out.append(ts)
        pnl_out.append(pnl)
        px_out.append(px if px is not None else float("nan"))
        py_out.append(py if py is not None else float("nan"))

    total_funding = cum_funding
    return {
        "ts": np.asarray(ts_out, dtype=np.int64),
        "pnl": np.asarray(pnl_out, dtype=np.float64),
        "x_price": np.asarray(px_out, dtype=np.float64),
        "y_price": np.asarray(py_out, dtype=np.float64),
        "funding": float(total_funding),
    }


def _pair_funding_events(pair_trades: list[PairTrades], funding_events: list[dict]) -> dict[str, list[dict]]:
    """Allocate every funding payment to the pairs holding that symbol at that ts.

    Mirrors the reporting convention: settlement happens before fills with the
    same timestamp, the amount is ``signed_quantity * mark * rate`` when the
    record carries them, otherwise the recorded payment is split pro rata by
    signed quantity.
    """
    result: dict[str, list[dict]] = {trades.pair_id: [] for trades in pair_trades}
    if not funding_events:
        return result

    replay = []
    for trades in pair_trades:
        replay.append(
            {
                "pair_id": trades.pair_id,
                "fills": trades.fills,
                "fx": trades.x_symbol,
                "fy": trades.y_symbol,
                "index": 0,
                "qty": {trades.x_symbol: 0.0, trades.y_symbol: 0.0},
            }
        )

    for event in funding_events:
        symbol = event["symbol"]
        event_ts = event["ts"]
        active = []
        for state in replay:
            while state["index"] < len(state["fills"]) and state["fills"][state["index"]].ts < event_ts:
                fill = state["fills"][state["index"]]
                if fill.symbol in state["qty"]:
                    state["qty"][fill.symbol] += fill.signed_qty
                state["index"] += 1
            quantity = state["qty"].get(symbol, 0.0)
            if abs(quantity) > 1e-12:
                active.append((state["pair_id"], quantity))
        if not active:
            continue

        rate, mark, payment = event["rate"], event["mark"], event["payment"]
        if rate is not None and mark is not None:
            allocations = [(pair_id, quantity * mark * rate) for pair_id, quantity in active]
        else:
            total_quantity = sum(quantity for _pair_id, quantity in active)
            if payment is None or abs(total_quantity) <= 1e-12:
                continue
            allocations = [(pair_id, payment * quantity / total_quantity) for pair_id, quantity in active]

        for pair_id, amount in allocations:
            result.setdefault(pair_id, []).append({"ts": event_ts, "payment": amount})
    return result


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def _pair_metrics(trades: PairTrades, points: dict, initial_equity: float | None) -> dict:
    pnl = points["pnl"]
    ts = points["ts"]
    equity = pnl + (float(initial_equity) if initial_equity else 0.0)

    max_drawdown = None
    sharpe = None
    calmar = None
    cagr = None
    if len(equity) >= 2:
        peak = np.maximum.accumulate(equity)
        with np.errstate(divide="ignore", invalid="ignore"):
            drawdown = np.where(peak > 0, equity / peak - 1.0, 0.0)
        max_drawdown = float(np.nanmin(drawdown))

        spacing_ms = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.0
        if spacing_ms > 0:
            factor = MILLISECONDS_PER_YEAR / spacing_ms
            with np.errstate(divide="ignore", invalid="ignore"):
                returns = np.diff(equity) / equity[:-1]
            returns = returns[np.isfinite(returns)]
            if len(returns) > 2:
                std = float(np.std(returns, ddof=1))
                if std > 0:
                    sharpe = float(np.mean(returns) * factor / (std * sqrt(factor)))
        elapsed_years = (int(ts[-1]) - int(ts[0])) / MILLISECONDS_PER_YEAR
        if elapsed_years > 0 and equity[0] > 0 and equity[-1] > 0:
            cagr = float((equity[-1] / equity[0]) ** (1.0 / elapsed_years) - 1.0)
        if cagr is not None and max_drawdown is not None and max_drawdown < 0:
            calmar = float(cagr / abs(max_drawdown))

    closed = [record for record in trades.positions.values() if record["close_ts"] is not None]
    wins = [record["cashflow"] for record in closed if record["cashflow"] > 0]
    losses = [record["cashflow"] for record in closed if record["cashflow"] < 0]
    holdings = [
        (record["close_ts"] - record["open_ts"]) / 60_000.0
        for record in closed
        if record["open_ts"] is not None and record["close_ts"] is not None
    ]

    fee = sum(fill.fee for fill in trades.fills)
    notional = sum(fill.notional for fill in trades.fills)
    final_pnl = float(pnl[-1]) if len(pnl) else 0.0
    return {
        "pair_id": trades.pair_id,
        "x_symbol": trades.x_symbol,
        "y_symbol": trades.y_symbol,
        "net_pnl": final_pnl,
        "contribution": (final_pnl / initial_equity) if initial_equity else None,
        "sharpe": sharpe,
        "calmar": calmar,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "closed_positions": len(closed),
        "open_positions": len(trades.positions) - len(closed),
        "win_rate": (len(wins) / len(closed)) if closed else None,
        "profit_loss_ratio": (
            (sum(wins) / len(wins)) / (abs(sum(losses)) / len(losses))
            if wins and losses else None
        ),
        "avg_holding_minutes": (sum(holdings) / len(holdings)) if holdings else None,
        "fee": fee,
        "funding": points["funding"],
        "turnover": notional,
        "window_start": int(ts[0]) if len(ts) else None,
        "window_end": int(ts[-1]) if len(ts) else None,
        "samples": int(len(ts)),
    }


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #

def _matplotlib_setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cjk = _enable_cjk_font(plt)
    plt.rcParams["axes.unicode_minus"] = False
    return plt, _labels(cjk), cjk


def _enable_cjk_font(plt) -> bool:
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in CJK_FONT_CANDIDATES:
        if candidate in available:
            plt.rcParams["font.sans-serif"] = [candidate, "DejaVu Sans"]
            return True
    return False


def _labels(cjk: bool) -> dict:
    if cjk:
        return {
            "equity": "资金曲线",
            "drawdown": "回撤",
            "pnl": "该 Pair 累计盈亏",
            "portfolio_title": "组合资金曲线",
            "pair_title": "Pair 资金曲线",
            "initial": "初始资金",
            "final": "期末权益",
            "total_return": "总收益",
            "cagr": "年化(CAGR)",
            "sharpe": "夏普",
            "calmar": "卡玛",
            "max_dd": "最大回撤",
            "trades": "交易数(已平仓)",
            "win_rate": "胜率",
            "plr": "盈亏比",
            "net_pnl": "净盈亏",
            "contribution": "贡献(占初始资金)",
            "fee": "手续费",
            "funding": "资金费",
            "slippage": "滑点",
            "avg_hold": "平均持仓",
            "symbols": "标的",
            "window": "区间",
            "open_positions": "期末未平仓",
            "minutes": "分钟",
            "days": "天",
            "samples": "采样点",
        }
    return {
        "equity": "Equity",
        "drawdown": "Drawdown",
        "pnl": "Pair cumulative PnL",
        "portfolio_title": "Portfolio equity curve",
        "pair_title": "Pair equity curve",
        "initial": "Initial equity",
        "final": "Final equity",
        "total_return": "Total return",
        "cagr": "CAGR",
        "sharpe": "Sharpe",
        "calmar": "Calmar",
        "max_dd": "Max drawdown",
        "trades": "Closed trades",
        "win_rate": "Win rate",
        "plr": "Profit/loss ratio",
        "net_pnl": "Net PnL",
        "contribution": "Contribution",
        "fee": "Fee",
        "funding": "Funding",
        "slippage": "Slippage",
        "avg_hold": "Avg holding",
        "symbols": "Symbols",
        "window": "Window",
        "open_positions": "Open at end",
        "minutes": "min",
        "days": "d",
        "samples": "samples",
    }


def _plot_portfolio(plt, labels, path: Path, ts, equity, metrics, title, result) -> None:
    figure = plt.figure(figsize=(16, 9), dpi=150)
    grid = figure.add_gridspec(
        2, 1, height_ratios=[3, 1], hspace=0.12, left=0.06, right=0.98, top=0.80, bottom=0.07
    )
    ax = figure.add_subplot(grid[0])
    ax_dd = figure.add_subplot(grid[1], sharex=ax)

    peak = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.where(peak > 0, equity / peak - 1.0, 0.0)

    ax.plot(ts, equity, color="#2563eb", linewidth=1.2, label=labels["equity"])
    ax.axhline(float(equity[0]), color="#94a3b8", linestyle="--", linewidth=0.9, label=labels["initial"])
    trough = int(np.argmin(drawdown))
    ax.scatter([ts[trough]], [equity[trough]], color="#dc2626", s=28, zorder=5)
    ax.annotate(
        f"{labels['max_dd']} {drawdown[trough]:.2%}",
        xy=(ts[trough], equity[trough]),
        xytext=(8, -18),
        textcoords="offset points",
        color="#dc2626",
        fontsize=9,
    )
    ax.set_ylabel(labels["equity"])
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="upper left", fontsize=9)
    ax.margins(x=0.01)

    ax_dd.fill_between(ts, drawdown * 100.0, 0, color="#f87171", alpha=0.45)
    ax_dd.set_ylabel(f"{labels['drawdown']} %")
    ax_dd.grid(alpha=0.25, linewidth=0.6)
    ax_dd.margins(x=0.01)

    figure.suptitle(f"{labels['portfolio_title']} · {title}", fontsize=14, x=0.06, ha="left", y=0.965)
    lines = [
        f"{labels['initial']:<14}{_money(metrics.get('initial_equity'))}"
        f"    {labels['final']:<12}{_money(metrics.get('final_equity'))}",
        f"{labels['total_return']:<14}{_percent(metrics.get('total_return'))}"
        f"    {labels['cagr']:<12}{_percent(metrics.get('annualized_return'))}"
        f"    {labels['sharpe']:<8}{_number(metrics.get('sharpe'))}"
        f"    {labels['calmar']:<8}{_number(metrics.get('calmar'))}"
        f"    {labels['max_dd']:<10}{_percent(metrics.get('max_drawdown'))}",
        f"{labels['trades']:<14}{_integer(metrics.get('trade_count'))}"
        f"    {labels['win_rate']:<12}{_percent(metrics.get('win_rate'))}"
        f"    {labels['plr']:<8}{_number(metrics.get('profit_loss_ratio'))}"
        f"    {labels['avg_hold']:<8}{_duration(metrics.get('average_holding_minutes'), labels)}"
        f"    {labels['samples']:<8}{len(ts)}",
        f"{labels['fee']:<14}{_money(metrics.get('total_fee'))}"
        f"    {labels['slippage']:<12}{_money(metrics.get('total_slippage'))}"
        f"    {labels['funding']:<8}{_money(metrics.get('funding_fee'))}"
        f"    {labels['window']:<8}{_date(ts[0])} → {_date(ts[-1])}",
    ]
    figure.text(
        0.06,
        0.90,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=10,
        family=_text_family(labels),
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )
    figure.savefig(path)
    plt.close(figure)


def _plot_pair(plt, labels, path: Path, trades: PairTrades, points: dict, stats: dict, initial_equity) -> None:
    ts = points["ts"]
    pnl = points["pnl"]
    base = float(initial_equity) if initial_equity else 0.0
    equity = pnl + base
    peak = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.where(peak > 0, equity / peak - 1.0, 0.0)

    figure = plt.figure(figsize=(15, 8.5), dpi=150)
    grid = figure.add_gridspec(
        2, 1, height_ratios=[3, 1], hspace=0.12, left=0.06, right=0.98, top=0.80, bottom=0.07
    )
    ax = figure.add_subplot(grid[0])
    ax_dd = figure.add_subplot(grid[1], sharex=ax)

    ax.plot(ts, equity, color="#0f766e", linewidth=1.3)
    ax.axhline(base, color="#94a3b8", linestyle="--", linewidth=0.9)
    fill_ts = np.asarray([fill.ts for fill in trades.fills], dtype=np.int64)
    if len(fill_ts):
        positions = np.searchsorted(ts, fill_ts).clip(0, len(ts) - 1)
        ax.scatter(fill_ts, equity[positions], color="#1d4ed8", s=12, alpha=0.6, zorder=5)
    trough = int(np.argmin(drawdown))
    ax.scatter([ts[trough]], [equity[trough]], color="#dc2626", s=30, zorder=6)
    ax.annotate(
        f"{labels['max_dd']} {drawdown[trough]:.2%}",
        xy=(ts[trough], equity[trough]),
        xytext=(8, -18),
        textcoords="offset points",
        color="#dc2626",
        fontsize=9,
    )
    ax.set_ylabel(f"{labels['pnl']} + {labels['initial']}")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.margins(x=0.01)

    ax_dd.fill_between(ts, drawdown * 100.0, 0, color="#f87171", alpha=0.45)
    ax_dd.set_ylabel(f"{labels['drawdown']} %")
    ax_dd.grid(alpha=0.25, linewidth=0.6)
    ax_dd.margins(x=0.01)

    figure.suptitle(
        f"{labels['pair_title']} · {trades.pair_id}",
        fontsize=14,
        x=0.06,
        ha="left",
        y=0.965,
    )
    lines = [
        f"{labels['symbols']:<14}{trades.x_symbol} / {trades.y_symbol}"
        f"    {labels['window']:<8}{_date(ts[0])} → {_date(ts[-1])}"
        f"    {labels['samples']:<8}{stats['samples']}",
        f"{labels['net_pnl']:<14}{_money(stats['net_pnl'])}"
        f"    {labels['contribution']:<12}{_percent(stats['contribution'])}"
        f"    {labels['sharpe']:<8}{_number(stats['sharpe'])}"
        f"    {labels['calmar']:<8}{_number(stats['calmar'])}"
        f"    {labels['max_dd']:<10}{_percent(stats['max_drawdown'])}",
        f"{labels['trades']:<14}{stats['closed_positions']}"
        f"    {labels['win_rate']:<12}{_percent(stats['win_rate'])}"
        f"    {labels['plr']:<8}{_number(stats['profit_loss_ratio'])}"
        f"    {labels['avg_hold']:<8}{_duration(stats['avg_holding_minutes'], labels)}"
        f"    {labels['open_positions']:<8}{stats['open_positions']}",
        f"{labels['fee']:<14}{_money(stats['fee'])}"
        f"    {labels['funding']:<12}{_money(stats['funding'])}"
        f"    {labels['cagr']:<8}{_percent(stats['cagr'])}"
        f"    {labels['initial']:<10}{_money(base)}",
    ]
    figure.text(
        0.06,
        0.90,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=10,
        family=_text_family(labels),
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )
    figure.savefig(path)
    plt.close(figure)


def _text_family(labels: dict) -> str:
    """Monospace only when the labels are pure ASCII.

    A CJK font has no monospace variant here, and forcing ``monospace`` makes
    matplotlib fall back to DejaVu Sans Mono, which cannot draw Chinese glyphs.
    """
    return "monospace" if labels.get("sharpe") == "Sharpe" else "sans-serif"


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #

def _safe_name(value: str) -> str:
    keep = "-_."
    return "".join(char if (char.isalnum() or char in keep) else "_" for char in str(value))


def _optional_float(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) else None


def _money(value) -> str:
    parsed = _optional_float(value)
    return "-" if parsed is None else f"{parsed:,.2f}"


def _percent(value) -> str:
    parsed = _optional_float(value)
    return "-" if parsed is None else f"{parsed:.2%}"


def _number(value) -> str:
    parsed = _optional_float(value)
    return "-" if parsed is None else f"{parsed:.3f}"


def _integer(value) -> str:
    parsed = _optional_float(value)
    return "-" if parsed is None else f"{int(parsed):,d}"


def _duration(minutes, labels) -> str:
    parsed = _optional_float(minutes)
    if parsed is None:
        return "-"
    if parsed >= 1440:
        return f"{parsed / 1440.0:.2f} {labels['days']}"
    return f"{parsed:,.0f} {labels['minutes']}"


def _date(ts: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


# --------------------------------------------------------------------------- #
# regenerate charts from an existing run directory
# --------------------------------------------------------------------------- #

def load_run_result(run_dir: str | Path):
    """Rebuild the minimal result object needed for charts from a finished run.

    Everything comes from the exported CSVs (``equity_curve.csv``,
    ``trades.csv``, ``funding_payments.csv``, ``metrics.json``), so charts can
    be (re)generated without re-running the backtest.
    """
    from types import SimpleNamespace

    run_dir = Path(run_dir)
    if not (run_dir / "trades.csv").exists() and not (run_dir / "equity_curve.csv").exists():
        raise FileNotFoundError(f"no exported CSVs in {run_dir}")

    metrics = {}
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        import json

        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}

    equity = _read_csv_rows(run_dir / "equity_curve.csv", ["ts", "equity"])
    trades = [
        SimpleNamespace(**row)
        for row in _read_csv_rows(
            run_dir / "trades.csv",
            ["ts", "symbol", "side", "quantity", "price", "notional", "fee", "action",
             "position_id", "pair_id"],
        )
    ]
    funding = [
        SimpleNamespace(**row)
        for row in _read_csv_rows(
            run_dir / "funding_payments.csv",
            ["ts", "symbol", "funding_rate", "mark_price", "payment"],
        )
    ]
    return (
        SimpleNamespace(
            metrics=metrics,
            equity_curve=equity,
            trades=trades,
            funding_payments=funding,
        ),
        run_dir / "config.yaml",
    )


def _read_csv_rows(path: Path, wanted: list[str]) -> list[dict]:
    if not path.exists():
        return []
    try:
        frame = pl.read_csv(path, infer_schema_length=10_000)
    except Exception as exc:
        logger.warning("curve charts: cannot read {}: {}", path, exc)
        return []
    columns = [column for column in wanted if column in frame.columns]
    if not columns:
        return []
    return frame.select(columns).to_dicts()


def main(argv: list[str] | None = None) -> int:
    """``python -m core.modules.reporting.curve_charts --run-dir <dir>``."""
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Regenerate curve_charts/ from a finished run directory")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="where curve_charts/ goes (default: the run directory)")
    parser.add_argument("--sample-minutes", type=int, default=DEFAULT_SAMPLE_MINUTES)
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="debug helper: only the first N traded pairs")
    args = parser.parse_args(argv)

    result, config_path = load_run_result(args.run_dir)
    if args.max_pairs:
        seen: list[str] = []
        kept = []
        for trade in result.trades:
            pair_id = str(getattr(trade, "pair_id", "") or "")
            if pair_id not in seen:
                if len(seen) >= args.max_pairs:
                    continue
                seen.append(pair_id)
            kept.append(trade)
        result.trades = kept

    written = export_curve_charts(
        result,
        config_path,
        args.output_dir or args.run_dir,
        initial_equity=result.metrics.get("initial_equity"),
        sample_minutes=args.sample_minutes,
    )
    print(written["dir"])
    print(f"portfolio: {written['portfolio']}")
    print(f"pairs: {len(written['pairs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
