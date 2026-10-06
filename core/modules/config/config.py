from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import re
from typing import Any

import yaml


SUPPORTED_MARKETS = ("crypto", "tradfi")


@dataclass
class SetupConfig:
    name: str
    strategy_type: str
    pairs: list[dict]
    pipeline: dict
    rebalance: dict = None
    market: str = "legacy"


@dataclass
class Config:
    symbols: dict
    benchmarks: dict
    initial_cash: float
    active_setup: str
    strategy: SetupConfig
    fee_rate: float
    slippage_bps: float
    funding_enabled: bool
    risk: dict
    backtest_start_time: Any = None
    backtest_start_ts: int | None = None
    backtest_end_time: Any = None
    backtest_end_ts: int | None = None
    active_market: str = "legacy"


def load_config(path: str | Path = "config/config.yaml") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    setup_name = raw["active_setup"]
    active_market = _active_market(raw)
    setup = _build_setup_config(raw, setup_name, active_market)
    symbols, benchmarks = _market_data(raw, active_market)
    _validate_pair_symbols(setup.pairs, symbols, setup_name, active_market)
    _validate_market_data_paths(symbols, active_market, setup.pairs)
    backtest = raw.get("backtest", {})
    start_time = backtest.get("start_time", backtest.get("start_ts"))
    end_time = backtest.get("end_time", backtest.get("end_ts"))
    start_ts = _parse_optional_timestamp(start_time)
    end_ts = _parse_optional_timestamp(end_time)
    if start_ts is not None and end_ts is not None and end_ts < start_ts:
        raise ValueError("backtest end_time must be greater than or equal to start_time")
    return Config(
        symbols=symbols,
        benchmarks=benchmarks,
        initial_cash=float(backtest["initial_cash"]),
        backtest_start_time=start_time,
        backtest_start_ts=start_ts,
        backtest_end_time=end_time,
        backtest_end_ts=end_ts,
        active_setup=setup_name,
        active_market=active_market,
        strategy=setup,
        fee_rate=float(raw["cost"]["fee_rate"]),
        slippage_bps=float(raw["cost"]["slippage_bps"]),
        funding_enabled=_as_bool(raw["cost"].get("funding_enabled", True)),
        risk=dict(raw.get("risk", {})),
    )


def _active_market(raw: dict) -> str:
    value = raw.get("active_market")
    if value is None or str(value).strip() == "":
        return "legacy"
    market = str(value).strip().lower()
    if market not in SUPPORTED_MARKETS:
        raise ValueError(
            f"active_market must be one of {SUPPORTED_MARKETS}, got {value!r}"
        )
    return market


def select_market_view(raw: dict, setup_name: str | None = None, market: str | None = None) -> dict:
    """Return the effective setup view for reports and config utilities.

    New configs keep market-specific pairs/pipeline under ``setup.markets``.
    Legacy configs continue to expose pairs/pipeline directly on the setup.
    The returned mapping always has ``pairs``, ``pipeline``, ``rebalance`` and
    ``market`` keys, so callers do not need to duplicate compatibility logic.
    """
    setup_name = setup_name or raw.get("active_setup")
    setups = raw.get("setups") or {}
    if setup_name not in setups:
        available = ", ".join(str(name) for name in setups)
        raise ValueError(f"active_setup {setup_name!r} not found, available: {available}")
    setup = setups[setup_name] or {}
    active_market = market or _active_market(raw)
    markets = setup.get("markets")
    if markets is None:
        if active_market != "legacy":
            raise ValueError(
                f"setup {setup_name!r} has no markets block for active_market={active_market!r}"
            )
        return {
            "name": setup_name,
            "strategy": setup.get("strategy", setup.get("strategy_type", "auto")),
            "pairs": setup.get("pairs"),
            "pipeline": setup.get("pipeline"),
            "rebalance": setup.get("rebalance", {}),
            "market": "legacy",
        }

    if active_market not in SUPPORTED_MARKETS:
        raise ValueError(f"setup {setup_name!r} requires active_market in {SUPPORTED_MARKETS}")
    market_view = markets.get(active_market)
    if market_view is None:
        raise ValueError(
            f"setup {setup_name!r} does not define market {active_market!r}"
        )
    market_view = market_view or {}
    return {
        "name": setup_name,
        "strategy": market_view.get("strategy", setup.get("strategy", setup.get("strategy_type", "auto"))),
        "pairs": market_view.get("pairs"),
        "pipeline": market_view.get("pipeline"),
        "rebalance": market_view.get("rebalance", {}),
        "market": active_market,
    }


def _market_data(raw: dict, active_market: str) -> tuple[dict, dict]:
    data = raw.get("data") or {}
    markets = data.get("markets")
    if markets is None:
        if active_market != "legacy":
            raise ValueError("data.markets is required when active_market is set")
        return dict(data.get("symbols") or {}), dict(data.get("benchmarks") or {})
    if active_market not in SUPPORTED_MARKETS:
        raise ValueError(f"data.markets requires active_market in {SUPPORTED_MARKETS}")
    market_data = markets.get(active_market)
    if market_data is None:
        raise ValueError(f"data.markets does not define market {active_market!r}")
    market_data = market_data or {}
    symbols = market_data.get("symbols")
    if not isinstance(symbols, dict):
        raise ValueError(f"data.markets.{active_market}.symbols must be a mapping")
    benchmarks = market_data.get("benchmarks", data.get("benchmarks", {})) or {}
    return dict(symbols), dict(benchmarks)


def select_market_data(raw: dict, market: str | None = None) -> dict:
    """Return the effective symbol and benchmark maps for a raw YAML mapping."""
    active_market = market or _active_market(raw)
    symbols, benchmarks = _market_data(raw, active_market)
    return {"symbols": symbols, "benchmarks": benchmarks, "market": active_market}


def _validate_pair_symbols(pairs: list[dict], symbols: dict, setup_name: str, market: str) -> None:
    known_symbols = set(symbols)
    missing: list[tuple[str, str]] = []
    for item in pairs:
        pair_id = str(item.get("pair_id", "?"))
        x_symbol = item.get("x_symbol", item.get("long_symbol"))
        y_symbol = item.get("y_symbol", item.get("short_symbol"))
        if x_symbol not in known_symbols:
            missing.append((pair_id, str(x_symbol)))
        if y_symbol not in known_symbols:
            missing.append((pair_id, str(y_symbol)))
    if missing:
        details = ", ".join(f"{pair_id}:{symbol}" for pair_id, symbol in missing)
        raise ValueError(
            f"setup {setup_name!r} market {market!r} contains Pair symbols missing "
            f"from its market data.symbols: {details}"
        )


def _validate_market_data_paths(symbols: dict, market: str, pairs: list[dict] | None = None) -> None:
    if market != "tradfi":
        return
    used_symbols = set()
    for item in pairs or []:
        if not item.get("enabled", True):
            continue
        used_symbols.update(
            symbol for symbol in (
                item.get("x_symbol", item.get("long_symbol")),
                item.get("y_symbol", item.get("short_symbol")),
            ) if symbol
        )
    if not used_symbols:
        used_symbols = set(symbols)
    invalid = []
    missing = []
    for symbol in sorted(used_symbols):
        info = symbols.get(symbol) or {}
        path = str((info or {}).get("path", "")).replace("\\", "/").lower()
        if "crypto_extended" not in path and not path.endswith("-extended.csv"):
            display_path = info.get("path") if isinstance(info, dict) else None
            invalid.append(f"{symbol}={display_path}")
            continue
        raw_path = (info or {}).get("path")
        if not raw_path:
            missing.append(f"{symbol}=<empty path>")
            continue
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = Path(__file__).resolve().parents[3] / candidate
        if not candidate.exists() or not candidate.is_file():
            missing.append(f"{symbol}={raw_path}")
    if invalid or missing:
        details = []
        if invalid:
            details.append("invalid merged-file naming: " + ", ".join(invalid))
        if missing:
            details.append("missing files: " + ", ".join(missing))
        raise ValueError(
            "tradfi market data paths must point to merged historical files "
            "under data/crypto_extended or use a -extended.csv filename; "
            + "; ".join(details)
        )


def _build_setup_config(raw: dict, setup_name: str, active_market: str = "legacy") -> SetupConfig:
    setups = raw["setups"]
    if setup_name not in setups:
        available = ", ".join(str(name) for name in setups)
        raise ValueError(f"active_setup {setup_name!r} not found, available: {available}")

    view = select_market_view(raw, setup_name, active_market)
    pipeline = view.get("pipeline")
    pairs = view.get("pairs")

    if not pairs:
        raise ValueError(f"setup {setup_name!r} must define pairs for the multi-pair framework")
    if not pipeline:
        raise ValueError(f"setup {setup_name!r} must define pipeline for the multi-pair framework")

    return SetupConfig(
        name=setup_name,
        strategy_type=view.get("strategy", "auto"),
        pairs=list(pairs),
        pipeline=dict(pipeline),
        rebalance=dict(view.get("rebalance", {})),
        market=view.get("market", active_market),
    )


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _parse_optional_timestamp(value) -> int | None:
    if value is None or value == "":
        return None

    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    if isinstance(value, date):
        dt = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    text = str(value).strip()
    if not text:
        return None

    try:
        parsed = int(float(text))
    except ValueError:
        dt = _parse_datetime_text(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    if parsed > 10_000_000_000_000_000:
        return parsed // 1000
    if parsed > 10_000_000_000:
        return parsed
    return parsed * 1000


def _parse_datetime_text(text: str) -> datetime:
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass

    match = re.fullmatch(
        r"(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T](\d{1,2}):(\d{1,2})(?::(\d{1,2}))?)?",
        text,
    )
    if not match:
        raise ValueError(f"invalid timestamp: {text!r}")

    year, month, day, hour, minute, second = match.groups()
    return datetime(
        int(year),
        int(month),
        int(day),
        int(hour or 0),
        int(minute or 0),
        int(second or 0),
    )
