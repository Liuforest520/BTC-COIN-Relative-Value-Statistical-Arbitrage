from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import re
from typing import Any

import yaml


@dataclass
class SetupConfig:
    name: str
    strategy_type: str
    pairs: list[dict]
    pipeline: dict
    rebalance: dict = None


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


def load_config(path: str | Path = "config/config.yaml") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    setup_name = raw["active_setup"]
    setup = _build_setup_config(raw, setup_name)
    backtest = raw.get("backtest", {})
    start_time = backtest.get("start_time", backtest.get("start_ts"))
    end_time = backtest.get("end_time", backtest.get("end_ts"))
    start_ts = _parse_optional_timestamp(start_time)
    end_ts = _parse_optional_timestamp(end_time)
    if start_ts is not None and end_ts is not None and end_ts < start_ts:
        raise ValueError("backtest end_time must be greater than or equal to start_time")
    return Config(
        symbols=raw["data"]["symbols"],
        benchmarks=raw["data"].get("benchmarks", {}),
        initial_cash=float(backtest["initial_cash"]),
        backtest_start_time=start_time,
        backtest_start_ts=start_ts,
        backtest_end_time=end_time,
        backtest_end_ts=end_ts,
        active_setup=setup_name,
        strategy=setup,
        fee_rate=float(raw["cost"]["fee_rate"]),
        slippage_bps=float(raw["cost"]["slippage_bps"]),
        funding_enabled=_as_bool(raw["cost"].get("funding_enabled", True)),
        risk=dict(raw.get("risk", {})),
    )


def _build_setup_config(raw: dict, setup_name: str) -> SetupConfig:
    setups = raw["setups"]
    if setup_name not in setups:
        available = ", ".join(str(name) for name in setups)
        raise ValueError(f"active_setup {setup_name!r} not found, available: {available}")

    setup = setups[setup_name]
    pipeline = setup.get("pipeline")
    pairs = setup.get("pairs")

    if not pairs:
        raise ValueError(f"setup {setup_name!r} must define pairs for the multi-pair framework")
    if not pipeline:
        raise ValueError(f"setup {setup_name!r} must define pipeline for the multi-pair framework")

    return SetupConfig(
        name=setup_name,
        strategy_type=setup.get("strategy", setup.get("strategy_type", "auto")),
        pairs=list(pairs),
        pipeline=dict(pipeline),
        rebalance=dict(setup.get("rebalance", {})),
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
