from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class SetupConfig:
    name: str
    pair: dict
    signal_name: str
    signal: dict
    sizing_name: str
    position_sizing: dict
    position_management_name: str
    position_management: dict
    hedge_method: str


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


def load_config(path: str | Path = "config/config.yaml") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    setup_name = raw["active_setup"]
    setup = _build_setup_config(raw, setup_name)

    return Config(
        symbols=raw["data"]["symbols"],
        benchmarks=raw["data"].get("benchmarks", {}),
        initial_cash=float(raw["backtest"]["initial_cash"]),
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
    signal_name = setup["signal"]
    sizing_name = setup["position_sizing"]
    management_name = setup.get("position_management", "default")
    signal = _lookup(raw, "signals", signal_name)
    sizing = _lookup(raw, "position_sizing", sizing_name)
    management = _lookup(raw, "position_management", management_name)
    pair = setup.get("pair", raw.get("pair", {}))

    return SetupConfig(
        name=setup_name,
        pair=pair,
        signal_name=signal_name,
        signal=dict(signal),
        sizing_name=sizing_name,
        position_sizing=dict(sizing),
        position_management_name=management_name,
        position_management=dict(management),
        hedge_method=sizing["method"],
    )


def _lookup(raw: dict, section: str, name: str) -> dict:
    values = raw.get(section, {})
    if name not in values:
        available = ", ".join(str(item) for item in values)
        raise ValueError(f"{section}.{name!r} not found, available: {available}")
    return values[name]


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)
