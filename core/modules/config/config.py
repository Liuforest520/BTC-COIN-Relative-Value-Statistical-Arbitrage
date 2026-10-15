from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
import re
from typing import Any

import yaml


SUPPORTED_MARKETS = ("crypto", "tradfi")
COMBINED_MARKET = "combined"


@dataclass
class WalkForwardConfig:
    """Configuration for natural-month rolling walk-forward runs."""

    enabled: bool = False
    train_months: int = 2
    validation_months: int = 1
    step_months: int = 1
    start_date: str | None = None
    end_date: str | None = None
    warmup_enabled: bool = True
    warmup_max_bars: int | None = None
    objective: str = "stable_score"
    min_trades: int = 20
    max_drawdown: float | None = None
    subperiod_count: int = 3
    min_active_subperiods: int = 2
    exclude_top_fraction: float = 0.02
    min_valid_neighbors: int = 2
    max_own_neighbor_gap: float | None = 0.15
    parameter_grid: dict = field(default_factory=dict)
    pair_ids: list[str] = field(default_factory=list)
    optimization_scope: str = "pair"
    portfolio_min_active_pairs: int = 1
    portfolio_require_all_pairs: bool = False
    portfolio_candidate_batch_size: int = 16
    process_workers: int = 1

    def __post_init__(self):
        for name in ("train_months", "validation_months", "step_months"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"walk_forward.{name} must be positive")
            setattr(self, name, value)
        self.min_trades = max(0, int(self.min_trades))
        self.subperiod_count = max(1, int(self.subperiod_count))
        self.min_active_subperiods = max(0, int(self.min_active_subperiods))
        self.exclude_top_fraction = float(self.exclude_top_fraction or 0.0)
        if not 0.0 <= self.exclude_top_fraction < 1.0:
            raise ValueError("walk_forward.selection.exclude_top_fraction must be in [0, 1)")
        self.min_valid_neighbors = max(0, int(self.min_valid_neighbors))
        if self.max_own_neighbor_gap is not None:
            self.max_own_neighbor_gap = float(self.max_own_neighbor_gap)
            if self.max_own_neighbor_gap < 0.0:
                raise ValueError("walk_forward.selection.max_own_neighbor_gap must be nonnegative or null")
        self.optimization_scope = str(self.optimization_scope or "pair").strip().lower()
        if self.optimization_scope not in {"pair", "portfolio"}:
            raise ValueError("walk_forward.optimization_scope must be 'pair' or 'portfolio'")
        self.portfolio_min_active_pairs = max(1, int(self.portfolio_min_active_pairs))
        self.portfolio_require_all_pairs = _as_bool(self.portfolio_require_all_pairs)
        self.portfolio_candidate_batch_size = max(1, int(self.portfolio_candidate_batch_size))
        self.process_workers = max(1, int(self.process_workers))
        if self.warmup_max_bars is not None:
            self.warmup_max_bars = max(1, int(self.warmup_max_bars))


@dataclass
class SetupConfig:
    name: str
    strategy_type: str
    pairs: list[dict]
    pipeline: dict
    rebalance: dict = None
    market: str = "legacy"
    execution: dict = field(default_factory=dict)
    # New multi-market schema: every non-empty market keeps its own pipeline
    # while all pairs still share one strategy/account.
    market_pipelines: dict[str, dict] = field(default_factory=dict)


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
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    backtest_start_time: Any = None
    backtest_start_ts: int | None = None
    backtest_end_time: Any = None
    backtest_end_ts: int | None = None
    data_start_ts: int | None = None
    measurement_start_ts: int | None = None
    backtest_end_exclusive: bool = False
    active_market: str = "legacy"
    active_markets: tuple[str, ...] = ()


def load_config(path: str | Path = "config/config.yaml") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    setup_name = raw["active_setup"]
    setup, symbols, benchmarks, active_markets = _build_combined_setup(raw, setup_name)
    backtest = raw.get("backtest", {})
    start_time = backtest.get("start_time", backtest.get("start_ts"))
    end_time = backtest.get("end_time", backtest.get("end_ts"))
    start_ts = _parse_optional_timestamp(start_time)
    end_ts = _parse_optional_timestamp(end_time)
    walk_forward = _walk_forward_config(raw.get("walk_forward") or {})
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
        active_market=(active_markets[0] if len(active_markets) == 1 else "combined"),
        active_markets=tuple(active_markets),
        strategy=setup,
        fee_rate=float(raw["cost"]["fee_rate"]),
        slippage_bps=float(raw["cost"]["slippage_bps"]),
        funding_enabled=_as_bool(raw["cost"].get("funding_enabled", True)),
        risk=dict(raw.get("risk", {})),
        walk_forward=walk_forward,
        data_start_ts=start_ts,
        measurement_start_ts=None,
    )


def _walk_forward_config(raw: dict) -> WalkForwardConfig:
    warmup = raw.get("warmup") or {}
    selection = raw.get("selection") or {}
    portfolio = raw.get("portfolio") or {}
    return WalkForwardConfig(
        enabled=_as_bool(raw.get("enabled", False)),
        train_months=raw.get("train_months", 2),
        validation_months=raw.get("validation_months", 1),
        step_months=raw.get("step_months", 1),
        start_date=raw.get("start_date"),
        end_date=raw.get("end_date"),
        warmup_enabled=_as_bool(warmup.get("enabled", True)),
        warmup_max_bars=warmup.get("max_bars"),
        objective=selection.get("objective", "stable_score"),
        min_trades=selection.get("min_trades", 20),
        max_drawdown=selection.get("max_drawdown"),
        subperiod_count=selection.get("subperiod_count", 3),
        min_active_subperiods=selection.get("min_active_subperiods", 2),
        exclude_top_fraction=selection.get("exclude_top_fraction", 0.02),
        min_valid_neighbors=selection.get("min_valid_neighbors", 2),
        max_own_neighbor_gap=selection.get("max_own_neighbor_gap", 0.15),
        parameter_grid=dict(raw.get("parameter_grid") or {}),
        pair_ids=[str(value) for value in (raw.get("pair_ids") or [])],
        optimization_scope=raw.get("optimization_scope", "pair"),
        portfolio_min_active_pairs=portfolio.get("min_active_pairs", 1),
        portfolio_require_all_pairs=_as_bool(portfolio.get("require_all_pairs", False)),
        portfolio_candidate_batch_size=portfolio.get("candidate_batch_size", 16),
        process_workers=raw.get("process_workers", raw.get("workers", 1)),
    )


def _active_market(raw: dict) -> str:
    value = raw.get("active_market")
    if value is None or str(value).strip() == "":
        return "legacy"
    market = str(value).strip().lower()
    if market not in (*SUPPORTED_MARKETS, COMBINED_MARKET):
        raise ValueError(
            f"active_market must be one of {(*SUPPORTED_MARKETS, COMBINED_MARKET)}, got {value!r}"
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
    if market == COMBINED_MARKET:
        market = None
    setups = raw.get("setups") or {}
    if setup_name not in setups:
        available = ", ".join(str(name) for name in setups)
        raise ValueError(f"active_setup {setup_name!r} not found, available: {available}")
    setup = setups[setup_name] or {}
    # Pair-driven schema: the Pair list is authoritative for what runs; the
    # market label only selects the parameter profile for each Pair.
    if setup.get("pipelines") is not None:
        pairs = list(setup.get("pairs") or [])
        pipelines = setup.get("pipelines") or {}
        if market in (None, COMBINED_MARKET):
            selected_pairs = pairs
            first_market = next(
                (str(item.get("market", "")).strip().lower() for item in pairs if item.get("enabled", True)),
                None,
            )
            first_pipeline = pipelines.get(first_market, {}) if first_market else {}
            return {
                "name": setup_name,
                "strategy": setup.get("strategy", setup.get("strategy_type", "auto")),
                "pairs": selected_pairs,
                "pipeline": first_pipeline,
                "pipelines": pipelines,
                "rebalance": setup.get("rebalance", raw.get("rebalance", {})) or {},
                "market": "combined",
            }
        market = str(market).strip().lower()
        if market not in SUPPORTED_MARKETS:
            raise ValueError(f"market must be one of {SUPPORTED_MARKETS}, got {market!r}")
        return {
            "name": setup_name,
            "strategy": setup.get("strategy", setup.get("strategy_type", "auto")),
            "pairs": [item for item in pairs if str(item.get("market", "")).strip().lower() == market],
            "pipeline": pipelines.get(market),
            "pipelines": {market: pipelines.get(market)} if market in pipelines else {},
            "rebalance": setup.get("rebalance", raw.get("rebalance", {})) or {},
            "market": market,
        }
    markets = setup.get("markets")
    requested_market = _active_market(raw)
    if market is None and markets is not None and requested_market not in SUPPORTED_MARKETS:
        combined = select_market_views(raw, setup_name)
        pairs = []
        for market_name, view in combined.items():
            for item in view.get("pairs") or []:
                tagged = dict(item)
                tagged["market"] = market_name
                pairs.append(tagged)
        first = next(iter(combined.values()), {})
        return {
            "name": setup_name,
            "strategy": setup.get("strategy", setup.get("strategy_type", "auto")),
            "pairs": pairs,
            "pipeline": first.get("pipeline"),
            "rebalance": first.get("rebalance", {}),
            "market": "combined",
        }
    active_market = market or requested_market
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


def select_market_views(raw: dict, setup_name: str | None = None) -> dict[str, dict]:
    """Return all configured market views that contain at least one Pair."""
    setup_name = setup_name or raw.get("active_setup")
    setups = raw.get("setups") or {}
    setup = setups.get(setup_name) or {}
    if setup.get("pipelines") is not None:
        pairs = setup.get("pairs") or []
        result = {}
        for market in SUPPORTED_MARKETS:
            if any(
                item.get("enabled", True)
                and str(item.get("market", "")).strip().lower() == market
                for item in pairs
            ):
                result[market] = select_market_view(raw, setup_name, market)
        return result
    markets = setup.get("markets")
    if markets is None:
        return {"legacy": select_market_view(raw, setup_name, "legacy")}
    result = {}
    for market in SUPPORTED_MARKETS:
        view = markets.get(market)
        if not isinstance(view, dict) or not view.get("pairs"):
            continue
        result[market] = select_market_view(raw, setup_name, market)
    return result


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
    if market == COMBINED_MARKET:
        market = None
    if market is None and (raw.get("data") or {}).get("markets") is not None:
        return _combined_market_data(raw)
    active_market = market or _active_market(raw)
    symbols, benchmarks = _market_data(raw, active_market)
    return {"symbols": symbols, "benchmarks": benchmarks, "market": active_market}


def _combined_market_data(raw: dict) -> dict:
    """Merge every market block into one symbol -> data mapping.

    Symbol data is global, so a symbol may only be declared once; the engine keeps
    exactly one price series per symbol (positions, margin and PnL are all keyed by
    symbol).  The flat ``data.symbols`` schema of a pair-driven setup therefore does
    not need this function at all - it is only reached through ``data.markets``.
    """
    data = raw.get("data") or {}
    markets = data.get("markets") or {}
    symbols: dict = {}
    benchmarks: dict = dict(data.get("benchmarks") or {})
    for market in SUPPORTED_MARKETS:
        market_data = markets.get(market) or {}
        raw_symbols = market_data.get("symbols")
        if raw_symbols is not None and not isinstance(raw_symbols, dict):
            raise ValueError(f"data.markets.{market}.symbols must be a mapping")
        overlap = set(symbols).intersection(raw_symbols or {})
        if overlap:
            raise ValueError(f"symbol(s) configured in multiple markets: {sorted(overlap)}")
        symbols.update(raw_symbols or {})
        benchmarks.update(market_data.get("benchmarks") or {})
    return {"symbols": symbols, "benchmarks": benchmarks, "market": "combined"}


def _known_market_symbols(raw: dict) -> dict:
    """Every symbol declared anywhere in the config, keyed by symbol name.

    Used to resolve Pair legs: with the multi-market schema the declaration is
    global, while the market block that owns the Pair only selects which
    pipeline parameters it runs with.
    """
    data = raw.get("data") or {}
    if data.get("markets") is not None:
        return _combined_market_data(raw)["symbols"]
    return dict(data.get("symbols") or {})


def _build_combined_setup(raw: dict, setup_name: str):
    """Build one shared-account setup.

    New pair-driven configs keep one flat symbol table and one Pair list.  The
    Pair's ``market`` selects only its model-parameter profile.  The older
    ``active_market``/``setup.markets`` schema remains a compatibility path.
    """
    setups = raw["setups"]
    if setup_name not in setups:
        available = ", ".join(str(name) for name in setups)
        raise ValueError(f"active_setup {setup_name!r} not found, available: {available}")
    setup_raw = setups[setup_name] or {}
    if setup_raw.get("pipelines") is not None:
        return _build_pair_driven_setup(raw, setup_name, setup_raw)
    if setup_raw.get("markets") is None:
        active_market = _active_market(raw)
        setup = _build_setup_config(raw, setup_name, active_market)
        symbols, benchmarks = _market_data(raw, active_market)
        _validate_pair_symbols(setup.pairs, symbols, setup_name, active_market)
        _validate_market_data_paths(symbols, active_market, setup.pairs)
        return setup, symbols, benchmarks, ([active_market] if active_market != "legacy" else ["legacy"])

    requested_market = _active_market(raw)
    if requested_market in SUPPORTED_MARKETS:
        view = (setup_raw.get("markets") or {}).get(requested_market) or {}
        market_pairs = list(view.get("pairs") or [])
        if not market_pairs:
            raise ValueError(f"setup {setup_name!r} must define pairs for the active market")
        pipeline = view.get("pipeline")
        if not pipeline:
            raise ValueError(f"setup {setup_name!r} market {requested_market!r} must define pipeline")
        market_symbols = ((raw.get("data") or {}).get("markets") or {}).get(requested_market, {}).get("symbols") or {}
        symbols = _known_market_symbols(raw)
        _validate_pair_symbols(market_pairs, symbols, setup_name, requested_market)
        _validate_market_data_paths(market_symbols, requested_market, market_pairs)
        tagged_pairs = [dict(pair, market=requested_market) for pair in market_pairs]
        setup = SetupConfig(
            name=setup_name,
            strategy_type=view.get("strategy", setup_raw.get("strategy", setup_raw.get("strategy_type", "auto"))),
            pairs=tagged_pairs,
            pipeline=dict(pipeline),
            rebalance=dict(view.get("rebalance", {})),
            market=requested_market,
            market_pipelines={requested_market: dict(view)},
        )
        market_block = ((raw.get("data") or {}).get("markets") or {}).get(requested_market) or {}
        benchmarks = market_block.get("benchmarks") or (raw.get("data") or {}).get("benchmarks") or {}
        return setup, symbols, dict(benchmarks), [requested_market]

    combined_data = _combined_market_data(raw)
    pairs: list[dict] = []
    market_pipelines: dict[str, dict] = {}
    active_markets: list[str] = []
    timeframes = set()
    for market in SUPPORTED_MARKETS:
        view = (setup_raw.get("markets") or {}).get(market) or {}
        market_pairs = list(view.get("pairs") or [])
        if not market_pairs:
            continue
        pipeline = view.get("pipeline")
        if not pipeline:
            raise ValueError(f"setup {setup_name!r} market {market!r} must define pipeline")
        market_symbols = ((raw.get("data") or {}).get("markets") or {}).get(market, {}).get("symbols") or {}
        # Pair legs are resolved against every market block: a tradfi Pair may use a
        # symbol whose file is declared under crypto (and the other way round).
        _validate_pair_symbols(market_pairs, combined_data["symbols"], setup_name, market)
        _validate_market_data_paths(market_symbols, market, market_pairs)
        estimator = pipeline.get("estimator", {}) or {}
        timeframes.add(str(estimator.get("model_timeframe", "1m")).strip().lower())
        market_pipelines[market] = dict(view)
        active_markets.append(market)
        for pair in market_pairs:
            tagged = dict(pair)
            tagged["market"] = market
            pairs.append(tagged)
    if not pairs:
        raise ValueError(f"setup {setup_name!r} must define at least one Pair in an active market")
    if len(timeframes) > 1:
        raise ValueError(
            "crypto and tradfi must use the same estimator.model_timeframe for a shared backtest; "
            f"got {sorted(timeframes)}"
        )
    first_market = active_markets[0]
    first_view = market_pipelines[first_market]
    setup = SetupConfig(
        name=setup_name,
        strategy_type=first_view.get("strategy", setup_raw.get("strategy", setup_raw.get("strategy_type", "auto"))),
        pairs=pairs,
        pipeline=dict(first_view.get("pipeline") or {}),
        rebalance=dict(first_view.get("rebalance", {})),
        market="combined",
        market_pipelines=market_pipelines,
    )
    return setup, combined_data["symbols"], combined_data["benchmarks"], active_markets


def _build_pair_driven_setup(raw: dict, setup_name: str, setup_raw: dict):
    """Normalize the preferred flat-data, Pair-driven configuration schema."""
    if (raw.get("data") or {}).get("markets") is not None:
        raise ValueError(
            "pair-driven setup requires data.symbols; remove data.markets and keep one path per symbol"
        )
    if raw.get("active_market") not in (None, ""):
        raise ValueError(
            "pair-driven setup does not use active_market; put market on every enabled Pair"
        )

    data = raw.get("data") or {}
    symbols_raw = data.get("symbols")
    if not isinstance(symbols_raw, dict):
        raise ValueError("pair-driven setup requires data.symbols to be a mapping")
    symbols_all = {str(symbol): dict(info or {}) for symbol, info in symbols_raw.items()}

    pairs = [dict(item or {}) for item in (setup_raw.get("pairs") or [])]
    enabled_pairs = [item for item in pairs if item.get("enabled", True)]
    if not enabled_pairs:
        raise ValueError(f"setup {setup_name!r} must define at least one enabled Pair")
    pipelines_raw = setup_raw.get("pipelines") or {}
    if not isinstance(pipelines_raw, dict):
        raise ValueError(f"setup {setup_name!r}.pipelines must be a mapping")

    active_markets: list[str] = []
    normalized_pipelines: dict[str, dict] = {}
    used_symbols: set[str] = set()
    timeframes: set[str] = set()
    for item in enabled_pairs:
        pair_id = str(item.get("pair_id", "?"))
        market = str(item.get("market", "")).strip().lower()
        if market not in SUPPORTED_MARKETS:
            raise ValueError(
                f"Pair {pair_id!r} must define market as one of {SUPPORTED_MARKETS}"
            )
        if market not in pipelines_raw or not isinstance(pipelines_raw.get(market), dict):
            raise ValueError(
                f"setup {setup_name!r} is missing pipelines.{market} for Pair {pair_id!r}"
            )
        try:
            target_capital = float(item.get("target_capital"))
        except (TypeError, ValueError):
            target_capital = 0.0
        if target_capital <= 0:
            raise ValueError(
                f"Pair {pair_id!r} must define a positive target_capital"
            )
        if market not in active_markets:
            active_markets.append(market)
            pipeline = dict(pipelines_raw[market])
            normalized_pipelines[market] = pipeline
            estimator = pipeline.get("estimator", {}) or {}
            timeframes.add(str(estimator.get("model_timeframe", "1m")).strip().lower())
        x_symbol = item.get("x_symbol", item.get("long_symbol"))
        y_symbol = item.get("y_symbol", item.get("short_symbol"))
        for symbol in (x_symbol, y_symbol):
            if symbol not in symbols_all:
                raise ValueError(
                    f"Pair {pair_id!r} references symbol {symbol!r} missing from data.symbols"
                )
            if not symbols_all[symbol].get("path"):
                raise ValueError(f"data.symbols.{symbol}.path is required")
            used_symbols.add(symbol)

    if len(timeframes) > 1:
        raise ValueError(
            "all Pair pipelines in one backtest must use the same estimator.model_timeframe; "
            f"got {sorted(timeframes)}"
        )

    # Rebalance and execution are account-level policies in the pair-driven
    # schema.  Normalize them into every market view so downstream code keeps
    # its existing PairPipeline interface without selecting one market's copy.
    global_rebalance = dict(
        setup_raw.get("rebalance", raw.get("rebalance", {})) or {}
    )
    common_execution = setup_raw.get("execution", raw.get("execution"))
    if common_execution is None:
        execution_values = [
            dict((normalized_pipelines[m].get("execution") or {}))
            for m in active_markets
        ]
        common_execution = execution_values[0] if execution_values else {}
        if any(value != common_execution for value in execution_values[1:]):
            raise ValueError(
                "pair-driven setup requires one shared execution configuration; "
                "put execution at setup.execution"
            )
    common_execution = dict(common_execution or {})
    for market in active_markets:
        pipeline = dict(normalized_pipelines[market])
        pipeline.pop("rebalance", None)
        pipeline["execution"] = dict(common_execution)
        normalized_pipelines[market] = pipeline

    tagged_pairs = []
    for item in pairs:
        tagged = dict(item)
        if item.get("enabled", True):
            tagged["market"] = str(item["market"]).strip().lower()
        tagged_pairs.append(tagged)

    market_pipelines = {
        market: {"pipeline": normalized_pipelines[market], "rebalance": global_rebalance}
        for market in active_markets
    }
    first_market = active_markets[0]
    setup = SetupConfig(
        name=setup_name,
        strategy_type=setup_raw.get("strategy", setup_raw.get("strategy_type", "auto")),
        pairs=tagged_pairs,
        pipeline=dict(normalized_pipelines[first_market]),
        rebalance=global_rebalance,
        market="combined" if len(active_markets) > 1 else first_market,
        execution=common_execution,
        market_pipelines=market_pipelines,
    )
    return (
        setup,
        {symbol: symbols_all[symbol] for symbol in sorted(used_symbols)},
        dict(data.get("benchmarks") or {}),
        active_markets,
    )


def _validate_pair_symbols(pairs: list[dict], symbols: dict, setup_name: str, market: str) -> None:
    """Check that every enabled Pair leg is declared somewhere in data.symbols.

    ``symbols`` is the merged symbol table of the whole config (all market
    blocks), so a Pair may mix legs that are declared under different markets.
    """
    known_symbols = set(symbols)
    missing: list[tuple[str, str]] = []
    for item in pairs:
        if not item.get("enabled", True):
            continue
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
            f"setup {setup_name!r} market {market!r} contains Pair symbols that are not declared "
            f"under data.markets.*.symbols (any market block counts): {details}"
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
        if symbol not in symbols:
            # Declared by another market block (e.g. a crypto kline file used as
            # one leg of a tradfi Pair).  That block owns the file, and the merged
            # -extended.csv rule does not apply to it.
            continue
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
