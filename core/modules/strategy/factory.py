from core.modules.position_sizing import PairSizing
from core.modules.signals import build_signal
from core.modules.strategy.pair_trading_strategy import PairTradingStrategy


def build_strategy(setup_config, symbols):
    pair = setup_config.pair
    x_symbol = pair.get("x_symbol", pair.get("long_symbol"))
    y_symbol = pair.get("y_symbol", pair.get("short_symbol"))
    if x_symbol not in symbols or y_symbol not in symbols:
        raise ValueError("pair must contain valid x_symbol and y_symbol")

    signal = build_signal(setup_config, symbols)
    sizing = _build_sizing(setup_config)
    management = setup_config.position_management

    return PairTradingStrategy(
        signal=signal,
        sizing=sizing,
        x_exchange=symbols[x_symbol]["exchange"],
        x_symbol=x_symbol,
        y_exchange=symbols[y_symbol]["exchange"],
        y_symbol=y_symbol,
        max_add_times=int(management.get("max_add_times", 0)),
        add_interval_bars=int(management.get("add_interval_bars", 1)),
        name=setup_config.name,
    )


def _build_sizing(setup_config):
    raw = setup_config.position_sizing
    method = raw.get("method", "fixed_notional")
    return PairSizing(
        method=method,
        notional=float(raw.get("notional", 10000.0)),
        equity=float(raw.get("equity", 100000.0)),
        gross_exposure_ratio=float(raw.get("gross_exposure_ratio", 0.2)),
        max_gross_exposure_ratio=float(raw.get("max_gross_exposure_ratio", 1.0)),
        min_notional=float(raw.get("min_notional", 0.0)),
    )
