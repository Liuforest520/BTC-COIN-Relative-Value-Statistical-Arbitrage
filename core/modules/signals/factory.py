from core.modules.signals.cointegration import CointegrationZScoreSignal
from core.modules.signals.funding_filter import FundingFilterZScoreSignal
from core.modules.signals.momentum_filter import MomentumFilterZScoreSignal
from core.modules.signals.ratio import RatioZScoreSignal


def build_signal(setup_config, symbols):
    pair = setup_config.pair
    x_symbol = pair.get("x_symbol", pair.get("long_symbol"))
    y_symbol = pair.get("y_symbol", pair.get("short_symbol"))
    if x_symbol not in symbols or y_symbol not in symbols:
        raise ValueError("pair must contain valid x_symbol and y_symbol")

    signal_config = setup_config.signal
    signal_type = signal_config["type"]
    common_kwargs = {
        "x_exchange": symbols[x_symbol]["exchange"],
        "x_symbol": x_symbol,
        "y_exchange": symbols[y_symbol]["exchange"],
        "y_symbol": y_symbol,
        "warmup_bars": int(signal_config["warmup_bars"]),
        "zscore_lookback_bars": int(signal_config.get("zscore_lookback_bars", signal_config["warmup_bars"])),
        "entry_z": float(signal_config.get("entry_z", 2.0)),
        "exit_z": float(signal_config.get("exit_z", 0.5)),
    }
    model_kwargs = {
        "model_lookback_bars": int(signal_config.get("model_lookback_bars", signal_config["warmup_bars"])),
        "model_update_interval_bars": int(signal_config.get("model_update_interval_bars", 1)),
        "regression_method": signal_config.get("regression_method", "log_price"),
    }

    if signal_type == "cointegration_zscore":
        return CointegrationZScoreSignal(**common_kwargs, **model_kwargs)
    if signal_type == "ratio_zscore":
        return RatioZScoreSignal(**common_kwargs)
    if signal_type == "momentum_filter_zscore":
        return MomentumFilterZScoreSignal(
            **common_kwargs,
            **model_kwargs,
            momentum_lookback_bars=int(signal_config.get("momentum_lookback_bars", 30)),
            max_abs_momentum=float(signal_config.get("max_abs_momentum", 0.5)),
        )
    if signal_type == "funding_filter_zscore":
        return FundingFilterZScoreSignal(
            **common_kwargs,
            **model_kwargs,
            funding_spread_threshold=float(signal_config.get("funding_spread_threshold", 0.0001)),
        )

    raise ValueError(f"unsupported signal type: {signal_type}")
