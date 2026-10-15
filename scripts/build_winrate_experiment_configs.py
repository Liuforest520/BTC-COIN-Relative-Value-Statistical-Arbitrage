"""Build a small matrix of large Walk-Forward parameter-search configs.

The experiments keep the same pair universe, dates, account settings and
Walk-Forward windows. Only the candidate grid changes, so the results can be
compared without changing the rest of the backtest definition.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


BASE_ESTIMATOR = {
    "regression_method": ["price", "log_price"],
    "model_lookback_bars": [720, 960, 1200, 1440, 1680, 1800, 2160, 2400, 2880],
    "model_update_interval_bars": [120, 180, 240, 300, 360],
    "residual_adf_max_pvalue": [1.0],
}
BASE_SIGNAL = {
    "entry_z": [2.5, 3.0, 3.5, 4.0],
    "exit_z": [0.25, 0.375, 0.5],
    "reversion_ma_lookback_bars": [60],
    "reversion_ma_short_lookback_bars": [15],
}
BASE_PROTECTION = {
    "pair_loss_stop_enabled": [True],
    "pair_loss_stop_return": [0.2, 0.4, 1.0],
}


def _grid(estimator=None, signal=None, protection=None):
    return {
        "estimator": copy.deepcopy(estimator or BASE_ESTIMATOR),
        "signal": copy.deepcopy(signal or BASE_SIGNAL),
        "protection": copy.deepcopy(protection or BASE_PROTECTION),
    }


def experiment_grids():
    grids = {}

    grids["01_baseline"] = _grid()

    # Close nearer to the mean. This is intended to improve hit rate, with
    # the explicit trade-off that average win size may become smaller.
    grids["02_early_exit"] = _grid(
        signal={**BASE_SIGNAL, "exit_z": [0.5, 0.75, 1.0]},
    )

    # Require a more extreme deviation before opening a position.
    grids["03_strict_entry"] = _grid(
        signal={**BASE_SIGNAL, "entry_z": [3.0, 3.5, 4.0, 4.5]},
    )

    grids["04_tight_stop"] = _grid(
        protection={**BASE_PROTECTION, "pair_loss_stop_return": [0.05, 0.1, 0.2, 0.3]},
    )

    grids["05_balanced_stop"] = _grid(
        protection={**BASE_PROTECTION, "pair_loss_stop_return": [0.1, 0.2, 0.3, 0.4]},
    )

    # More responsive TLS windows, while retaining enough update choices to
    # measure whether short lookbacks improve regime adaptation.
    grids["06_short_model"] = _grid(
        estimator={
            **BASE_ESTIMATOR,
            "model_lookback_bars": [480, 720, 960, 1200, 1440, 1680, 2160, 2400],
            "model_update_interval_bars": [120, 180, 240, 300, 360],
        },
    )

    # Moderate model windows, excluding the shortest and longest extremes.
    grids["07_medium_model"] = _grid(
        estimator={
            **BASE_ESTIMATOR,
            "model_lookback_bars": [960, 1200, 1440, 1680, 1800, 2160, 2400, 2880, 3360],
            "model_update_interval_bars": [120, 180, 240, 300, 360, 420],
        },
    )

    grids["08_price_only"] = _grid(
        estimator={
            **BASE_ESTIMATOR,
            "regression_method": ["price"],
            "model_lookback_bars": [480, 720, 960, 1200, 1440, 1680, 2160, 2400, 2880, 3360, 3840, 4320],
            "model_update_interval_bars": [120, 180, 240, 300, 360],
        },
    )

    grids["09_log_price_only"] = _grid(
        estimator={
            **BASE_ESTIMATOR,
            "regression_method": ["log_price"],
            "model_lookback_bars": [480, 720, 960, 1200, 1440, 1680, 2160, 2400, 2880, 3360, 3840, 4320],
            "model_update_interval_bars": [120, 180, 240, 300, 360],
        },
    )

    # A combined conservative configuration: high entry threshold, early
    # exit, and tighter protection. It is deliberately separate from the
    # single-factor experiments so its result is easy to attribute.
    grids["10_conservative_combo"] = _grid(
        estimator={
            **BASE_ESTIMATOR,
            "model_lookback_bars": [720, 960, 1200, 1440, 1680, 1800, 2160, 2400, 2880],
            "model_update_interval_bars": [120, 180, 240, 300, 360],
        },
        signal={
            **BASE_SIGNAL,
            "entry_z": [3.0, 3.5, 4.0, 4.5],
            "exit_z": [0.5, 0.75, 1.0],
        },
        protection={**BASE_PROTECTION, "pair_loss_stop_return": [0.05, 0.1, 0.2, 0.3]},
    )
    return grids


EXPERIMENT_WINDOWS = {
    "01_baseline": (8, 4, 4),
    "02_early_exit": (6, 2, 2),
    "03_strict_entry": (12, 2, 2),
    "04_tight_stop": (6, 4, 4),
    "05_balanced_stop": (12, 4, 4),
    "06_short_model": (8, 2, 2),
    "07_medium_model": (8, 6, 6),
    "08_price_only": (6, 4, 4),
    "09_log_price_only": (12, 6, 6),
    "10_conservative_combo": (9, 3, 3),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default="config/config_walk_forward_tradfi_full.yaml",
        help="Base config whose pair universe and account settings are preserved.",
    )
    parser.add_argument(
        "--output-dir",
        default="config/winrate_experiments",
    )
    args = parser.parse_args()

    base_path = Path(args.base)
    output_dir = Path(args.output_dir)
    raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, grid in experiment_grids().items():
        config = copy.deepcopy(raw)
        walk_forward = config.setdefault("walk_forward", {})
        walk_forward["parameter_grid"] = grid
        train_months, validation_months, step_months = EXPERIMENT_WINDOWS[name]
        walk_forward["train_months"] = train_months
        walk_forward["validation_months"] = validation_months
        walk_forward["step_months"] = step_months
        output_path = output_dir / f"config_walk_forward_{name}.yaml"
        output_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        print(output_path)


if __name__ == "__main__":
    main()
