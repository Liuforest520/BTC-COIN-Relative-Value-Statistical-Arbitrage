from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "tls_huber_selected_24"
    / "tls_price_L60D_U1D_E2p0_X0p5.yaml"
)
OUTPUT_DIR = PROJECT_ROOT / "config" / "tls_huber_selected_24"
ENTRY_VALUES = (2.0, 2.5, 3.0)
EXIT_VALUE = 0.5


BASE_SETUPS = (
    {
        "method": "tls",
        "data_label": "price",
        "regression_method": "price",
        "return_interval_bars": 1,
        "model_lookback_bars": 86400,
        "model_update_interval_bars": 1440,
        "position_exit_zscore_method": "standard",
    },
    {
        "method": "tls",
        "data_label": "log_price",
        "regression_method": "log_price",
        "return_interval_bars": 1,
        "model_lookback_bars": 86400,
        "model_update_interval_bars": 1440,
        "position_exit_zscore_method": "standard",
    },
    {
        "method": "tls",
        "data_label": "log_return_1m",
        "regression_method": "log_return",
        "return_interval_bars": 1,
        "model_lookback_bars": 86400,
        "model_update_interval_bars": 1440,
        "position_exit_zscore_method": "sum_zscore",
    },
    {
        "method": "tls",
        "data_label": "log_return_300m",
        "regression_method": "log_return",
        "return_interval_bars": 300,
        "model_lookback_bars": 86400,
        "model_update_interval_bars": 1440,
        "position_exit_zscore_method": "standard",
    },
    {
        "method": "huber",
        "data_label": "price",
        "regression_method": "price",
        "return_interval_bars": 1,
        "model_lookback_bars": 86400,
        "model_update_interval_bars": 1440,
        "position_exit_zscore_method": "standard",
    },
    {
        "method": "huber",
        "data_label": "log_price",
        "regression_method": "log_price",
        "return_interval_bars": 1,
        "model_lookback_bars": 86400,
        "model_update_interval_bars": 1440,
        "position_exit_zscore_method": "standard",
    },
    {
        "method": "huber",
        "data_label": "log_return_1m",
        "regression_method": "log_return",
        "return_interval_bars": 1,
        "model_lookback_bars": 86400,
        "model_update_interval_bars": 1440,
        "position_exit_zscore_method": "sum_zscore",
    },
    {
        "method": "huber",
        "data_label": "log_return_300m",
        "regression_method": "log_return",
        "return_interval_bars": 300,
        "model_lookback_bars": 86400,
        "model_update_interval_bars": 1440,
        "position_exit_zscore_method": "standard",
    },
)


def _days(bars: int) -> int:
    if bars % 1440:
        raise ValueError(f"bars cannot be represented as whole days: {bars}")
    return bars // 1440


def _z_label(value: float) -> str:
    return f"{value:.1f}".replace(".", "p")


def _run_name(setup: dict, entry_z: float) -> str:
    return (
        f"{setup['method']}_{setup['data_label']}"
        f"_L{_days(setup['model_lookback_bars'])}D"
        f"_U{_days(setup['model_update_interval_bars'])}D"
        f"_E{_z_label(entry_z)}"
        f"_X{_z_label(EXIT_VALUE)}"
    )


def main() -> None:
    with BASE_CONFIG.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    expected_names: set[str] = set()

    for setup in BASE_SETUPS:
        for entry_z in ENTRY_VALUES:
            config = deepcopy(base)
            estimator = config["setups"]["multi_pair_beta"]["pipeline"]["estimator"]
            signal = config["setups"]["multi_pair_beta"]["pipeline"]["signal"]

            estimator["method"] = setup["method"]
            estimator["regression_method"] = setup["regression_method"]
            estimator["return_interval_bars"] = setup["return_interval_bars"]
            estimator["model_lookback_bars"] = setup["model_lookback_bars"]
            estimator["model_update_interval_bars"] = setup["model_update_interval_bars"]
            if setup["method"] == "huber":
                estimator["huber_delta"] = 1.96
                estimator["huber_iterations"] = 5

            signal["entry_z"] = entry_z
            signal["exit_z"] = EXIT_VALUE
            signal["position_exit_zscore_method"] = setup["position_exit_zscore_method"]

            run_name = _run_name(setup, entry_z)
            config["report"] = {"run_name": run_name}
            expected_names.add(f"{run_name}.yaml")

            output_path = OUTPUT_DIR / f"{run_name}.yaml"
            with output_path.open("w", encoding="utf-8", newline="\n") as handle:
                yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    for path in OUTPUT_DIR.glob("*.yaml"):
        if path.name not in expected_names:
            path.unlink()

    print(f"generated {len(expected_names)} configs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
