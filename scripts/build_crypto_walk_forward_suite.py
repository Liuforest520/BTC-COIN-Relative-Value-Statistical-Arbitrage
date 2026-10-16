"""Build the six crypto-only Walk-Forward experiment configs.

The generated YAML files are self-contained copies of the selected historical
configuration, with only the 17 pure crypto pairs retained.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


CRYPTO_PAIR_IDS = [
    "arb_op",
    "atom_dot",
    "avax_dot",
    "bico_dydx",
    "op_strk",
    "fet_pha",
    "fet_render",
    "bico_pyth",
    "bico_eigen",
    "blur_gala",
    "atom_avax",
    "dot_sei",
    "fil_render",
    "pyth_tnsr",
    "arb_strk",
    "fil_wld",
    "dydx_ldo",
]

LOOKBACK_BARS = [2880, 5760, 10080, 20160, 43200, 86400]
UPDATE_BARS = [240, 480, 720, 1440, 2880]


def _walk_forward(scope: str, transition: str) -> dict:
    return {
        "enabled": True,
        "optimization_scope": scope,
        "train_months": 12,
        "validation_months": 4,
        "step_months": 4,
        "start_date": "2023-01-01",
        "end_date": "2026-09-11",
        "warmup": {"enabled": True, "max_bars": None},
        "selection": {
            "objective": "stable_score",
            "min_trades": 0,
            "max_drawdown": None,
            "subperiod_count": 3,
            "min_active_subperiods": 2,
            "exclude_top_fraction": 0.02,
            "min_valid_neighbors": 2,
            "max_own_neighbor_gap": 0.15,
        },
        "portfolio": {
            "min_active_pairs": 1,
            "require_all_pairs": False,
            "candidate_batch_size": 16,
        },
        "process_workers": 12,
        "parameter_transition_mode": transition,
        "pair_ids": list(CRYPTO_PAIR_IDS),
        "parameter_grid": {
            "estimator": {
                "regression_method": ["price", "log_price"],
                "model_lookback_bars": list(LOOKBACK_BARS),
                "model_update_interval_bars": list(UPDATE_BARS),
                "residual_adf_max_pvalue": [0.4, 1.0],
            },
            "signal": {
                "entry_z": [2.5, 3.0, 3.5, 4.0],
                "exit_z": [0.25, 0.5, 0.75],
                "reversion_ma_lookback_bars": [60],
                "reversion_ma_short_lookback_bars": [15],
            },
            "protection": {
                "enabled": [True],
                "pair_loss_stop_enabled": [True],
                "pair_loss_stop_return": [0.15, 0.30, 0.45],
            },
        },
    }


def build_config(source_path: Path, output_path: Path, scope: str, transition: str) -> None:
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    setup_name = raw["active_setup"]
    setup = raw["setups"][setup_name]
    pairs_by_id = {str(pair.get("pair_id")): pair for pair in setup.get("pairs", [])}
    missing = [pair_id for pair_id in CRYPTO_PAIR_IDS if pair_id not in pairs_by_id]
    if missing:
        raise ValueError(f"source config is missing crypto pairs: {missing}")

    output = copy.deepcopy(raw)
    output_setup = output["setups"][setup_name]
    output_setup["pairs"] = []
    for pair_id in CRYPTO_PAIR_IDS:
        pair = copy.deepcopy(pairs_by_id[pair_id])
        pair["target_capital"] = 150000
        pair["tags"] = ["crypto"]
        output_setup["pairs"].append(pair)

    # The historical result config used an older ``data/<symbol>`` layout.
    # The current repository stores Binance crypto data under
    # ``data/crypto/<symbol>``; normalize only the symbols used by this suite.
    for symbol in {
        symbol
        for pair_id in CRYPTO_PAIR_IDS
        for symbol in (
            pairs_by_id[pair_id].get("x_symbol"),
            pairs_by_id[pair_id].get("y_symbol"),
        )
    }:
        output["data"]["symbols"][symbol] = {
            "exchange": "binance",
            "path": f"data/crypto/{symbol}/{symbol}-1m.csv",
            "funding_path": f"data/crypto/{symbol}/{symbol}-funding_rate_daily.csv",
        }

    output["backtest"]["initial_cash"] = 3000000
    output["backtest"]["start_time"] = "2023-01-01"
    output["backtest"]["end_time"] = "2026-09-11"
    pipeline = output_setup.setdefault("pipeline", {})
    protection = pipeline.setdefault("protection", {})
    protection.update({
        "enabled": True,
        "pair_loss_stop_enabled": True,
        "pair_loss_stop_return": 0.30,
    })
    output["walk_forward"] = _walk_forward(scope, transition)
    output.setdefault("report", {})["run_name"] = output_path.stem

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(output, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=(
            "results/backtests/single10_02_L2D_U4H_MA60_E3p0_ADF0p4_drop12/"
            "config.yaml"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="config/crypto_walk_forward_2023_2026",
    )
    args = parser.parse_args()
    source = Path(args.source)
    output_dir = Path(args.output_dir)
    if not source.exists():
        raise SystemExit(f"source config not found: {source}")

    for scope in ("pair", "portfolio"):
        for transition in ("close_at_boundary", "switch_immediately", "defer_until_flat"):
            name = f"config_crypto_{scope}_{transition}.yaml"
            build_config(source, output_dir / name, scope, transition)
            print(output_dir / name)


if __name__ == "__main__":
    main()
