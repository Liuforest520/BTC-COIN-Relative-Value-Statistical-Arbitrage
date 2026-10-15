"""Build the full TradFi RTH Walk-Forward configuration from the keep list."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIR_LIST = ROOT / "research" / "新增tradfi交易对" / "配对统计" / "筛选剔除" / "保留清单.csv"
DEFAULT_RTH_DIR = ROOT / "research" / "新增tradfi交易对" / "by_stock_adjusted_RTH"
DEFAULT_OUTPUT = ROOT / "config" / "config_walk_forward_tradfi_full.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-list", type=Path, default=DEFAULT_PAIR_LIST)
    parser.add_argument("--rth-dir", type=Path, default=DEFAULT_RTH_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    pairs = []
    symbols = {}
    missing = []
    used_pair_ids = set()
    with args.pair_list.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            x_symbol = _symbol(row.get("标的A"))
            y_symbol = _symbol(row.get("标的B"))
            if not x_symbol or not y_symbol:
                continue
            x_path = args.rth_dir / f"{_ticker(x_symbol)}.csv"
            y_path = args.rth_dir / f"{_ticker(y_symbol)}.csv"
            if not x_path.exists() or not y_path.exists():
                missing.append(f"{x_symbol}-{y_symbol}")
                continue

            pair_id = _pair_id(x_symbol, y_symbol)
            if pair_id in used_pair_ids:
                pair_id = f"{pair_id}_{len(pairs) + 1}"
            used_pair_ids.add(pair_id)
            pairs.append({
                "pair_id": pair_id,
                "target_capital": 30000,
                "x_symbol": x_symbol,
                "y_symbol": y_symbol,
                "market": "tradfi",
            })
            symbols[x_symbol] = {"path": _relative_path(x_path)}
            symbols[y_symbol] = {"path": _relative_path(y_path)}

    config = {
        "data": {"symbols": dict(sorted(symbols.items())), "benchmarks": {}},
        "backtest": {
            "initial_cash": 3000000,
            "start_time": "2024-01-01",
            "end_time": "2026-11-01",
        },
        "active_setup": "tradfi_walk_forward",
        "setups": {
            "tradfi_walk_forward": {
                "strategy": "multi_pair",
                "pairs": pairs,
                "execution": {"order_type": "market"},
                "rebalance": {
                    "enabled": True,
                    "candidate_quality": {
                        "enabled": True,
                        "min_theoretical_zero_return_x_move": 0.2,
                        "max_adf_pvalue": 1.0,
                        "min_expected_net_return": 0.0,
                    },
                    "profitable_position_replacement": {
                        "enabled": True,
                        "min_net_return": 0.0,
                        "min_convergence_ratio": 0.8,
                    },
                },
                "pipelines": {"tradfi": _pipeline()},
            }
        },
        "cost": {
            "fee_rate": 0.0005,
            "slippage_bps": 1,
            "funding_enabled": True,
        },
        "risk": {
            "enabled": False,
            "mode": "pass_through",
            "max_leverage": 1.0,
            "diagnostic_sample_interval_bars": 60,
            "order_hedge_ratio_tolerance": 0.02,
            "position_hedge_ratio_tolerance": 0.05,
        },
        "report": {"run_name": "tradfi_walk_forward_numba"},
        "walk_forward": {
            "enabled": True,
            "train_months": 6,
            "validation_months": 1,
            "step_months": 1,
            "start_date": "2024-01-01",
            "end_date": "2026-11-01",
            "warmup": {"enabled": True, "max_bars": None},
            "selection": {
                "objective": "stable_score",
                "min_trades": 20,
                "max_drawdown": None,
                "subperiod_count": 3,
                "min_active_subperiods": 2,
                "exclude_top_fraction": 0.02,
                "min_valid_neighbors": 2,
                "max_own_neighbor_gap": 0.15,
            },
            "parameter_grid": {
                "estimator": {
                    "regression_method": ["price", "log_price"],
                    "model_lookback_bars": [1080, 1440, 2160, 2880, 7200, 10080],
                    "model_update_interval_bars": [120, 240, 480],
                    "residual_adf_max_pvalue": [1.0],
                },
                "signal": {
                    "entry_z": [2.5, 3.0, 3.5, 4.0],
                    "exit_z": [0.25, 0.5],
                    "reversion_ma_lookback_bars": [60],
                    "reversion_ma_short_lookback_bars": [15],
                },
                "protection": {
                    "pair_loss_stop_enabled": [True],
                    "pair_loss_stop_return": [0.2, 0.4, 1.0],
                },
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")

    print(f"wrote: {args.output}")
    print(f"pairs: {len(pairs)}")
    print(f"symbols: {len(symbols)}")
    print("missing pairs:")
    for pair in missing:
        print(f"  {pair}")


def _symbol(value: str | None) -> str:
    value = str(value or "").strip().upper()
    if not value:
        return ""
    return value if value.endswith("USDT") else f"{value}USDT"


def _ticker(symbol: str) -> str:
    return symbol[:-4] if symbol.endswith("USDT") else symbol


def _pair_id(x_symbol: str, y_symbol: str) -> str:
    value = f"{_ticker(x_symbol)}_{_ticker(y_symbol)}".lower()
    value = re.sub(r"[^a-z0-9_]+", "_", value).strip("_")
    return value or "pair"


def _relative_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _pipeline() -> dict:
    return {
        "estimator": {
            "method": "tls",
            "regression_method": "log_price",
            "model_timeframe": "1m",
            "model_lookback_bars": 2880,
            "model_update_interval_bars": 240,
            "position_update_policy": "freeze",
            "residual_adf_max_pvalue": 1.0,
            "residual_adf_maxlag": 1,
            "residual_adf_regression": "c",
        },
        "signal": {
            "method": "zscore_reversion_ma",
            "entry_z": 3.0,
            "exit_z": 0.5,
            "reversion_ma_lookback_bars": 60,
            "reversion_ma_short_lookback_bars": 15,
            "reversion_min_samples": 120,
        },
        "sizing": {
            "method": "beta_neutral",
            "hedge_beta_lookback_bars": 1440,
            "hedge_beta_min_samples": 1440,
        },
        "portfolio": {"method": "pair_target_capital"},
        "protection": {
            "enabled": True,
            "pair_loss_stop_enabled": True,
            "pair_loss_stop_return": 0.4,
            "pair_loss_stop_freeze_bars": 0,
            "pair_loss_stop_freeze_model_lookback_multiplier": 1.0,
            "wait_for_model_update_after_non_z_exit": True,
        },
    }


if __name__ == "__main__":
    main()
