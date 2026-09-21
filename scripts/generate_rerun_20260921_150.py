from __future__ import annotations

from copy import deepcopy
from csv import DictWriter
from pathlib import Path
import shutil

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "config" / "generated" / "rerun_20260920_logic_67e556a_68"
OUTPUT_ROOT = ROOT / "config" / "generated" / "rerun_20260921_150_v4"
RESULT_ROOT = ROOT / "results" / "sweeps" / "rerun_20260921_150_v4"


BASE_CONFIGS = {
    "L2D_U8H_MA60_E2p0_ADF0p4": SOURCE_ROOT / "ma60_threshold48" / "000007_L2D_U8H_MA60_E2p0_ADF0p4.yaml",
    "L2D_U8H_MA60_E3p0_ADF0p4": SOURCE_ROOT / "ma60_threshold48" / "000006_L2D_U8H_MA60_E3p0_ADF0p4.yaml",
    "L2D_U4H_MA60_E3p0_ADF0p4": SOURCE_ROOT / "ma60_threshold48" / "000002_L2D_U4H_MA60_E3p0_ADF0p4.yaml",
    "L2D_U4H_MA240_ADF1p0": SOURCE_ROOT / "ptc20" / "000017_L2D_U4H_MA240_ADF1p0.yaml",
    "L10D_U1D_MA60_E3p0_ADF1p0": SOURCE_ROOT / "ma60_threshold48" / "000017_L10D_U1D_MA60_E3p0_ADF1p0.yaml",
    "L10D_U2D_MA120_ADF1p0": SOURCE_ROOT / "ptc20" / "000015_L10D_U2D_MA120_ADF1p0.yaml",
    "L5D_U1D_MA240_ADF0p4": SOURCE_ROOT / "ptc20" / "000020_L5D_U1D_MA240_ADF0p4.yaml",
    "L5D_U8H_MA120_ADF0p4": SOURCE_ROOT / "ptc20" / "000010_L5D_U8H_MA120_ADF0p4.yaml",
    "L5D_U8H_MA120_ADF1p0": SOURCE_ROOT / "ptc20" / "000009_L5D_U8H_MA120_ADF1p0.yaml",
    "L10D_U1D_MA60_E4p0_ADF0p4": SOURCE_ROOT / "ma60_threshold48" / "000020_L10D_U1D_MA60_E4p0_ADF0p4.yaml",
}

GRANULARITIES = {
    "5m": {"minutes": 5, "short_lookback": 2400, "short_update": 720, "long_lookback": 3600, "long_update": 1440},
    "15m": {"minutes": 15, "short_lookback": 7200, "short_update": 1440, "long_lookback": 10800, "long_update": 2880},
    "30m": {"minutes": 30, "short_lookback": 14400, "short_update": 2160, "long_lookback": 21600, "long_update": 4320},
    "1h": {"minutes": 60, "short_lookback": 28800, "short_update": 2880, "long_lookback": 43200, "long_update": 5760},
    "2h": {"minutes": 120, "short_lookback": 57600, "short_update": 4320, "long_lookback": 86400, "long_update": 8640},
    "4h": {"minutes": 240, "short_lookback": 115200, "short_update": 5760, "long_lookback": 172800, "long_update": 11520},
}

GRANULARITY_PROFILES = {
    "MA30_10_E2_ADF0p4": (30, 10, 2.0, 0.4),
    "MA60_15_E3_ADF1p0": (60, 15, 3.0, 1.0),
    "MA120_30_E3_ADF0p4": (120, 30, 3.0, 0.4),
}

STOP_THRESHOLDS = (0.03, 0.05, 0.10, 0.20, 0.30)


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)


def set_run_name(config: dict, run_name: str) -> None:
    config.setdefault("report", {})["run_name"] = run_name


def protection(config: dict) -> dict:
    return config.setdefault("setups", {})[config["active_setup"]].setdefault("pipeline", {}).setdefault("protection", {})


def estimator(config: dict) -> dict:
    return config["setups"][config["active_setup"]]["pipeline"]["estimator"]


def signal(config: dict) -> dict:
    return config["setups"][config["active_setup"]]["pipeline"]["signal"]


def disable_protection(config: dict) -> None:
    p = protection(config)
    p.update({
        "enabled": False,
        "stop_loss_enabled": False,
        "pair_loss_stop_enabled": False,
        "take_profit_enabled": False,
        "max_holding_time_enabled": False,
    })


def write_manifest(path: Path, rows: list[dict]) -> None:
    fields = ["config_id", "experiment_name", "config_path"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def make_granularity_configs(rows: list[dict]) -> None:
    base = read_yaml(BASE_CONFIGS["L2D_U8H_MA60_E2p0_ADF0p4"])
    index = len(rows) + 1
    for timeframe, values in GRANULARITIES.items():
        for family in ("short", "long"):
            for profile, (ma_long, ma_short, entry_z, adf) in GRANULARITY_PROFILES.items():
                config = deepcopy(base)
                est = estimator(config)
                sig = signal(config)
                est["model_timeframe"] = timeframe
                est["model_lookback_bars"] = values[f"{family}_lookback"]
                est["model_update_interval_bars"] = values[f"{family}_update"]
                est["residual_adf_max_pvalue"] = adf
                sig["entry_z"] = entry_z
                sig["reversion_ma_lookback_bars"] = ma_long * values["minutes"]
                sig["reversion_ma_short_lookback_bars"] = ma_short * values["minutes"]
                sig["reversion_min_samples"] = ma_long * values["minutes"]
                disable_protection(config)
                name = f"granularity_{family}_{timeframe}_{profile}"
                set_run_name(config, name)
                path = OUTPUT_ROOT / "granularity" / f"{index:06d}_{name}.yaml"
                write_yaml(path, config)
                rows.append({"config_id": f"{index:06d}", "experiment_name": name, "config_path": rel(path)})
                index += 1


def max_holding_multiplier(model_name: str) -> float:
    if model_name.startswith("L2D_"):
        return 15.0  # 30D
    if model_name.startswith("L5D_"):
        return 6.0   # 30D
    if model_name.startswith("L10D_"):
        return 6.0   # 60D
    raise ValueError(model_name)


def max_holding_days(model_name: str) -> int:
    if model_name.startswith("L2D_"):
        return 30
    if model_name.startswith("L5D_"):
        return 30
    if model_name.startswith("L10D_"):
        return 60
    raise ValueError(model_name)


def mutate_protection(config: dict, *, stop: float | None, max_holding: bool, max_multiplier: float | None) -> None:
    p = protection(config)
    p.update({
        "enabled": True,
        "stop_loss_enabled": False,
        "pair_loss_stop_enabled": stop is not None,
        "pair_loss_stop_return": stop if stop is not None else 0.05,
        "pair_loss_stop_freeze_model_lookback_multiplier": 1.0,
        "take_profit_enabled": False,
        "max_holding_time_enabled": max_holding,
        "max_holding_time_model_lookback_multiplier": max_multiplier or 2.0,
        "max_holding_time_freeze_bars": 0,
        "wait_for_model_update_after_non_z_exit": True,
    })


def make_protection_configs(rows: list[dict]) -> None:
    index = len(rows) + 1
    base_configs = {name: read_yaml(path) for name, path in BASE_CONFIGS.items()}

    # Pair stop only: 10 bases x 5 thresholds.
    for model_name, base in base_configs.items():
        for threshold in STOP_THRESHOLDS:
            config = deepcopy(base)
            mutate_protection(config, stop=threshold, max_holding=False, max_multiplier=None)
            name = f"{model_name}_PLS{str(threshold).replace('.', 'p')}"
            set_run_name(config, name)
            path = OUTPUT_ROOT / "protection" / f"{index:06d}_{name}.yaml"
            write_yaml(path, config)
            rows.append({"config_id": f"{index:06d}", "experiment_name": name, "config_path": rel(path)})
            index += 1

    # Max holding only: 10 standard configs plus two 2D sensitivity configs
    # at 20D and 40D, in addition to the standard 30D setting.
    for model_name, base in base_configs.items():
        multiplier = max_holding_multiplier(model_name)
        config = deepcopy(base)
        mutate_protection(config, stop=None, max_holding=True, max_multiplier=multiplier)
        max_days = max_holding_days(model_name)
        name = f"{model_name}_MAX{max_days}D"
        set_run_name(config, name)
        path = OUTPUT_ROOT / "protection" / f"{index:06d}_{name}.yaml"
        write_yaml(path, config)
        rows.append({"config_id": f"{index:06d}", "experiment_name": name, "config_path": rel(path)})
        index += 1

    for model_name in ("L2D_U8H_MA60_E2p0_ADF0p4", "L2D_U8H_MA60_E3p0_ADF0p4"):
        for days, multiplier in ((20, 10.0), (40, 20.0)):
            config = deepcopy(base_configs[model_name])
            mutate_protection(config, stop=None, max_holding=True, max_multiplier=multiplier)
            name = f"{model_name}_MAX{days}D"
            set_run_name(config, name)
            path = OUTPUT_ROOT / "protection" / f"{index:06d}_{name}.yaml"
            write_yaml(path, config)
            rows.append({"config_id": f"{index:06d}", "experiment_name": name, "config_path": rel(path)})
            index += 1

    # Pair stop + max holding: 10 bases x 5 thresholds.
    for model_name, base in base_configs.items():
        multiplier = max_holding_multiplier(model_name)
        max_days = max_holding_days(model_name)
        for threshold in STOP_THRESHOLDS:
            config = deepcopy(base)
            mutate_protection(config, stop=threshold, max_holding=True, max_multiplier=multiplier)
            name = f"{model_name}_PLS{str(threshold).replace('.', 'p')}_MAX{max_days}D"
            set_run_name(config, name)
            path = OUTPUT_ROOT / "protection" / f"{index:06d}_{name}.yaml"
            write_yaml(path, config)
            rows.append({"config_id": f"{index:06d}", "experiment_name": name, "config_path": rel(path)})
            index += 1


def write_sweep(path: Path, manifest: Path, result_dir: Path) -> None:
    data = {
        "manifest_path": rel(manifest),
        "result_path": rel(result_dir / "results.csv"),
        "sort_by": "sharpe",
        "fast_summary": True,
        "workers": 4,
        "top_backtest_count": 0,
        "export_each_equity_curve": True,
        "equity_curve_output_dir": rel(result_dir / "equity_curves"),
    }
    write_yaml(path, data)


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise RuntimeError(f"output already exists: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True)
    all_rows: list[dict] = []
    granularity_rows: list[dict] = []
    protection_rows: list[dict] = []
    make_granularity_configs(granularity_rows)
    make_protection_configs(protection_rows)
    all_rows.extend(granularity_rows)
    all_rows.extend(protection_rows)
    write_manifest(OUTPUT_ROOT / "manifest_granularity.csv", granularity_rows)
    write_manifest(OUTPUT_ROOT / "manifest_protection.csv", protection_rows)
    write_manifest(OUTPUT_ROOT / "manifest.csv", all_rows)
    write_sweep(
        OUTPUT_ROOT / "sweep_granularity.yaml",
        OUTPUT_ROOT / "manifest_granularity.csv",
        RESULT_ROOT / "granularity",
    )
    write_sweep(
        OUTPUT_ROOT / "sweep_protection.yaml",
        OUTPUT_ROOT / "manifest_protection.csv",
        RESULT_ROOT / "protection",
    )
    write_sweep(
        OUTPUT_ROOT / "sweep_all.yaml",
        OUTPUT_ROOT / "manifest.csv",
        RESULT_ROOT / "all",
    )
    batch = {
        "sweeps": [
            rel(OUTPUT_ROOT / "sweep_granularity.yaml"),
            rel(OUTPUT_ROOT / "sweep_protection.yaml"),
        ],
        "workers": 4,
    }
    write_yaml(OUTPUT_ROOT / "sweep_batch.yaml", batch)
    run_script = OUTPUT_ROOT / "run_all.ps1"
    sweep_batch_path = rel(OUTPUT_ROOT / "sweep_batch.yaml").replace("/", "\\")
    run_script.write_text(
        "$ErrorActionPreference = 'Stop'\n"
        "$Workers = if ($args.Count -gt 0) { [int]$args[0] } else { 4 }\n"
        f"python scripts\\run_config_sweep.py .\\{sweep_batch_path} --workers $Workers\n"
        "if ($LASTEXITCODE -ne 0) { throw \"Sweep failed with exit code $LASTEXITCODE.\" }\n",
        encoding="ascii",
        newline="\r\n",
    )
    print(f"generated {len(all_rows)} configs")
    print(f"granularity={len(granularity_rows)} protection={len(protection_rows)}")


if __name__ == "__main__":
    main()
