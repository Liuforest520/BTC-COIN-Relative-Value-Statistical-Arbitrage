from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "config" / "generated" / "rerun_20260922_single10_unique"
OUTPUT_ROOT = ROOT / "config" / "generated" / "rerun_20260922_single6_exclude12_freeze"

SOURCE_NAMES = [
    "01_L2D_U8H_MA60_E2p0_ADF0p4.yaml",
    "02_L2D_U4H_MA60_E3p0_ADF0p4.yaml",
    "03_L10D_U1D_MA60_E3p0_ADF1p0.yaml",
    "04_L2D_U8H_MA60_E3p0_ADF0p4.yaml",
    "05_L2D_U4H_MA240_ADF1p0.yaml",
    "06_L10D_U2D_MA120_ADF1p0.yaml",
]

EXCLUDED_PAIR_IDS = {
    "dash_zec", "agld_blur", "sol_sui", "avax_sol", "1000shib_pol",
    "eigen_tnsr", "agld_gala", "aave_crv", "ordi_portal", "eigen_jto",
    "atom_sei", "fil_pha",
}


def model_freeze_multiplier(config: dict) -> float:
    setup_name = config.get("active_setup", "multi_pair_beta")
    setup = config["setups"][setup_name]
    lookback = float((setup.get("estimator", {}) or {}).get("model_lookback_bars", 0) or 0)
    if lookback == 2 * 24 * 60:
        return 5.0
    if lookback == 10 * 24 * 60:
        return 2.0
    raise ValueError(f"unexpected model lookback for freeze mapping: {lookback}")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    generated = []
    for source_name in SOURCE_NAMES:
        source = SOURCE_ROOT / source_name
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
        setup_name = config.get("active_setup", "multi_pair_beta")
        setup = config["setups"][setup_name]
        original_pairs = setup.get("pairs", []) or []
        setup["pairs"] = [
            pair for pair in original_pairs
            if str(pair.get("pair_id", "")).lower() not in EXCLUDED_PAIR_IDS
        ]
        removed = len(original_pairs) - len(setup["pairs"])
        if removed != len(EXCLUDED_PAIR_IDS):
            raise ValueError(f"{source_name}: removed {removed}, expected {len(EXCLUDED_PAIR_IDS)}")
        multiplier = model_freeze_multiplier(config)
        setup.setdefault("protection", {})[
            "pair_loss_stop_freeze_model_lookback_multiplier"
        ] = multiplier
        stem = source.stem + "_EX12_FREEZE" + str(int(multiplier))
        config.setdefault("report", {})["run_name"] = f"single6_ex12_{stem}"
        destination = OUTPUT_ROOT / f"{stem}.yaml"
        destination.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
            newline="\n",
        )
        generated.append(destination.name)

    lines = ["$ErrorActionPreference = 'Stop'", "$configs = @("]
    for index, name in enumerate(generated):
        comma = "," if index < len(generated) - 1 else ""
        lines.append(
            f'  ".\\config\\generated\\rerun_20260922_single6_exclude12_freeze\\{name}"{comma}'
        )
    lines.extend([
        ")",
        "foreach ($config in $configs) {",
        '    Write-Host "开始回测: $config" -ForegroundColor Cyan',
        "    python scripts\\run_backtest.py $config",
        '    if ($LASTEXITCODE -ne 0) { throw "回测失败: $config" }',
        "}",
    ])
    (OUTPUT_ROOT / "run_all.ps1").write_text(
        "\n".join(lines) + "\n", encoding="utf-8-sig", newline="\r\n"
    )
    (OUTPUT_ROOT / "启动命令.txt").write_text(
        'powershell -NoProfile -ExecutionPolicy Bypass -File ".\\config\\generated\\rerun_20260922_single6_exclude12_freeze\\run_all.ps1"\n',
        encoding="utf-8",
    )
    print(f"generated {len(generated)} configs in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
