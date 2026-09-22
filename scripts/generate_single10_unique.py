from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "config" / "generated" / "rerun_20260920_logic_67e556a_68"
OUTPUT_ROOT = ROOT / "config" / "generated" / "rerun_20260922_single10_unique"


CONFIGS = [
    ("ma60_threshold48/000007_L2D_U8H_MA60_E2p0_ADF0p4.yaml", "01_L2D_U8H_MA60_E2p0_ADF0p4"),
    ("ma60_threshold48/000002_L2D_U4H_MA60_E3p0_ADF0p4.yaml", "02_L2D_U4H_MA60_E3p0_ADF0p4"),
    ("ma60_threshold48/000017_L10D_U1D_MA60_E3p0_ADF1p0.yaml", "03_L10D_U1D_MA60_E3p0_ADF1p0"),
    ("ma60_threshold48/000006_L2D_U8H_MA60_E3p0_ADF0p4.yaml", "04_L2D_U8H_MA60_E3p0_ADF0p4"),
    ("ptc20/000017_L2D_U4H_MA240_ADF1p0.yaml", "05_L2D_U4H_MA240_ADF1p0"),
    ("ptc20/000015_L10D_U2D_MA120_ADF1p0.yaml", "06_L10D_U2D_MA120_ADF1p0"),
    ("ptc20/000009_L5D_U8H_MA120_ADF1p0.yaml", "07_L5D_U8H_MA120_ADF1p0"),
    ("ptc20/000010_L5D_U8H_MA120_ADF0p4.yaml", "08_L5D_U8H_MA120_ADF0p4"),
    ("ptc20/000020_L5D_U1D_MA240_ADF0p4.yaml", "09_L5D_U1D_MA240_ADF0p4"),
    ("ma60_threshold48/000020_L10D_U1D_MA60_E4p0_ADF0p4.yaml", "10_L10D_U1D_MA60_E4p0_ADF0p4"),
]


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    script_lines = ["$ErrorActionPreference = 'Stop'", "$configs = @("]
    for index, (relative_path, name) in enumerate(CONFIGS):
        source = SOURCE_ROOT / relative_path
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
        config.setdefault("report", {})["run_name"] = f"single10_{name}"
        destination = OUTPUT_ROOT / f"{name}.yaml"
        destination.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
            newline="\n",
        )
        comma = "," if index < len(CONFIGS) - 1 else ""
        script_lines.append(
            f'  ".\\config\\generated\\rerun_20260922_single10_unique\\{name}.yaml"{comma}'
        )

    script_lines.extend(
        [
            ")",
            "foreach ($config in $configs) {",
            '    Write-Host "开始回测: $config" -ForegroundColor Cyan',
            "    python scripts\\run_backtest.py $config",
            '    if ($LASTEXITCODE -ne 0) { throw "回测失败: $config" }',
            "}",
        ]
    )
    (OUTPUT_ROOT / "run_all.ps1").write_text(
        "\n".join(script_lines) + "\n", encoding="utf-8-sig", newline="\r\n"
    )
    (OUTPUT_ROOT / "启动命令.txt").write_text(
        'powershell -NoProfile -ExecutionPolicy Bypass -File ".\\config\\generated\\rerun_20260922_single10_unique\\run_all.ps1"\n',
        encoding="utf-8",
    )
    print(f"generated {len(CONFIGS)} configs in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
