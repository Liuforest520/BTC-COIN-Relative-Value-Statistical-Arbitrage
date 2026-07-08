from copy import deepcopy
from itertools import product
from pathlib import Path
import csv
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.modules.logger import logger


def main():
    sweep_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config/sweep.yaml")
    sweep_path = resolve_path(sweep_path)
    sweep = read_yaml(sweep_path)
    base_config = read_yaml(resolve_path(sweep["base_config"]))
    output_dir = resolve_path(sweep["output_dir"])
    manifest_path = resolve_path(sweep.get("manifest_path", output_dir / "manifest.csv"))

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    clean_generated_configs(output_dir)

    rows = []
    config_id = 1

    for experiment in sweep["experiments"]:
        name = experiment["name"]
        active_setup = experiment["active_setup"]
        params = experiment.get("params", {})

        validate_active_setup(base_config, active_setup)
        validate_params(params)
        param_names = list(params)
        combinations = list(product(*(params[item] for item in param_names))) if param_names else [()]

        for values in combinations:
            config = deepcopy(base_config)
            config["active_setup"] = active_setup

            row = {
                "config_id": f"{config_id:06d}",
                "experiment_name": name,
                "active_setup": active_setup,
            }

            for key, value in zip(param_names, values):
                set_by_path(config, key, value)
                row[key] = value

            file_name = f"{config_id:06d}_{name}.yaml"
            config_path = output_dir / file_name
            write_yaml(config_path, config)

            row["config_path"] = display_path(config_path)
            rows.append(row)
            config_id += 1

    write_manifest(manifest_path, rows)
    logger.info("generated configs: {}", len(rows))
    logger.info("manifest: {}", manifest_path)


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def display_path(path):
    try:
        return str(Path(path).relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_yaml(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def clean_generated_configs(output_dir):
    output_dir = output_dir.resolve()
    project_root = PROJECT_ROOT.resolve()
    if output_dir == project_root or project_root not in output_dir.parents:
        raise ValueError(f"refuse to clean generated configs outside project: {output_dir}")

    for path in output_dir.glob("*.yaml"):
        path.unlink()


def set_by_path(data, path, value):
    parts = path.split(".")
    current = data

    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"config path not found: {path}")
        current = current[part]

    last = parts[-1]
    if not isinstance(current, dict) or last not in current:
        raise KeyError(f"config path not found: {path}")

    current[last] = value


def validate_active_setup(config, active_setup):
    setups = config.get("setups", {})
    if active_setup not in setups:
        available = ", ".join(setups)
        raise ValueError(f"active_setup {active_setup!r} not found, available: {available}")


def validate_params(params):
    for key, values in params.items():
        if not isinstance(values, list):
            raise TypeError(f"sweep param {key!r} must be a list")
        if not values:
            raise ValueError(f"sweep param {key!r} cannot be empty")


def write_manifest(path, rows):
    if not rows:
        return

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
