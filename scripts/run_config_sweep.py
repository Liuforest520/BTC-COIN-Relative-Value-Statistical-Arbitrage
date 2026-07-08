from pathlib import Path
import csv
import sys
import traceback

import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.backtest import run_backtest
from core.modules.logger import logger
from core.modules.reporting import export_backtest_report


def main():
    sweep_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config/sweep.yaml")
    sweep_path = resolve_path(sweep_path)
    sweep = read_yaml(sweep_path)
    manifest_path = resolve_path(sweep.get("manifest_path", Path(sweep["output_dir"]) / "manifest.csv"))
    result_path = resolve_path(sweep["result_path"])
    sort_by = sweep.get("sort_by", "sharpe")
    top_backtest_count = int(sweep.get("top_backtest_count", 5))
    top_backtest_output_dir = resolve_path(
        sweep.get("top_backtest_output_dir", result_path.parent / f"{result_path.stem}_top_backtests")
    )
    top_backtest_result_path = resolve_path(
        sweep.get("top_backtest_result_path", result_path.with_name(f"{result_path.stem}_top5.csv"))
    )

    rows = read_manifest(manifest_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for row in tqdm(rows, desc="Sweep", unit="config", dynamic_ncols=True, mininterval=1.0):
        result_row = dict(row)
        config_path = resolve_path(row["config_path"])

        try:
            result = run_backtest(config_path, show_progress=False)
            result_row.update(result.metrics)
            result_row["orders"] = len(result.orders)
            result_row["trades"] = len(result.trades)
            result_row["risk_checks"] = len(result.risk_history)
            result_row["error"] = ""
        except Exception as exc:
            result_row["error"] = str(exc)
            result_row["traceback"] = traceback.format_exc()

        results.append(result_row)

    results = sort_results(results, sort_by)
    write_results(result_path, results)
    top_results = run_top_backtests(results, top_backtest_count, top_backtest_output_dir)
    write_results(top_backtest_result_path, top_results)

    logger.info("sweep finished")
    logger.info("configs: {}", len(rows))
    logger.info("results: {}", result_path)
    logger.info("top backtest count: {}", len(top_results))
    logger.info("top backtest results: {}", top_backtest_result_path)
    logger.info("top backtest reports: {}", top_backtest_output_dir)


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_manifest(path):
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}. run scripts/generate_configs.py first")
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sort_results(rows, sort_by):
    key = sort_by if any(sort_by in row.keys() for row in rows) else "final_equity"

    def score(row):
        try:
            return float(row.get(key, "-inf"))
        except (TypeError, ValueError):
            return float("-inf")

    return sorted(rows, key=score, reverse=True)


def run_top_backtests(rows, count, output_dir):
    if count <= 0:
        return []

    top_rows = [row for row in rows if not row.get("error")][:count]
    if not top_rows:
        logger.warning("no successful sweep results, skip top backtests")
        return []

    results = []
    for rank, row in enumerate(tqdm(top_rows, desc="Top backtests", unit="config", dynamic_ncols=True), start=1):
        config_path = resolve_path(row["config_path"])
        result_row = {"rank": rank, **row}

        try:
            result = run_backtest(config_path, show_progress=False)
            report_dir = export_backtest_report(
                result,
                config_path,
                output_root=output_dir,
                run_name=top_run_name(rank, row),
            )
            result_row.update(result.metrics)
            result_row["orders"] = len(result.orders)
            result_row["trades"] = len(result.trades)
            result_row["risk_checks"] = len(result.risk_history)
            result_row["report_dir"] = display_path(report_dir)
            result_row["error"] = ""
        except Exception as exc:
            result_row["error"] = str(exc)
            result_row["traceback"] = traceback.format_exc()

        results.append(result_row)

    return results


def top_run_name(rank, row):
    config_id = row.get("config_id") or f"{rank:06d}"
    experiment_name = row.get("experiment_name") or row.get("active_setup") or "config"
    return f"top_{rank:02d}_{safe_name(config_id)}_{safe_name(experiment_name)}"


def safe_name(value):
    text = str(value)
    chars = [char if char.isalnum() or char in {"-", "_"} else "_" for char in text]
    name = "".join(chars).strip("_")
    return name or "value"


def display_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def write_results(path, rows):
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

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
