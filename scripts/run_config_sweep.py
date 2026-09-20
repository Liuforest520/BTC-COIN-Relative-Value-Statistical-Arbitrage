import os

# Each sweep worker is already a separate process. Keep native math libraries
# single-threaded so their internal thread pools do not oversubscribe the CPU.
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")

from pathlib import Path
import argparse
import csv
import json
import subprocess
import sys
import time
import traceback
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.backtest import run_backtest
from core.backtest.sweep_backtest import run_sweep_backtest
from core.modules.logger import logger
from core.modules.reporting import export_backtest_report
from core.modules.reporting.equity_summary import export_equity_summary_image


def main():
    _configure_sweep_logger()
    args = parse_args()
    sweep_path = Path(args.sweep_path)
    sweep_path = resolve_path(sweep_path)
    sweep = read_yaml(sweep_path)
    if "sweeps" in sweep:
        run_sweep_batch(sweep_path, sweep, args.workers)
        return

    manifest_path = resolve_path(sweep.get("manifest_path", Path(sweep["output_dir"]) / "manifest.csv"))
    result_path = resolve_path(sweep["result_path"])
    sort_by = sweep.get("sort_by", "sharpe")
    fast_summary = _as_bool(sweep.get("fast_summary", True))
    progress_interval_bars = max(0, int(sweep.get("progress_interval_bars", 100_000)))
    workers = _worker_count(args.workers, sweep.get("workers"))
    top_backtest_count = int(sweep.get("top_backtest_count", 5))
    top_backtest_output_dir = resolve_path(
        sweep.get("top_backtest_output_dir", result_path.parent / f"{result_path.stem}_top_backtests")
    )
    top_backtest_result_path = resolve_path(
        sweep.get("top_backtest_result_path", result_path.with_name(f"{result_path.stem}_top5.csv"))
    )
    export_each_equity_curve = _as_bool(sweep.get("export_each_equity_curve", False))
    equity_curve_output_dir = resolve_path(
        sweep.get("equity_curve_output_dir", result_path.parent / "equity_curves")
    )

    rows = read_manifest(manifest_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("sweep execution workers: {}", workers)
    logger.info("fast summary mode: {}", fast_summary)
    results = run_sweep_rows(
        rows,
        fast_summary=fast_summary,
        workers=workers,
        progress_interval_bars=progress_interval_bars,
        equity_curve_output_dir=equity_curve_output_dir if export_each_equity_curve else None,
    )

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
    if export_each_equity_curve:
        logger.info("equity curves: {}", equity_curve_output_dir)


def run_sweep_batch(batch_path, batch, cli_workers=None):
    sweep_paths = batch.get("sweeps")
    if not isinstance(sweep_paths, list) or not sweep_paths:
        raise ValueError("batch sweeps must be a non-empty list")

    workers = _worker_count(cli_workers, batch.get("workers"))
    script_path = Path(__file__).resolve()
    resolved_batch_path = Path(batch_path).resolve()

    logger.info("combined sweep batch: {} child sweeps", len(sweep_paths))
    for index, item in enumerate(sweep_paths, start=1):
        child_path = resolve_path(item).resolve()
        if child_path == resolved_batch_path:
            raise ValueError("combined sweep batch cannot include itself")
        if not child_path.exists():
            raise FileNotFoundError(f"child sweep not found: {child_path}")

        logger.info("combined sweep {}/{}: {}", index, len(sweep_paths), child_path)
        subprocess.run(
            [
                sys.executable,
                str(script_path),
                str(child_path),
                "--workers",
                str(workers),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

    logger.info("combined sweep batch finished: {}", batch_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_path", nargs="?", default="config/sweep.yaml")
    parser.add_argument("--workers", type=int, default=None)
    return parser.parse_args()


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


def run_sweep_rows(rows, fast_summary=True, workers=4, progress_interval_bars=100_000, equity_curve_output_dir=None):
    tasks = [(row, fast_summary, progress_interval_bars, equity_curve_output_dir) for row in rows]
    if workers <= 1 or len(tasks) <= 1:
        return [
            run_sweep_row(task)
            for task in tqdm(tasks, desc="Sweep", unit="config", dynamic_ncols=True, mininterval=1.0)
        ]

    worker_count = min(int(workers), len(tasks))
    with ProcessPoolExecutor(max_workers=worker_count, initializer=_initialize_worker) as executor:
        futures = {
            executor.submit(run_sweep_row, task): index
            for index, task in enumerate(tasks)
        }
        ordered_results = [None] * len(tasks)
        for future in tqdm(
            as_completed(futures),
            total=len(tasks),
            desc="Sweep",
            unit="config",
            dynamic_ncols=True,
            mininterval=1.0,
        ):
            ordered_results[futures[future]] = future.result()
        return ordered_results


def run_sweep_row(task):
    if len(task) == 2:
        row, fast_summary = task
        progress_interval_bars = 100_000
        equity_curve_output_dir = None
    elif len(task) == 3:
        row, fast_summary, progress_interval_bars = task
        equity_curve_output_dir = None
    else:
        row, fast_summary, progress_interval_bars, equity_curve_output_dir = task
    result_row = dict(row)
    config_path = resolve_path(row["config_path"])
    config_id = row.get("config_id", config_path.stem)
    started_at = time.perf_counter()

    try:
        if fast_summary:
            progress_callback = None
            if progress_interval_bars > 0:
                def progress_callback(processed_bars):
                    elapsed = time.perf_counter() - started_at
                    print(
                        f"[config {config_id}] {processed_bars:,} bars, {elapsed / 60:.1f} min elapsed",
                        flush=True,
                    )
            result = run_sweep_backtest(
                config_path,
                progress_callback=progress_callback,
                progress_interval_bars=max(1, progress_interval_bars),
            )
        else:
            result = run_backtest(config_path, show_progress=False)
        result_row.update(result.metrics)
        result_row["orders"] = len(result.orders)
        result_row["trades"] = len(result.trades)
        result_row["risk_checks"] = len(result.risk_history)
        if equity_curve_output_dir is not None:
            try:
                report_dir = export_sweep_equity_curve(
                    result=result,
                    config_path=config_path,
                    row=row,
                    output_root=equity_curve_output_dir,
                )
                result_row["equity_curve_dir"] = display_path(report_dir)
                result_row["equity_curve_csv"] = display_path(report_dir / "equity_curve.csv")
                result_row["equity_curve_png"] = display_path(report_dir / "portfolio_summary.png")
                result_row["equity_curve_error"] = ""
            except Exception as exc:
                result_row["equity_curve_error"] = f"{type(exc).__name__}: {exc}"
                logger.exception("equity curve export failed for {}: {}", config_id, exc)
        result_row["error"] = ""
    except Exception as exc:
        result_row["error"] = str(exc)
        result_row["traceback"] = traceback.format_exc()

    return result_row


def export_sweep_equity_curve(result, config_path, row, output_root):
    """Export the compact sweep equity curve without rerunning a detailed report."""
    config_id = safe_name(row.get("config_id") or Path(config_path).stem)
    experiment_name = safe_name(row.get("experiment_name") or Path(config_path).stem)
    report_dir = Path(output_root) / f"{config_id}_{experiment_name}"
    report_dir.mkdir(parents=True, exist_ok=True)

    equity_curve = result.equity_curve
    if isinstance(equity_curve, dict):
        timestamps = list(equity_curve.get("ts", []))
        equities = list(equity_curve.get("equity", []))
    else:
        timestamps = [item.get("ts") for item in equity_curve]
        equities = [item.get("equity") for item in equity_curve]
    if len(timestamps) != len(equities):
        raise ValueError("equity curve timestamps and values have different lengths")

    with (report_dir / "equity_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ts", "equity"])
        writer.writeheader()
        writer.writerows({"ts": ts, "equity": equity} for ts, equity in zip(timestamps, equities))
    (report_dir / "metrics.json").write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    shutil.copyfile(config_path, report_dir / "config.yaml")
    export_equity_summary_image(result, report_dir / "portfolio_summary.png")
    return report_dir


def _initialize_worker():
    # Each spawned Windows process otherwise opens the same rotating log file.
    logger.remove()


def _configure_sweep_logger():
    # Sweep parents, child sweeps and workers can coexist on Windows. Keeping
    # sweep logs on stderr avoids all of them rotating the same app.log file.
    logger.remove()
    logger.add(sys.stderr, level="INFO", enqueue=False)


def _worker_count(cli_value, config_value):
    value = cli_value if cli_value is not None else config_value
    if value is None:
        value = min(4, os.cpu_count() or 1)
    value = int(value)
    if value < 1:
        raise ValueError("workers must be at least 1")
    return value


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


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
                # Sweeps rank many configs; the per-pair chart set is only
                # wanted for the full single-run report.
                include_curve_charts=False,
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
