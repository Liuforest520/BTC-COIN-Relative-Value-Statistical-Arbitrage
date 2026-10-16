"""Rolling natural-month walk-forward parameter selection and OOS runner."""
from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import itertools
import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import yaml
import numpy as np
from numba import set_num_threads
from loguru import logger

from core.backtest.backtest import Backtest
from core.backtest.pair_data_cache import NoAlignedPairDataError, load_pair_data
from core.backtest.numba_pair_optimizer import (
    apply_relative_metric_scores,
    build_candidate_grid,
    run_numba_pair_optimizer,
    score_candidate_metrics,
)
from core.backtest.numba_portfolio_optimizer import run_numba_portfolio_optimizer
from core.backtest.sweep_backtest import SweepBacktest
from core.modules.config import Config, load_config
from core.modules.reporting import export_backtest_report


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str

    @property
    def train_start_ts(self):
        return _date_ts(self.train_start)

    @property
    def train_end_ts(self):
        return _date_ts(self.train_end)

    @property
    def validation_start_ts(self):
        return _date_ts(self.validation_start)

    @property
    def validation_end_ts(self):
        return _date_ts(self.validation_end)


def generate_month_windows(start_date, end_date, train_months=2, validation_months=1, step_months=1):
    start = _as_date(start_date)
    end = _as_date(end_date)
    cursor = start
    windows = []
    while True:
        train_end = _add_months(cursor, train_months)
        nominal_validation_end = _add_months(train_end, validation_months)
        if nominal_validation_end > end:
            break
        # Keep natural-month boundaries for rolling windows, but allow the
        # caller to request a final partial validation period (for example,
        # an end date on the 11th of a month) without silently truncating it
        # to the first day of that month.
        next_cursor = _add_months(cursor, step_months)
        next_train_end = _add_months(next_cursor, train_months)
        next_nominal_validation_end = _add_months(next_train_end, validation_months)
        is_partial_final = (
            nominal_validation_end < end
            and next_nominal_validation_end > end
        )
        validation_end = end if is_partial_final else nominal_validation_end
        windows.append(WalkForwardWindow(
            train_start=cursor.isoformat(),
            train_end=train_end.isoformat(),
            validation_start=train_end.isoformat(),
            validation_end=validation_end.isoformat(),
        ))
        if is_partial_final:
            break
        cursor = next_cursor
    return windows


def expand_parameter_grid(grid: dict | None, base_parameters: dict | None = None):
    """Expand nested stage grids into independent parameter patches."""
    base = copy.deepcopy(base_parameters or {})
    grid = grid or {}
    normalized = {}
    for key, values in grid.items():
        if "." in str(key):
            stage, field = str(key).split(".", 1)
            normalized.setdefault(stage, {})[field] = values
        elif isinstance(values, dict):
            normalized[str(key)] = dict(values)
        else:
            normalized.setdefault("pipeline", {})[str(key)] = values
    dimensions = []
    for stage, fields in normalized.items():
        if stage == "pipeline":
            for field, values in fields.items():
                dimensions.append((("pipeline", field), _as_choices(values)))
        else:
            for field, values in fields.items():
                dimensions.append(((stage, field), _as_choices(values)))
    if not dimensions:
        return [base]
    patches = []
    for values in itertools.product(*(choices for _, choices in dimensions)):
        candidate = copy.deepcopy(base)
        for (stage, field), value in zip((item[0] for item in dimensions), values):
            if stage == "pipeline":
                candidate[field] = value
            else:
                candidate.setdefault(stage, {})[field] = value
        patches.append(candidate)
    return patches


def stable_score(metrics: dict, min_trades=20, max_drawdown=None, subperiod_count=3):
    del min_trades, max_drawdown, subperiod_count
    return score_candidate_metrics(metrics)["own_score"]


class WalkForwardParameterOptimizer:
    def __init__(self, config: Config, pair: dict, windows_config=None, optimizer: str = "numba"):
        self.config = config
        self.pair = dict(pair)
        self.wf = windows_config or config.walk_forward
        self.optimizer = str(optimizer or "numba").lower()
        if self.optimizer not in {"numba", "sweep"}:
            raise ValueError("optimizer must be 'numba' or 'sweep'")

    def optimize(self, window: WalkForwardWindow):
        started = time.perf_counter()
        pair_id = self.pair.get("pair_id")
        logger.info(
            "walk-forward optimize started pair={} train=[{}, {}) validation=[{}, {}) optimizer={}",
            pair_id,
            window.train_start,
            window.train_end,
            window.validation_start,
            window.validation_end,
            self.optimizer,
        )
        market = str(self.pair.get("market", "legacy"))
        base_pipeline = _pipeline_for_pair(self.config, market, self.pair)
        base_parameters = {stage: dict(base_pipeline.get(stage) or {}) for stage in ("estimator", "signal", "sizing", "protection")}
        candidates = expand_parameter_grid(self.pair.get("parameter_grid") or self.wf.parameter_grid, base_parameters)
        max_bars = _max_history_bars(candidates, base_pipeline)
        if self.wf.warmup_max_bars is not None:
            max_bars = min(max_bars, int(self.wf.warmup_max_bars))
        timeframe_minutes = _timeframe_minutes(base_pipeline)
        warmup_start_ts = window.train_start_ts
        if self.wf.warmup_enabled:
            warmup_start_ts -= int(max_bars * timeframe_minutes * 60_000)

        logger.info(
            "walk-forward expanded candidate grid pair={} train=[{}, {}) candidates={} max_warmup_bars={} warmup_start_ts={}",
            pair_id,
            window.train_start,
            window.train_end,
            len(candidates),
            max_bars,
            warmup_start_ts,
        )

        if self.optimizer == "numba":
            try:
                result = self._optimize_numba(window, candidates, base_pipeline, warmup_start_ts, max_bars)
            except NoAlignedPairDataError as exc:
                reason = (
                    f"no aligned 1m data for train window "
                    f"[{window.train_start}, {window.train_end}): {exc}"
                )
                logger.warning(
                    "walk-forward window skipped pair={} train=[{}, {}) validation=[{}, {}) reason={}",
                    pair_id,
                    window.train_start,
                    window.train_end,
                    window.validation_start,
                    window.validation_end,
                    reason,
                )
                result = {
                    "selected": None,
                    "candidates": [],
                    "warmup_start_ts": warmup_start_ts,
                    "max_warmup_bars": max_bars,
                    "skipped": True,
                    "skip_reason": reason,
                }
            logger.info(
                "walk-forward optimize finished pair={} train=[{}, {}) selected_candidate={} selection_score={} elapsed_seconds={:.3f}",
                pair_id,
                window.train_start,
                window.train_end,
                _candidate_id(result.get("selected")),
                _number((result.get("selected") or {}).get("selection_score"), None),
                time.perf_counter() - started,
            )
            return result

        rows = []
        for index, parameters in enumerate(candidates):
            candidate_config = _candidate_config(
                self.config, self.pair, market, parameters,
                data_start_ts=warmup_start_ts,
                measurement_start_ts=window.train_start_ts,
                end_ts=window.train_end_ts,
            )
            try:
                result = SweepBacktest(candidate_config, show_progress=False).run()
                target_capital = float(
                    self.pair.get("target_capital")
                    or (base_pipeline.get("sizing") or {}).get("notional")
                    or 0.0
                )
                metrics = _training_metrics(
                    dict(result.metrics),
                    result.equity_curve,
                    result.trades,
                    target_capital,
                    self.config.initial_cash,
                    window.train_start_ts,
                    window.train_end_ts,
                    self.wf.subperiod_count,
                )
                score = stable_score(metrics)
                rows.append({
                    "pair_id": self.pair.get("pair_id"),
                    "train_start": window.train_start,
                    "train_end": window.train_end,
                    "candidate_index": index,
                    "parameters": parameters,
                    "metrics": metrics,
                    "selection_score": score,
                    "eligible": _eligible(
                        metrics,
                        self.wf.min_trades,
                        self.wf.max_drawdown,
                        self.wf.min_active_subperiods,
                        self.wf.subperiod_count,
                    ),
                    "error": "",
                })
            except Exception as exc:
                rows.append({
                    "pair_id": self.pair.get("pair_id"),
                    "train_start": window.train_start,
                    "train_end": window.train_end,
                    "candidate_index": index,
                    "parameters": parameters,
                    "metrics": {},
                    "selection_score": float("-inf"),
                    "eligible": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            if (index + 1) == len(candidates) or (index + 1) % max(1, len(candidates) // 10) == 0:
                logger.info(
                    "walk-forward sweep progress pair={} train=[{}, {}) candidates={}/{} ({:.1f}%)",
                    pair_id,
                    window.train_start,
                    window.train_end,
                    index + 1,
                    len(candidates),
                    100.0 * (index + 1) / max(1, len(candidates)),
                )
        apply_relative_metric_scores(rows)
        selected = _select_best_eligible(rows)
        logger.info(
            "walk-forward optimize finished pair={} train=[{}, {}) selected_candidate={} selection_score={} elapsed_seconds={:.3f}",
            pair_id,
            window.train_start,
            window.train_end,
            _candidate_id(selected),
            _number((selected or {}).get("selection_score"), None),
            time.perf_counter() - started,
        )
        return {
            "selected": selected,
            "candidates": rows,
            "warmup_start_ts": warmup_start_ts,
            "max_warmup_bars": max_bars,
        }

    def _optimize_numba(self, window, candidates, base_pipeline, warmup_start_ts, max_bars):
        estimator = dict(base_pipeline.get("estimator") or {})
        if str(estimator.get("method", "tls")).lower() != "tls":
            raise ValueError("Numba Walk-Forward optimizer requires estimator.method=tls")
        if _timeframe_minutes(base_pipeline) != 1:
            raise ValueError("Numba Walk-Forward optimizer requires a strict 1m model timeframe")

        x_symbol = self.pair.get("x_symbol", self.pair.get("long_symbol"))
        y_symbol = self.pair.get("y_symbol", self.pair.get("short_symbol"))
        x_info = self.config.symbols.get(x_symbol) or {}
        y_info = self.config.symbols.get(y_symbol) or {}
        if not x_info.get("path") or not y_info.get("path"):
            raise ValueError(f"missing market path for pair {self.pair.get('pair_id')}: {x_symbol}, {y_symbol}")
        if str((base_pipeline.get("sizing") or {}).get("method", "beta_neutral")).lower() != "beta_neutral":
            raise ValueError("Numba Walk-Forward optimizer requires sizing.method=beta_neutral")
        if str((base_pipeline.get("signal") or {}).get("method", "zscore_reversion_ma")).lower() != "zscore_reversion_ma":
            raise ValueError("Numba Walk-Forward optimizer requires signal.method=zscore_reversion_ma")
        x_funding = Backtest._funding_path(x_symbol, x_info)
        y_funding = Backtest._funding_path(y_symbol, y_info)
        data = load_pair_data(
            x_info["path"], y_info["path"], x_funding, y_funding,
            start_ts=warmup_start_ts, end_ts=window.train_end_ts,
            measurement_start_ts=window.train_start_ts,
            warmup_bars=max_bars if self.wf.warmup_enabled else 0,
        )
        actual_warmup_start_ts = int(data.timestamps[0])
        try:
            grid = build_candidate_grid(candidates)
            logger.info(
                "walk-forward data loaded pair={} train=[{}, {}) aligned_bars={} coverage=[{}, {}) candidate_count={} x={} y={}",
                self.pair.get("pair_id"),
                window.train_start,
                window.train_end,
                data.size,
                _timestamp_text(data.timestamps[0]) if data.size else "none",
                _timestamp_text(data.timestamps[-1] + 1) if data.size else "none",
                grid.size,
                x_symbol,
                y_symbol,
            )
            search_started = time.perf_counter()
            logger.info(
                "walk-forward numba search started pair={} train=[{}, {}) candidates={}",
                self.pair.get("pair_id"),
                window.train_start,
                window.train_end,
                grid.size,
            )
            sizing = dict(base_pipeline.get("sizing") or {})
            signal = dict(base_pipeline.get("signal") or {})
            protection = dict(base_pipeline.get("protection") or {})
            rows = run_numba_pair_optimizer(
                data, grid,
                train_start_ts=window.train_start_ts,
                train_end_ts=window.train_end_ts,
                initial_equity=self.config.initial_cash,
                target_capital=float(self.pair.get("target_capital") or sizing.get("notional") or 0.0),
                fee_rate=self.config.fee_rate,
                slippage_bps=self.config.slippage_bps,
                hedge_beta_lookback_bars=int(sizing.get("hedge_beta_lookback_bars", 1440)),
                hedge_beta_min_samples=int(sizing.get("hedge_beta_min_samples", 30)),
                reversion_min_samples=int(signal.get("reversion_min_samples") or signal.get("reversion_ma_lookback_bars", 60)),
                pair_loss_stop_freeze_model_lookback_multiplier=float(
                    protection.get("pair_loss_stop_freeze_model_lookback_multiplier") or 0.0
                ),
                wait_for_model_update_after_non_z_exit=bool(
                    protection.get("wait_for_model_update_after_non_z_exit", False)
                ),
                min_trades=self.wf.min_trades,
                max_drawdown=self.wf.max_drawdown,
                subperiod_count=self.wf.subperiod_count,
                min_active_subperiods=self.wf.min_active_subperiods,
                batch_size=64,
                progress_callback=lambda completed, total: logger.info(
                    "walk-forward numba candidate progress pair={} train=[{}, {}) candidates={}/{} ({:.1f}%)",
                    self.pair.get("pair_id"),
                    window.train_start,
                    window.train_end,
                    completed,
                    total,
                    100.0 * completed / max(1, total),
                ),
            )
            logger.info(
                "walk-forward numba search finished pair={} train=[{}, {}) candidates={}/{} elapsed_seconds={:.3f}",
                self.pair.get("pair_id"),
                window.train_start,
                window.train_end,
                len(rows),
                grid.size,
                time.perf_counter() - search_started,
            )
        finally:
            data.close()

        normalized = []
        for row in rows:
            normalized.append({
                "pair_id": self.pair.get("pair_id"),
                "train_start": window.train_start,
                "train_end": window.train_end,
                **row,
            })
        apply_relative_metric_scores(normalized)
        selected = _select_best_eligible(normalized)
        if selected is None:
            logger.warning(
                "walk-forward no candidate selected pair={} train=[{}, {}) candidate_count={} errors={}",
                self.pair.get("pair_id"),
                window.train_start,
                window.train_end,
                len(normalized),
                sum(1 for row in normalized if row.get("error")),
            )
        else:
            logger.info(
                "walk-forward candidate selected pair={} train=[{}, {}) candidate={} score={} own_score={} neighbor_count={} eligible={}",
                self.pair.get("pair_id"),
                window.train_start,
                window.train_end,
                selected.get("candidate_index"),
                selected.get("selection_score"),
                selected.get("own_score"),
                selected.get("neighbor_count"),
                selected.get("eligible"),
            )
        return {
            "selected": selected,
            "candidates": normalized,
            "warmup_start_ts": actual_warmup_start_ts,
            "max_warmup_bars": max_bars,
        }



def _configure_worker_numba_threads(process_workers):
    """Avoid oversubscribing CPU and memory when workers run Numba kernels."""
    workers = max(1, int(process_workers or 1))
    cpu_count = max(1, int(os.cpu_count() or 1))
    set_num_threads(max(1, cpu_count // workers))

def _run_portfolio_window_worker(payload):
    """Compute one portfolio training window in a fresh process."""
    config_path, pairs, window_payload, candidates, max_bars, candidate_batch_size, window_index, process_workers = payload
    _configure_worker_numba_threads(process_workers)
    config = load_config(config_path)
    window = WalkForwardWindow(**window_payload)
    logger.info(
        "portfolio window worker started window={} train=[{}, {}) candidates={} pid={}",
        window_index, window.train_start, window.train_end, len(candidates), os.getpid(),
    )
    warmup_start_ts = window.train_start_ts
    if config.walk_forward.warmup_enabled:
        warmup_start_ts -= int(max_bars * 60_000)
    result = run_numba_portfolio_optimizer(
        config, pairs, window, candidates, warmup_start_ts, max_bars,
        config.walk_forward, candidate_batch_size=candidate_batch_size,
    )
    logger.info(
        "portfolio window worker finished window={} train=[{}, {}) candidates={} selected={}",
        window_index, window.train_start, window.train_end, len(candidates),
        (result.get("selected") or {}).get("candidate_index"),
    )
    return window_index, result


def _run_pair_window_worker(payload):
    """Compute one pair/window task in a fresh process."""
    config_path, pair, window_payload, optimizer, window_index, process_workers = payload
    _configure_worker_numba_threads(process_workers)
    config = load_config(config_path)
    window = WalkForwardWindow(**window_payload)
    pair_id = pair.get("pair_id")
    logger.info(
        "pair window worker started pair={} window={} train=[{}, {}) pid={}",
        pair_id, window_index, window.train_start, window.train_end, os.getpid(),
    )
    result = WalkForwardParameterOptimizer(
        config, pair, config.walk_forward, optimizer=optimizer,
    ).optimize(window)
    logger.info(
        "pair window worker finished pair={} window={} selected={}",
        pair_id, window_index, (result.get("selected") or {}).get("candidate_index"),
    )
    return pair_id, window_index, result


def _submit_parallel_portfolio_window_futures(
    config_path, selected_pairs, windows, public_candidates, max_bars,
    candidate_batch_size, workers,
):
    payloads = [
        (
            str(config_path),
            [dict(pair) for pair in selected_pairs],
            window.__dict__,
            public_candidates,
            max_bars,
            candidate_batch_size,
            window_index,
            workers,
        )
        for window_index, window in enumerate(windows, start=1)
    ]
    logger.info(
        "portfolio process parallelism enabled workers={} windows={}",
        workers, len(payloads),
    )
    executor = ProcessPoolExecutor(max_workers=workers)
    futures = [executor.submit(_run_portfolio_window_worker, payload) for payload in payloads]
    return executor, futures


def _parallel_pair_window_results(
    config_path, selected_pairs, windows, optimizer, workers,
    candidate_output_path=None,
):
    payloads = [
        (str(config_path), dict(pair), window.__dict__, optimizer, window_index, workers)
        for pair in selected_pairs
        for window_index, window in enumerate(windows, start=1)
    ]
    results = {}
    logger.info(
        "pair process parallelism enabled workers={} tasks={}",
        workers, len(payloads),
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_run_pair_window_worker, payload):
            (payload[1].get("pair_id"), payload[-1])
            for payload in payloads
        }
        for future in as_completed(futures):
            key = futures.pop(future)
            pair_id, window_index, result = future.result()
            candidates = result.pop("candidates", [])
            if candidate_output_path is not None and candidates:
                _append_csv(candidate_output_path, _flatten_candidate_rows(candidates))
            results[(pair_id, window_index)] = result
            logger.info(
                "pair window process completed pair={} window={} workers={}",
                key[0], key[1], workers,
            )
    return results

def _run_portfolio_walk_forward(
    config_path,
    config,
    output_root=None,
    run_name=None,
    show_progress=False,
    optimizer="numba",
    include_trade_review=True,
    include_curve_charts=True,
    candidate_batch_size=None,
    process_workers=None,
):
    """Run the shared-parameter portfolio scope without changing pair scope."""
    if str(optimizer).lower() != "numba":
        raise ValueError("portfolio optimization currently requires --optimizer numba")
    wf = config.walk_forward
    process_workers = max(1, int(process_workers if process_workers is not None else getattr(wf, "process_workers", 1)))
    start = wf.start_date or _date_from_ts(config.backtest_start_ts)
    end = wf.end_date or _date_from_ts(config.backtest_end_ts)
    if start is None or end is None:
        raise ValueError("walk_forward requires start_date/end_date or backtest.start_time/end_time")
    windows = generate_month_windows(start, end, wf.train_months, wf.validation_months, wf.step_months)
    if not windows:
        raise ValueError("walk_forward produced no complete train/validation windows")

    requested_pair_ids = set(getattr(wf, "pair_ids", []) or [])
    selected_pairs = [
        pair for pair in config.strategy.pairs
        if pair.get("enabled", True) and (not requested_pair_ids or pair.get("pair_id") in requested_pair_ids)
    ]
    if not selected_pairs:
        raise ValueError(f"walk_forward pair_ids matched no enabled pairs: {sorted(requested_pair_ids)}")
    portfolio_base = _portfolio_base_parameters(config, selected_pairs)
    public_candidates = expand_parameter_grid(wf.parameter_grid, portfolio_base)
    max_bars = max(
        _max_history_bars(public_candidates, _pipeline_for_pair(config, str(pair.get("market", "legacy")), pair))
        for pair in selected_pairs
    )
    if wf.warmup_max_bars is not None:
        max_bars = min(max_bars, int(wf.warmup_max_bars))
    logger.info(
        "portfolio walk-forward run started pairs={} windows={} candidates={} train_months={} validation_months={} step_months={} date_range=[{}, {})",
        len(selected_pairs), len(windows), len(public_candidates), wf.train_months,
        wf.validation_months, wf.step_months, start, end,
    )

    output_dir = Path(output_root or "results/walk_forward") / (run_name or config_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(output_dir / "walk_forward_config.yaml", {
        "walk_forward": _dataclass_dict(wf), "optimizer": optimizer,
        "optimization_scope": "portfolio", "windows": [window.__dict__ for window in windows],
    })
    if config_path.exists():
        (output_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    schedule = {pair.get("pair_id"): [] for pair in selected_pairs}
    candidate_rows = []
    portfolio_pair_rows = []
    window_rows = []
    previous = None
    previous_candidate_index = None
    run_started = time.perf_counter()
    parallel_executor = None
    parallel_futures = None
    if process_workers > 1:
        parallel_executor, parallel_futures = _submit_parallel_portfolio_window_futures(
            config_path, selected_pairs, windows, public_candidates, max_bars,
            candidate_batch_size, process_workers,
        )
    for window_index, window in enumerate(windows, start=1):
        window_started = time.perf_counter()
        warmup_start_ts = window.train_start_ts
        if wf.warmup_enabled:
            warmup_start_ts -= int(max_bars * 60_000)
        logger.info(
            "portfolio window started window={}/{} train=[{}, {}) validation=[{}, {}) candidates={} warmup_start_ts={}",
            window_index, len(windows), window.train_start, window.train_end,
            window.validation_start, window.validation_end, len(public_candidates), warmup_start_ts,
        )
        try:
            if parallel_futures is not None:
                completed_index, result = parallel_futures[window_index - 1].result()
                if completed_index != window_index:
                    raise RuntimeError(f"portfolio worker returned window {completed_index}, expected {window_index}")
            else:
                result = run_numba_portfolio_optimizer(
                    config, selected_pairs, window, public_candidates,
                    warmup_start_ts, max_bars, wf,
                    progress_callback=lambda done, total, active, skipped: logger.info(
                        "portfolio candidate progress window={}/{} candidates={}/{} active_pairs={} skipped_pairs={}",
                        window_index, len(windows), done, total, active, skipped,
                    ),
                    candidate_batch_size=candidate_batch_size,
                )
        except Exception:
            if parallel_executor is not None:
                parallel_executor.shutdown(wait=True, cancel_futures=True)
                parallel_executor = None
            logger.exception(
                "portfolio window failed window={}/{} train=[{}, {}) validation=[{}, {})",
                window_index, len(windows), window.train_start, window.train_end,
                window.validation_start, window.validation_end,
            )
            raise
        candidate_rows.extend(result.get("candidates", []))
        portfolio_pair_rows.extend([
            {
                "pair_id": item.get("pair_id"),
                "candidate_index": item.get("candidate_index"),
                "train_start": window.train_start,
                "train_end": window.train_end,
                "metrics": item.get("metrics", {}),
            }
            for item in result.get("pair_metrics", [])
        ])
        selected = result.get("selected")
        active_count = int(result.get("active_pair_count", 0))
        skipped_count = len(result.get("skipped_pairs", []))
        if selected is None:
            has_previous = previous is not None
            base_reason = result.get("skip_reason") or "no portfolio candidate passed selection eligibility rules"
            parameters = previous or {}
            selection_score = None
            global_index = previous_candidate_index
            fallback = has_previous
            skipped = not has_previous
            reason = f"{base_reason}; reused previous global parameters" if fallback else f"{base_reason}; global parameters disabled"
        else:
            has_previous = False
            parameters = selected["parameters"]
            selection_score = selected.get("selection_score")
            global_index = selected.get("candidate_index")
            fallback = False
            skipped = False
            reason = selected.get("selection_reason") or "highest eligible portfolio score"
            previous = parameters
            previous_candidate_index = global_index

        skipped_ids = {item.get("pair_id") for item in result.get("skipped_pairs", [])}
        for pair in selected_pairs:
            pair_id = pair.get("pair_id")
            pair_skipped = skipped or pair_id in skipped_ids
            schedule[pair_id].append({
                "pair_id": pair_id,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "validation_start": window.validation_start,
                "validation_end": window.validation_end,
                "parameters": copy.deepcopy(parameters),
                "train_metrics": (selected or {}).get("metrics", {}),
                "selection_score": selection_score,
                "selection_reason": reason,
                "data_coverage": {
                    "warmup_start_ts": result.get("warmup_start_ts", warmup_start_ts),
                    "max_warmup_bars": max_bars,
                },
                "fallback": fallback,
                "skipped": pair_skipped,
                "optimization_scope": "portfolio",
                "global_candidate_index": global_index,
                "active_pair_count": active_count,
                "skipped_pair_count": skipped_count,
                "active_pair_ratio": active_count / max(1, len(selected_pairs)),
            })
        window_rows.append({
            "optimization_scope": "portfolio",
            "window_index": window_index,
            "train_start": window.train_start,
            "train_end": window.train_end,
            "validation_start": window.validation_start,
            "validation_end": window.validation_end,
            "selected": bool(selected),
            "global_candidate_index": global_index,
            "selection_score": selection_score,
            "active_pair_count": active_count,
            "skipped_pair_count": skipped_count,
            "active_pair_ratio": active_count / max(1, len(selected_pairs)),
            "fallback": fallback,
            "skipped": skipped,
            "skip_reason": "" if selected else reason,
            "skipped_pairs": json.dumps(result.get("skipped_pairs", []), ensure_ascii=False),
            "selection_filter_summary": json.dumps(result.get("selection_filter_summary", {}), ensure_ascii=False),
        })
        _write_json(output_dir / "parameter_schedule.json", schedule)
        _write_csv(output_dir / "parameter_schedule.csv", _schedule_rows(schedule))
        _write_csv(output_dir / "window_summary.csv", window_rows)
        _write_csv(output_dir / "portfolio_window_summary.csv", window_rows)
        _write_json(output_dir / "walk_forward_checkpoint.json", {
            "optimization_scope": "portfolio",
            "completed_windows": window_index,
            "total_windows": len(windows),
            "last_completed_window": window.__dict__,
            "schedule_file": "parameter_schedule.json",
        })
        logger.info(
            "portfolio window finished window={}/{} selected_candidate={} score={} active_pairs={} skipped_pairs={} elapsed_seconds={:.3f} checkpoint_saved={}",
            window_index, len(windows), global_index, selection_score, active_count,
            skipped_count, time.perf_counter() - window_started, output_dir / "parameter_schedule.json",
        )

    if parallel_executor is not None:
        parallel_executor.shutdown(wait=True)
        parallel_executor = None
    usable_schedule = {key: value for key, value in schedule.items() if value}
    first_validation = windows[0].validation_start_ts
    final_config = copy.deepcopy(config)
    final_config.strategy.pairs = [pair for pair in final_config.strategy.pairs if pair.get("pair_id") in usable_schedule]
    final_config.data_start_ts = min(
        _number(entry["data_coverage"]["warmup_start_ts"], first_validation)
        for entries in usable_schedule.values() for entry in entries
    )
    final_config.measurement_start_ts = first_validation
    final_config.backtest_end_ts = windows[-1].validation_end_ts
    final_config.backtest_end_exclusive = True
    logger.info("portfolio parameter search finished; starting continuous OOS backtest")
    final = Backtest(
        final_config, show_progress=show_progress,
        parameter_schedule=usable_schedule, oos_start_ts=first_validation,
    ).run()

    _write_json(output_dir / "parameter_schedule.json", usable_schedule)
    _write_yaml(output_dir / "walk_forward_config.yaml", {
        "walk_forward": _dataclass_dict(wf), "optimizer": optimizer,
        "optimization_scope": "portfolio", "windows": [window.__dict__ for window in windows],
    })
    _write_csv(output_dir / "parameter_schedule.csv", _schedule_rows(usable_schedule))
    flattened = _flatten_candidate_rows(candidate_rows)
    _write_csv(output_dir / "parameter_candidates.csv", flattened)
    _write_csv(output_dir / "portfolio_candidates.csv", flattened)
    _write_csv(output_dir / "portfolio_pair_train_metrics.csv", _portfolio_pair_rows(portfolio_pair_rows))
    _write_csv(output_dir / "window_summary.csv", window_rows)
    _write_csv(output_dir / "portfolio_window_summary.csv", window_rows)
    _write_csv(output_dir / "pair_window_metrics.csv", _pair_window_rows(final, usable_schedule, config_path))
    _write_csv(output_dir / "portfolio_equity_curve.csv", _rows(final.equity_curve))
    _write_csv(output_dir / "oos_equity_curve.csv", _rows(_filter_curve(final.equity_curve, first_validation)))
    _write_csv(output_dir / "trades.csv", _object_rows(final.trades))
    _write_csv(output_dir / "orders.csv", _object_rows(final.orders))
    _write_csv(output_dir / "funding_payments.csv", _object_rows(final.funding_payments))
    _write_json(output_dir / "metrics.json", final.metrics)
    _write_json(output_dir / "final_position_valuation.json", final.final_position_valuation)
    _write_csv(output_dir / "metrics.csv", [{"metric": key, "value": value} for key, value in final.metrics.items()])
    if config_path.exists():
        (output_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    report_dir = export_backtest_report(
        final, config_path, output_root=output_dir, run_name="backtest_report",
        include_trade_review=include_trade_review, include_curve_charts=include_curve_charts,
        release_result_memory=False,
    )
    (output_dir / "backtest_report_dir.txt").write_text(str(report_dir), encoding="utf-8")
    logger.info("portfolio walk-forward run finished output_dir={} total_elapsed_seconds={:.3f}", output_dir, time.perf_counter() - run_started)
    return output_dir, final, usable_schedule


def run_walk_forward(
    config_path="config/config.yaml",
    output_root=None,
    run_name=None,
    show_progress=False,
    optimizer="numba",
    include_trade_review=True,
    include_curve_charts=True,
    optimization_scope=None,
    candidate_batch_size=None,
    process_workers=None,
    parameter_transition_mode=None,
):
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    wf = config.walk_forward
    if parameter_transition_mode is not None:
        mode = str(parameter_transition_mode).strip().lower()
        if mode not in {
            "defer_until_flat", "close_at_boundary", "switch_immediately",
        }:
            raise ValueError(
                "parameter_transition_mode must be 'defer_until_flat', "
                "'close_at_boundary', or 'switch_immediately'"
            )
        wf.parameter_transition_mode = mode
    optimization_scope = str(optimization_scope or getattr(wf, "optimization_scope", "pair")).lower()
    if optimization_scope not in {"pair", "portfolio"}:
        raise ValueError("optimization_scope must be 'pair' or 'portfolio'")
    if optimization_scope == "portfolio":
        return _run_portfolio_walk_forward(
            config_path=config_path,
            config=config,
            output_root=output_root,
            run_name=run_name,
            show_progress=show_progress,
            optimizer=optimizer,
            include_trade_review=include_trade_review,
            include_curve_charts=include_curve_charts,
            candidate_batch_size=candidate_batch_size,
            process_workers=process_workers,
        )
    start = wf.start_date or _date_from_ts(config.backtest_start_ts)
    end = wf.end_date or _date_from_ts(config.backtest_end_ts)
    if start is None or end is None:
        raise ValueError("walk_forward requires start_date/end_date or backtest.start_time/end_time")
    windows = generate_month_windows(start, end, wf.train_months, wf.validation_months, wf.step_months)
    if not windows:
        raise ValueError("walk_forward produced no complete train/validation windows")

    schedule = {}
    window_rows = []
    requested_pair_ids = set(getattr(wf, "pair_ids", []) or [])
    selected_pairs = [
        pair for pair in config.strategy.pairs
        if pair.get("enabled", True) and (not requested_pair_ids or pair.get("pair_id") in requested_pair_ids)
    ]
    if not selected_pairs:
        raise ValueError(f"walk_forward pair_ids matched no enabled pairs: {sorted(requested_pair_ids)}")
    total_tasks = len(selected_pairs) * len(windows)
    completed_tasks = 0
    run_started = time.perf_counter()
    output_dir = Path(output_root or "results/walk_forward") / (run_name or config_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_output_path = output_dir / "parameter_candidates.csv"
    candidate_output_path.unlink(missing_ok=True)
    _write_yaml(output_dir / "walk_forward_config.yaml", {
        "walk_forward": _dataclass_dict(wf), "optimizer": optimizer,
        "optimization_scope": "pair", "windows": [window.__dict__ for window in windows],
    })
    if config_path.exists():
        (output_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    logger.info(
        "walk-forward run started config={} pairs={} windows={} total_pair_windows={} optimizer={} train_months={} validation_months={} step_months={} date_range=[{}, {})",
        config_path,
        len(selected_pairs),
        len(windows),
        total_tasks,
        optimizer,
        wf.train_months,
        wf.validation_months,
        wf.step_months,
        start,
        end,
    )
    parallel_pair_results = None
    effective_workers = max(1, int(process_workers if process_workers is not None else getattr(wf, "process_workers", 1)))
    if effective_workers > 1:
        parallel_pair_results = _parallel_pair_window_results(
            config_path, selected_pairs, windows, optimizer, effective_workers,
            candidate_output_path=candidate_output_path,
        )
    for pair_index, pair in enumerate(selected_pairs, start=1):
        pair_id = pair.get("pair_id")
        pair_started = time.perf_counter()
        logger.info(
            "walk-forward pair started pair_index={}/{} pair={} windows={}",
            pair_index,
            len(selected_pairs),
            pair_id,
            len(windows),
        )
        pair_optimizer = WalkForwardParameterOptimizer(config, pair, wf, optimizer=optimizer)
        pair_schedule = []
        previous = None
        for window_index, window in enumerate(windows, start=1):
            task_started = time.perf_counter()
            logger.info(
                "walk-forward window started pair={}:{} window={}/{} train=[{}, {}) validation=[{}, {}) global_completed={}/{}",
                pair_id,
                window_index,
                window_index,
                len(windows),
                window.train_start,
                window.train_end,
                window.validation_start,
                window.validation_end,
                completed_tasks,
                total_tasks,
            )
            try:
                if parallel_pair_results is not None:
                    result = parallel_pair_results[(pair_id, window_index)]
                else:
                    result = pair_optimizer.optimize(window)
            except Exception:
                logger.exception(
                    "walk-forward window failed pair={}:{} train=[{}, {}) validation=[{}, {})",
                    pair_id,
                    window_index,
                    window.train_start,
                    window.train_end,
                    window.validation_start,
                    window.validation_end,
                )
                raise
            selected = result["selected"]
            if selected is None:
                has_previous = previous is not None
                base_reason = result.get("skip_reason") or "no candidate passed the selection eligibility rules"
                entry = {
                    "pair_id": pair_id, "train_start": window.train_start, "train_end": window.train_end,
                    "validation_start": window.validation_start, "validation_end": window.validation_end,
                    "parameters": previous or {}, "train_metrics": {}, "selection_score": None,
                    "selection_reason": (
                        f"{base_reason}; reused previous validation parameters"
                        if has_previous else
                        f"{base_reason}; pair disabled for this validation window"
                    ),
                    "data_coverage": {"warmup_start_ts": result["warmup_start_ts"], "max_warmup_bars": result["max_warmup_bars"]},
                    "fallback": has_previous,
                    "skipped": not has_previous,
                }
            else:
                entry = {
                    "pair_id": pair_id, "train_start": window.train_start, "train_end": window.train_end,
                    "validation_start": window.validation_start, "validation_end": window.validation_end,
                    "parameters": selected["parameters"], "train_metrics": selected["metrics"],
                    "selection_score": selected["selection_score"],
                    "selection_reason": selected.get("selection_reason") or "highest score among eligible candidates",
                    "data_coverage": {"warmup_start_ts": result["warmup_start_ts"], "max_warmup_bars": result["max_warmup_bars"]},
                    "fallback": False, "skipped": False,
                }
                previous = selected["parameters"]
            pair_schedule.append(entry)
            schedule[pair_id] = pair_schedule
            candidates = result.pop("candidates", [])
            if candidates:
                _append_csv(candidate_output_path, _flatten_candidate_rows(candidates))
            window_rows.append({"pair_id": pair_id, "train_start": window.train_start, "train_end": window.train_end, "validation_start": window.validation_start, "validation_end": window.validation_end, "selected": bool(selected), "selection_score": entry["selection_score"], "fallback": entry["fallback"], "skipped": entry["skipped"]})
            _write_json(output_dir / "parameter_schedule.json", schedule)
            _write_csv(output_dir / "parameter_schedule.csv", _schedule_rows(schedule))
            _write_csv(output_dir / "window_summary.csv", window_rows)
            _write_json(output_dir / "walk_forward_checkpoint.json", {
                "optimization_scope": "pair",
                "completed_pair_windows": completed_tasks + 1,
                "total_pair_windows": total_tasks,
                "last_pair_id": pair_id,
                "last_completed_window": window.__dict__,
                "schedule_file": "parameter_schedule.json",
            })
            completed_tasks += 1
            elapsed = time.perf_counter() - run_started
            rate = completed_tasks / elapsed if elapsed > 0 else 0.0
            remaining = (total_tasks - completed_tasks) / rate if rate > 0 else 0.0
            logger.info(
                "walk-forward window finished pair={}:{} selected_candidate={} score={} fallback={} skipped={} task_elapsed_seconds={:.3f} global_progress={}/{} ({:.2f}%) eta_seconds={:.0f}",
                pair_id,
                window_index,
                _candidate_id(selected),
                entry["selection_score"],
                entry["fallback"],
                entry["skipped"],
                time.perf_counter() - task_started,
                completed_tasks,
                total_tasks,
                100.0 * completed_tasks / max(1, total_tasks),
                remaining,
            )
        schedule[pair_id] = pair_schedule
        logger.info(
            "walk-forward pair finished pair_index={}/{} pair={} pair_elapsed_seconds={:.3f} global_progress={}/{} ({:.2f}%)",
            pair_index,
            len(selected_pairs),
            pair_id,
            time.perf_counter() - pair_started,
            completed_tasks,
            total_tasks,
            100.0 * completed_tasks / max(1, total_tasks),
        )

    first_validation = windows[0].validation_start_ts
    final_config = copy.deepcopy(config)
    final_config.strategy.pairs = [
        pair for pair in final_config.strategy.pairs
        if pair.get("pair_id") in schedule
    ]
    final_config.data_start_ts = min(_number(entry["data_coverage"]["warmup_start_ts"], first_validation) for entries in schedule.values() for entry in entries)
    final_config.measurement_start_ts = first_validation
    final_config.backtest_end_ts = windows[-1].validation_end_ts
    final_config.backtest_end_exclusive = True
    logger.info(
        "walk-forward parameter search finished completed_pair_windows={}/{} elapsed_seconds={:.3f}; starting continuous OOS backtest",
        completed_tasks,
        total_tasks,
        time.perf_counter() - run_started,
    )
    final_started = time.perf_counter()
    final = Backtest(final_config, show_progress=show_progress, parameter_schedule=schedule, oos_start_ts=first_validation).run()
    logger.info(
        "walk-forward continuous OOS backtest finished elapsed_seconds={:.3f} final_equity={} total_return={}",
        time.perf_counter() - final_started,
        final.metrics.get("final_equity"),
        final.metrics.get("total_return"),
    )

    _write_json(output_dir / "parameter_schedule.json", schedule)
    _write_yaml(output_dir / "walk_forward_config.yaml", {"walk_forward": _dataclass_dict(wf), "optimizer": optimizer, "windows": [window.__dict__ for window in windows]})
    _write_csv(output_dir / "parameter_schedule.csv", _schedule_rows(schedule))
    if not candidate_output_path.exists():
        candidate_output_path.write_text("", encoding="utf-8")
    _write_csv(output_dir / "window_summary.csv", window_rows)
    _write_csv(output_dir / "pair_window_metrics.csv", _pair_window_rows(final, schedule, config_path))
    _write_csv(output_dir / "portfolio_equity_curve.csv", _rows(final.equity_curve))
    _write_csv(output_dir / "oos_equity_curve.csv", _rows(_filter_curve(final.equity_curve, first_validation)))
    _write_csv(output_dir / "trades.csv", _object_rows(final.trades))
    _write_csv(output_dir / "orders.csv", _object_rows(final.orders))
    _write_csv(output_dir / "funding_payments.csv", _object_rows(final.funding_payments))
    _write_json(output_dir / "metrics.json", final.metrics)
    _write_json(output_dir / "final_position_valuation.json", final.final_position_valuation)
    _write_csv(output_dir / "metrics.csv", [{"metric": key, "value": value} for key, value in final.metrics.items()])
    if config_path.exists():
        (output_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    report_dir = export_backtest_report(
        final,
        config_path,
        output_root=output_dir,
        run_name="backtest_report",
        include_trade_review=include_trade_review,
        include_curve_charts=include_curve_charts,
        release_result_memory=False,
    )
    (output_dir / "backtest_report_dir.txt").write_text(str(report_dir), encoding="utf-8")
    logger.info("walk-forward standard backtest report saved report_dir={}", report_dir)
    logger.info(
        "walk-forward run finished output_dir={} total_elapsed_seconds={:.3f}",
        output_dir,
        time.perf_counter() - run_started,
    )
    return output_dir, final, schedule


def _pipeline_for_pair(config, market, pair=None):
    views = getattr(config.strategy, "market_pipelines", {}) or {}
    if market in views:
        pipeline = copy.deepcopy(views[market].get("pipeline") or {})
    else:
        pipeline = copy.deepcopy(config.strategy.pipeline or {})
    for stage, patch in dict((pair or {}).get("parameters") or {}).items():
        if isinstance(patch, dict):
            pipeline.setdefault(stage, {}).update(patch)
    return pipeline


def _portfolio_base_parameters(config, pairs):
    """Return common fixed parameters beneath the public search grid.

    The fast shared kernel has one public candidate definition.  Unsearched
    fields must therefore be identical across participating pairs; silently
    falling back to Numba defaults would train a different strategy than OOS.
    """
    stages = ("estimator", "signal", "sizing", "protection")
    first_pair = pairs[0]
    first = _pipeline_for_pair(
        config, str(first_pair.get("market", "legacy")), first_pair,
    )
    base = {stage: copy.deepcopy(first.get(stage) or {}) for stage in stages}
    compared_fields = {
        "estimator": (
            "method", "regression_method", "model_timeframe",
            "model_lookback_bars", "model_update_interval_bars",
            "position_update_policy", "residual_adf_max_pvalue",
        ),
        "signal": (
            "method", "entry_z", "exit_z", "reversion_ma_lookback_bars",
            "reversion_ma_short_lookback_bars", "reversion_min_samples",
        ),
        "sizing": (
            "method", "hedge_beta_lookback_bars", "hedge_beta_min_samples",
        ),
        "protection": (
            "enabled", "pair_loss_stop_enabled", "pair_loss_stop_return",
            "pair_loss_stop_freeze_model_lookback_multiplier",
            "wait_for_model_update_after_non_z_exit",
        ),
    }
    for pair in pairs[1:]:
        pipeline = _pipeline_for_pair(config, str(pair.get("market", "legacy")), pair)
        for stage, fields in compared_fields.items():
            current = pipeline.get(stage) or {}
            for field in fields:
                if current.get(field) != base[stage].get(field):
                    raise ValueError(
                        "portfolio optimization requires identical fixed core parameters; "
                        f"pair={pair.get('pair_id')} field={stage}.{field} "
                        f"value={current.get(field)!r} expected={base[stage].get(field)!r}"
                    )
    return base


def _select_best_eligible(rows):
    eligible = [row for row in rows if row.get("eligible") and not row.get("error")]
    return max(eligible, key=lambda row: row["selection_score"]) if eligible else None


def _candidate_config(config, pair, market, parameters, data_start_ts, measurement_start_ts, end_ts):
    result = copy.deepcopy(config)
    result.strategy.pairs = [copy.deepcopy(pair)]
    pipeline = _pipeline_for_pair(config, market, pair)
    for stage in ("estimator", "signal", "sizing", "protection"):
        if stage in parameters:
            pipeline.setdefault(stage, {}).update(parameters[stage])
    if getattr(result.strategy, "market_pipelines", None) and market in result.strategy.market_pipelines:
        result.strategy.market_pipelines[market]["pipeline"] = pipeline
    result.strategy.pipeline = pipeline
    result.data_start_ts = int(data_start_ts)
    result.measurement_start_ts = int(measurement_start_ts)
    result.backtest_start_ts = int(data_start_ts)
    result.backtest_end_ts = int(end_ts)
    result.backtest_end_exclusive = True
    return result


def _max_history_bars(candidates, pipeline):
    values = []
    for candidate in candidates:
        for stage in ("estimator", "signal", "sizing"):
            for key, value in (candidate.get(stage) or {}).items():
                if "lookback" in key or key in {"warmup_bars", "min_samples"}:
                    try:
                        values.append(int(float(value)))
                    except (TypeError, ValueError):
                        pass
    if not values:
        values = [int(float((pipeline.get("estimator") or {}).get("model_lookback_bars", 10080)))]
    return max(1, max(values))


def _timeframe_minutes(pipeline):
    text = str((pipeline.get("estimator") or {}).get("model_timeframe", "1m")).strip().lower()
    if text.endswith("m"):
        return max(1, int(text[:-1]))
    if text.endswith("h"):
        return max(1, int(text[:-1]) * 60)
    if text.endswith("d"):
        return max(1, int(text[:-1]) * 1440)
    return 1



def _training_metrics(metrics, curve, trades, target_capital, initial_equity, start_ts, end_ts, count):
    result = dict(metrics or {})
    count = max(1, int(count or 1))
    target_capital = max(float(target_capital), 1e-12)
    rows = [
        row for row in _rows(curve)
        if start_ts <= _timestamp_millis(row.get("ts")) < end_ts
    ]
    equities = [_number(row.get("equity"), None) for row in rows]
    equities = [value for value in equities if value is not None]
    final_equity = equities[-1] if equities else _number(result.get("final_equity"), initial_equity)
    net_pnl = final_equity - float(initial_equity)
    peak = float(initial_equity)
    max_drawdown_cash = 0.0
    for value in equities:
        peak = max(peak, value)
        max_drawdown_cash = min(max_drawdown_cash, value - peak)

    segment_start = [None] * count
    segment_end = [None] * count
    for row in rows:
        ts = _timestamp_millis(row.get("ts"))
        segment = min(count - 1, max(0, ((ts - start_ts) * count) // max(1, end_ts - start_ts)))
        equity = _number(row.get("equity"), None)
        if equity is None:
            continue
        if segment_start[segment] is None:
            segment_start[segment] = equity
        segment_end[segment] = equity

    active_segments = set()
    for trade in trades or []:
        action = getattr(trade, "action", None) if not isinstance(trade, dict) else trade.get("action")
        if str(getattr(action, "value", action)).lower() != "close":
            continue
        ts = _timestamp_millis(getattr(trade, "ts", None) if not isinstance(trade, dict) else trade.get("ts"))
        if start_ts <= ts < end_ts:
            active_segments.add(min(count - 1, max(0, ((ts - start_ts) * count) // max(1, end_ts - start_ts))))

    subperiod_returns = []
    profitable = 0
    for segment in range(count):
        first = segment_start[segment]
        last = segment_end[segment]
        if first is None or last is None:
            continue
        segment_return = (last - first) / target_capital
        subperiod_returns.append(segment_return)
        if segment in active_segments and segment_return > 0:
            profitable += 1
    if len(subperiod_returns) > 1:
        mean = sum(subperiod_returns) / len(subperiod_returns)
        subperiod_std = (sum((value - mean) ** 2 for value in subperiod_returns) / len(subperiod_returns)) ** 0.5
    else:
        subperiod_std = 0.0

    result.update({
        "net_pnl": net_pnl,
        "max_drawdown_cash": max_drawdown_cash,
        "return_on_target_capital": net_pnl / target_capital,
        "max_drawdown_on_target_capital": max_drawdown_cash / target_capital,
        "active_subperiod_count": len(active_segments),
        "profitable_subperiod_count": profitable,
        "subperiod_return_std": subperiod_std,
        "subperiod_count": count,
    })
    holding_minutes = _closed_holding_minutes(trades, start_ts, end_ts)
    result["holding_time_sum_hours"] = sum(holding_minutes) / 60.0
    result["holding_trade_count"] = len(holding_minutes)
    result["average_holding_time_hours"] = (
        sum(holding_minutes) / len(holding_minutes) / 60.0 if holding_minutes else 0.0
    )
    result.update(score_candidate_metrics(result))
    return result


def _closed_holding_minutes(trades, start_ts, end_ts):
    """Reconstruct position holding times from fill-level Trade events."""
    positions = {}
    for trade in trades or []:
        if isinstance(trade, dict):
            action = trade.get("action")
            timestamp = trade.get("ts")
            position_id = trade.get("position_id") or trade.get("group_id")
        else:
            action = getattr(trade, "action", None)
            timestamp = getattr(trade, "ts", None)
            position_id = getattr(trade, "position_id", None) or getattr(trade, "group_id", None)
        action = getattr(action, "value", action)
        if timestamp is None or position_id is None:
            continue
        try:
            timestamp = int(timestamp)
        except (TypeError, ValueError):
            continue
        state = positions.setdefault(str(position_id), {"open": None, "close": None})
        if str(action).lower() == "open":
            state["open"] = timestamp if state["open"] is None else min(state["open"], timestamp)
        elif str(action).lower() == "close":
            state["close"] = timestamp if state["close"] is None else max(state["close"], timestamp)

    result = []
    for state in positions.values():
        open_ts = state["open"]
        close_ts = state["close"]
        if (
            open_ts is not None
            and close_ts is not None
            and start_ts <= close_ts < end_ts
            and close_ts >= open_ts
        ):
            result.append((close_ts - open_ts) / 60000.0)
    return result


def _eligible(metrics, min_trades, max_drawdown, min_active_subperiods=2, subperiod_count=3):
    del min_trades
    active = int(_number(metrics.get("active_subperiod_count"), 0))
    required_active = min(max(1, int(subperiod_count)), max(0, int(min_active_subperiods)))
    dd = _number(metrics.get("max_drawdown_on_target_capital"), 0)
    return (
        active >= required_active
        and (max_drawdown is None or dd >= -abs(float(max_drawdown)))
    )


def _filter_curve(curve, start_ts):
    if isinstance(curve, dict):
        rows = [{"ts": ts, "equity": eq} for ts, eq in zip(curve.get("ts", []), curve.get("equity", [])) if ts >= start_ts]
        return rows
    return [row for row in curve if int(row.get("ts", 0)) >= start_ts]


def _pair_window_rows(result, schedule, config_path=None):
    """Build validation-window metrics from each pair's marked PnL curve."""
    try:
        from core.modules.reporting.curve_charts import (
            _collect_pair_trades,
            _config_pair_defs,
            _equity_series,
            _funding_events,
            _pair_points,
        )
        pair_defs, symbol_paths = _config_pair_defs(Path(config_path))
        pair_trades = {item.pair_id: item for item in _collect_pair_trades(result, pair_defs)}
        funding_events = _funding_events(result)
        equity_ts, _equity = _equity_series(result)
        timeline = (int(equity_ts[0]), int(equity_ts[-1])) if len(equity_ts) else None
        target_by_pair = {
            str(pair.get("pair_id")): float(pair.get("target_capital") or 0.0)
            for pair in load_config(config_path).strategy.pairs
        }
    except Exception as exc:
        logger.warning("pair window curve preparation failed: {}", exc)
        pair_trades, symbol_paths, funding_events, timeline, target_by_pair = {}, {}, [], None, {}

    all_trades = list(result.trades or [])
    all_funding = list(result.funding_payments or [])
    points_by_pair = {}
    for pair_id, trades in pair_trades.items():
        try:
            points_by_pair[pair_id] = _pair_points(
                trades, funding_events, symbol_paths, timeline, 15 * 60_000,
            )
        except Exception as exc:
            logger.warning("pair window curve failed pair={} reason={}", pair_id, exc)
            points_by_pair[pair_id] = None

    rows = []
    for pair_id, entries in schedule.items():
        trades = pair_trades.get(pair_id)
        points = points_by_pair.get(pair_id)
        for entry in entries:
            start = _date_ts(entry["validation_start"])
            end = _date_ts(entry["validation_end"])
            window_trades = [
                trade for trade in all_trades
                if getattr(trade, "pair_id", None) == pair_id
                and start <= int(getattr(trade, "ts", 0) or 0) < end
            ]
            window_funding = [
                payment for payment in all_funding
                if getattr(payment, "pair_id", None) == pair_id
                and start <= int(getattr(payment, "ts", 0) or 0) < end
            ]
            curve_metrics = _validation_pair_curve_metrics(
                points, start, end, target_by_pair.get(pair_id, 0.0)
            )
            closed = [] if trades is None else [
                record for record in trades.positions.values()
                if record.get("close_ts") is not None
                and start <= int(record["close_ts"]) < end
            ]
            wins = sum(1 for record in closed if float(record.get("cashflow", 0.0)) > 0.0)
            rows.append({
                "pair_id": pair_id,
                "validation_start": entry["validation_start"],
                "validation_end": entry["validation_end"],
                "train_metrics": json.dumps(entry.get("train_metrics", {}), ensure_ascii=False),
                "validation_return": curve_metrics["return"],
                "validation_net_pnl": curve_metrics["net_pnl"],
                "validation_sharpe": curve_metrics["sharpe"],
                "validation_max_drawdown": curve_metrics["max_drawdown"],
                "validation_win_rate": wins / len(closed) if closed else 0.0,
                "validation_trade_count": len(closed),
                "validation_fill_count": len(window_trades),
                "validation_fee": sum(float(getattr(trade, "fee", 0.0) or 0.0) for trade in window_trades),
                "validation_slippage": sum(float(getattr(trade, "slippage", 0.0) or 0.0) for trade in window_trades),
                "validation_funding_fee": sum(float(getattr(payment, "payment", 0.0) or 0.0) for payment in window_funding),
                "selection_score": entry.get("selection_score"),
                "fallback": entry.get("fallback"),
                "skipped": entry.get("skipped"),
            })
    return rows


def _validation_pair_curve_metrics(points, start, end, target_capital):
    if not points or len(points.get("ts", [])) == 0:
        return {"return": 0.0, "net_pnl": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    timestamps = np.asarray(points["ts"], dtype=np.int64)
    pnl = np.asarray(points["pnl"], dtype=np.float64)
    before = int(np.searchsorted(timestamps, start, side="right")) - 1
    finish = int(np.searchsorted(timestamps, end, side="left")) - 1
    baseline = float(pnl[before]) if before >= 0 else 0.0
    if finish < 0:
        return {"return": 0.0, "net_pnl": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    indices = np.flatnonzero((timestamps >= start) & (timestamps < end))
    values = pnl[indices] - baseline if indices.size else np.asarray([pnl[finish] - baseline])
    capital = max(float(target_capital or 0.0), 1e-12)
    equity = capital + values
    peak = np.maximum.accumulate(equity)
    drawdown = equity / np.maximum(peak, 1e-12) - 1.0
    daily = {}
    source_ts = timestamps[indices] if indices.size else np.asarray([timestamps[finish]])
    for timestamp, value in zip(source_ts, equity):
        daily[int(timestamp) // 86_400_000] = float(value)
    daily_values = np.asarray(list(daily.values()), dtype=np.float64)
    sharpe = 0.0
    if daily_values.size > 2:
        returns = np.diff(daily_values) / np.maximum(daily_values[:-1], 1e-12)
        returns = returns[np.isfinite(returns)]
        if returns.size > 1 and float(np.std(returns, ddof=1)) > 1e-12:
            sharpe = float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252.0))
    net = float(pnl[finish] - baseline)
    return {
        "return": net / capital,
        "net_pnl": net,
        "sharpe": sharpe,
        "max_drawdown": float(np.min(drawdown)) if drawdown.size else 0.0,
    }


def _schedule_rows(schedule):
    rows=[]
    for pair_id, entries in schedule.items():
        for entry in entries:
            rows.append({
                "pair_id": pair_id,
                "train_start": entry["train_start"],
                "train_end": entry["train_end"],
                "validation_start": entry["validation_start"],
                "validation_end": entry["validation_end"],
                "parameters": json.dumps(entry.get("parameters", {}), ensure_ascii=False),
                "train_metrics": json.dumps(entry.get("train_metrics", {}), ensure_ascii=False),
                "selection_score": entry.get("selection_score"),
                "selection_reason": entry.get("selection_reason"),
                "fallback": entry.get("fallback"),
                "skipped": entry.get("skipped"),
                "optimization_scope": entry.get("optimization_scope", "pair"),
                "global_candidate_index": entry.get("global_candidate_index"),
                "active_pair_count": entry.get("active_pair_count"),
                "skipped_pair_count": entry.get("skipped_pair_count"),
                "active_pair_ratio": entry.get("active_pair_ratio"),
            })
    return rows


def _flatten_candidate_rows(rows):
    fields = (
        "trade_count", "win_rate", "sharpe", "max_drawdown",
        "return_on_target_capital", "max_drawdown_on_target_capital",
        "return_score", "sharpe_score", "calmar", "calmar_score",
        "average_holding_time_hours", "holding_time_score", "trade_count_score",
        "win_rate_score", "drawdown_score",
        "subperiod_score", "active_subperiod_count", "profitable_subperiod_count",
        "subperiod_return_std",
    )
    flattened = []
    for row in rows:
        metrics = row.get("metrics") or {}
        item = {
            "pair_id": row["pair_id"], "train_start": row["train_start"],
            "train_end": row["train_end"], "candidate_index": row["candidate_index"],
            "parameters": json.dumps(row["parameters"], ensure_ascii=False),
            "metrics": json.dumps(metrics, ensure_ascii=False), "own_score": row.get("own_score"),
            "neighbor_mean_score": row.get("neighbor_mean_score"),
            "neighbor_score_std": row.get("neighbor_score_std"),
            "neighbor_count": row.get("neighbor_count"),
            "valid_neighbor_count": row.get("valid_neighbor_count"),
            "final_selection_score": row.get("final_selection_score", row.get("selection_score")),
            "selection_score": row["selection_score"],
            "selection_reason": row.get("selection_reason", ""),
            "eligible": row["eligible"], "error": row["error"],
            "base_eligible": row.get("base_eligible", row.get("eligible")),
            "excluded_top_fraction": row.get("excluded_top_fraction", False),
            "own_neighbor_gap": row.get("own_neighbor_gap"),
            "selection_filter_reason": row.get("selection_filter_reason", ""),
        }
        for field in fields:
            item[field] = metrics.get(field)
        if row.get("optimization_scope") == "portfolio" or row.get("pair_id") == "__portfolio__":
            item.update({
                "portfolio_initial_equity": metrics.get("initial_equity"),
                "portfolio_final_equity": metrics.get("final_equity"),
                "portfolio_return": metrics.get("total_return"),
                "portfolio_annualized_return": metrics.get("annualized_return"),
                "portfolio_sharpe": metrics.get("sharpe"),
                "portfolio_win_rate": metrics.get("win_rate"),
                "portfolio_max_drawdown": metrics.get("max_drawdown"),
                "portfolio_trade_count": metrics.get("trade_count"),
                "portfolio_average_holding_time_hours": metrics.get("average_holding_time_hours"),
                "portfolio_fee": metrics.get("total_fee"),
                "portfolio_slippage": metrics.get("total_slippage"),
                "portfolio_funding": metrics.get("funding_fee"),
            })
        flattened.append(item)
    return flattened


def _portfolio_pair_rows(rows):
    """Flatten optional pair diagnostics emitted by the portfolio optimizer."""
    output = []
    for row in rows:
        if row.get("pair_id") == "__portfolio__":
            continue
        metrics = row.get("metrics") or {}
        output.append({
            "pair_id": row.get("pair_id"),
            "candidate_index": row.get("candidate_index"),
            "train_start": row.get("train_start"),
            "train_end": row.get("train_end"),
            "metrics": json.dumps(metrics, ensure_ascii=False),
            "pair_return": metrics.get("return_on_target_capital"),
            "pair_trade_count": metrics.get("trade_count"),
            "pair_win_rate": metrics.get("win_rate"),
            "pair_average_holding_time_hours": metrics.get("average_holding_time_hours"),
            "pair_max_drawdown": metrics.get("max_drawdown_on_target_capital"),
        })
    return output


def _rows(curve):
    if isinstance(curve, dict):
        return [{"ts": ts, "equity": equity} for ts, equity in zip(curve.get("ts", []), curve.get("equity", []))]
    return list(curve or [])


def _object_rows(items):
    import dataclasses
    rows=[]
    for item in items or []:
        if dataclasses.is_dataclass(item): rows.append(dataclasses.asdict(item))
        elif isinstance(item, dict): rows.append(item)
        else: rows.append({"value": str(item)})
    return rows



def _dataclass_dict(value):
    import dataclasses
    return dataclasses.asdict(value) if dataclasses.is_dataclass(value) else value


def _write_yaml(path, value):
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _write_csv(path, rows):
    rows=list(rows or [])
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _append_csv(path, rows):
    """Append stable-schema rows without retaining prior results in memory."""
    rows = list(rows or [])
    if not rows:
        return
    path = Path(path)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open("r", newline="", encoding="utf-8") as handle:
            fields = next(csv.reader(handle), [])
    else:
        fields = list(rows[0].keys())
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _as_choices(value):
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _number(value, default=0.0):
    try: return float(value) if value is not None else default
    except (TypeError, ValueError): return default


def _timestamp_millis(value):
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return int(value.timestamp() * 1000)
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _candidate_id(candidate):
    if not candidate:
        return "none"
    return candidate.get("candidate_index", "unknown")


def _timestamp_text(value):
    return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _as_date(value):
    if isinstance(value, datetime): return value.date()
    if isinstance(value, date): return value
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def _add_months(value, months):
    index=value.year*12+(value.month-1)+int(months)
    year, month=divmod(index,12); month+=1
    import calendar
    return date(year, month, min(value.day, calendar.monthrange(year, month)[1]))


def _date_ts(value):
    dt=datetime.combine(_as_date(value), datetime.min.time(), tzinfo=timezone.utc)
    return int(dt.timestamp()*1000)


def _date_from_ts(value):
    if value is None: return None
    return datetime.fromtimestamp(int(value)/1000, tz=timezone.utc).date().isoformat()
