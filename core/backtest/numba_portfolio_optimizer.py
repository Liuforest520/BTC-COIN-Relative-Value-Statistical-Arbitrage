"""Shared-parameter optimization on a synchronized portfolio account."""
from __future__ import annotations

import math
from loguru import logger

from core.backtest.numba_pair_optimizer import (
    _slice_grid,
    add_neighbor_scores,
    build_candidate_grid,
    score_candidate_metrics,
)
from core.backtest.numba_shared_portfolio import run_shared_portfolio_batch
from core.backtest.portfolio_data_stream import PortfolioDataStream


def run_numba_portfolio_optimizer(
    config,
    pairs,
    window,
    candidates,
    warmup_start_ts,
    max_bars,
    wf,
    progress_callback=None,
):
    """Run every candidate against all pairs on one shared account.

    Pair model/signal/position state is isolated, while cash, equity, margin,
    drawdown and selection metrics are candidate-level portfolio state.
    """
    grid = build_candidate_grid(candidates)
    stream = PortfolioDataStream(
        config,
        pairs,
        warmup_start_ts,
        window.train_end_ts,
        measurement_start_ts=window.train_start_ts,
        warmup_bars=max_bars if wf.warmup_enabled else 0,
    )
    prepared = stream.prepare()
    active_pairs = prepared.pair_count
    skipped_pairs = prepared.skipped
    actual_warmup_start_ts = int(prepared.timestamps[0])
    if active_pairs < int(getattr(wf, "portfolio_min_active_pairs", 1)):
        prepared.close()
        return _skipped(
            active_pairs, skipped_pairs, actual_warmup_start_ts, max_bars,
            f"active pair count {active_pairs} is below minimum {wf.portfolio_min_active_pairs}",
        )
    if getattr(wf, "portfolio_require_all_pairs", False) and skipped_pairs:
        prepared.close()
        return _skipped(
            active_pairs, skipped_pairs, actual_warmup_start_ts, max_bars,
            f"require_all_pairs enabled; skipped pairs: {len(skipped_pairs)}",
        )

    rows = []
    pair_metrics = []
    initial_equity = float(config.initial_cash)
    max_leverage = float((config.risk or {}).get("max_leverage", 1.0) or 1.0)
    batch_size = max(1, int(getattr(wf, "portfolio_candidate_batch_size", 16) or 16))
    logger.info(
        "portfolio synchronized data ready pairs={} skipped={} timestamps={} rows={} candidates={} batch_size={}",
        active_pairs, len(skipped_pairs), len(prepared.timestamps),
        int(prepared.offsets[-1]), grid.size, batch_size,
    )
    try:
        for batch_start in range(0, grid.size, batch_size):
            batch_end = min(grid.size, batch_start + batch_size)
            batch = _slice_grid(grid, batch_start, batch_end)
            result = run_shared_portfolio_batch(
                prepared.timestamps,
                prepared.bar_indices,
                prepared.values,
                prepared.offsets,
                prepared.target_capital,
                prepared.hedge_beta_lookback_bars,
                prepared.hedge_beta_min_samples,
                prepared.reversion_min_samples,
                prepared.stop_freeze_lookback_multiplier,
                prepared.wait_for_model_update_after_stop,
                int(window.train_start_ts),
                int(window.train_end_ts),
                int(wf.subperiod_count),
                initial_equity,
                float(config.fee_rate),
                float(config.slippage_bps) / 10000.0,
                max_leverage,
                batch.regression_method_code,
                batch.model_lookback_bars,
                batch.model_update_interval_bars,
                batch.residual_adf_max_pvalue,
                batch.entry_z,
                batch.exit_z,
                batch.reversion_ma_lookback_bars,
                batch.reversion_ma_short_lookback_bars,
                batch.pair_loss_stop_enabled,
                batch.pair_loss_stop_return,
            )
            for local_index, parameters in enumerate(batch.parameters):
                candidate_index = batch_start + local_index
                metrics = _portfolio_metrics(result, local_index, initial_equity, wf.subperiod_count)
                components = score_candidate_metrics(metrics)
                metrics.update(components)
                row = {
                    "pair_id": "__portfolio__",
                    "optimization_scope": "portfolio",
                    "train_start": window.train_start,
                    "train_end": window.train_end,
                    "candidate_index": candidate_index,
                    "parameters": parameters,
                    "metrics": metrics,
                    "own_score": components["own_score"],
                    "selection_score": components["own_score"],
                    "eligible": _eligible(metrics, wf),
                    "active_pair_count": active_pairs,
                    "skipped_pair_count": len(skipped_pairs),
                    "active_pair_ratio": active_pairs / max(1, len(pairs)),
                    "error": "",
                }
                rows.append(row)
                pair_metrics.extend(_pair_rows(
                    prepared, result, local_index, candidate_index,
                    window.train_start, window.train_end,
                ))
            logger.info(
                "portfolio candidate progress train=[{}, {}) candidates={}/{} ({:.1f}%) active_pairs={} skipped_pairs={}",
                window.train_start, window.train_end, batch_end, grid.size,
                100.0 * batch_end / max(1, grid.size), active_pairs, len(skipped_pairs),
            )
            if progress_callback:
                progress_callback(batch_end, grid.size, active_pairs, len(skipped_pairs))
    finally:
        prepared.close()

    add_neighbor_scores(rows, grid)
    for row in rows:
        row["metrics"]["selection_score"] = row["selection_score"]
    filter_summary = _apply_overfit_filters(rows, wf)
    eligible = [row for row in rows if row["eligible"]]
    selected = max(eligible, key=lambda item: item["selection_score"]) if eligible else None
    return {
        "selected": selected,
        "candidates": rows,
        "pair_metrics": pair_metrics,
        "active_pair_count": active_pairs,
        "skipped_pairs": skipped_pairs,
        "skipped": False,
        "selection_filter_summary": filter_summary,
        "warmup_start_ts": actual_warmup_start_ts,
        "max_warmup_bars": max_bars,
    }


def _portfolio_metrics(result, index, initial, subperiod_count):
    return {
        "initial_equity": initial,
        "final_equity": float(result[0][index]),
        "total_return": float(result[1][index]),
        "annualized_return": float(result[2][index]),
        "sharpe": float(result[3][index]),
        "max_drawdown": float(result[4][index]),
        "trade_count": int(result[5][index]),
        "win_rate": float(result[6][index]),
        "total_fee": float(result[7][index]),
        "total_slippage": float(result[8][index]),
        "funding_fee": float(result[9][index]),
        "net_pnl": float(result[10][index]),
        "max_drawdown_cash": float(result[11][index]),
        "return_on_target_capital": float(result[10][index]) / max(initial, 1e-12),
        "max_drawdown_on_target_capital": float(result[11][index]) / max(initial, 1e-12),
        "active_subperiod_count": int(result[12][index]),
        "profitable_subperiod_count": int(result[13][index]),
        "subperiod_return_std": float(result[14][index]),
        "subperiod_count": int(subperiod_count),
        "rejected_open_count": int(result[23][index]),
    }


def _pair_rows(prepared, result, local_index, candidate_index, train_start, train_end):
    output = []
    for pair_index, pair_id in enumerate(prepared.pair_ids):
        trades = int(result[18][local_index, pair_index])
        target = max(float(prepared.target_capital[pair_index]), 1e-12)
        net = float(result[17][local_index, pair_index])
        output.append({
            "pair_id": pair_id,
            "candidate_index": candidate_index,
            "train_start": train_start,
            "train_end": train_end,
            "metrics": {
                "net_pnl": net,
                "return_on_target_capital": net / target,
                "trade_count": trades,
                "win_rate": float(result[19][local_index, pair_index]) / trades if trades else 0.0,
                "max_drawdown_cash": float(result[20][local_index, pair_index]),
                "max_drawdown_on_target_capital": float(result[20][local_index, pair_index]) / target,
                "total_fee": float(result[21][local_index, pair_index]),
                "funding_fee": float(result[22][local_index, pair_index]),
            },
        })
    return output


def _skipped(active, skipped, warmup_start, max_bars, reason):
    return {
        "selected": None,
        "candidates": [],
        "pair_metrics": [],
        "active_pair_count": active,
        "skipped_pairs": skipped,
        "skipped": True,
        "skip_reason": reason,
        "warmup_start_ts": warmup_start,
        "max_warmup_bars": max_bars,
    }


def _apply_overfit_filters(rows, wf):
    """Reject extreme winners and unsupported parameter islands."""
    base = [row for row in rows if row.get("eligible") and not row.get("error")]
    fraction = float(getattr(wf, "exclude_top_fraction", 0.0) or 0.0)
    exclude_count = int(math.ceil(len(base) * fraction)) if fraction > 0.0 else 0
    exclude_count = min(exclude_count, max(0, len(base) - 1))
    ranked = sorted(base, key=lambda row: float(row.get("selection_score", -math.inf)), reverse=True)
    excluded_ids = {id(row) for row in ranked[:exclude_count]}
    min_neighbors = max(0, int(getattr(wf, "min_valid_neighbors", 0) or 0))
    max_gap = getattr(wf, "max_own_neighbor_gap", None)
    max_gap = None if max_gap is None else float(max_gap)
    insufficient = excessive_gap = selected = 0

    for row in rows:
        base_eligible = bool(row.get("eligible") and not row.get("error"))
        row["base_eligible"] = base_eligible
        row["excluded_top_fraction"] = False
        row["own_neighbor_gap"] = None
        reasons = []
        if not base_eligible:
            reasons.append("failed hard metric eligibility")
        elif id(row) in excluded_ids:
            row["excluded_top_fraction"] = True
            reasons.append(f"excluded from top {fraction:.2%} of portfolio scores")
        valid_neighbors = int(row.get("valid_neighbor_count", 0) or 0)
        if base_eligible and valid_neighbors < min_neighbors:
            insufficient += 1
            reasons.append(f"valid neighbors {valid_neighbors} < {min_neighbors}")
        if base_eligible and valid_neighbors > 0:
            gap = float(row.get("own_score", 0.0)) - float(row.get("neighbor_mean_score", 0.0))
            row["own_neighbor_gap"] = gap
            if max_gap is not None and gap > max_gap:
                excessive_gap += 1
                reasons.append(f"own-neighbor score gap {gap:.6f} > {max_gap:.6f}")
        row["eligible"] = base_eligible and not reasons
        row["selection_filter_reason"] = "; ".join(reasons)
        if row["eligible"]:
            selected += 1

    summary = {
        "base_eligible_count": len(base),
        "excluded_top_count": exclude_count,
        "insufficient_neighbor_count": insufficient,
        "excessive_neighbor_gap_count": excessive_gap,
        "selection_eligible_count": selected,
        "exclude_top_fraction": fraction,
        "min_valid_neighbors": min_neighbors,
        "max_own_neighbor_gap": max_gap,
    }
    logger.info(
        "portfolio overfit filters base_eligible={} excluded_top={} insufficient_neighbors={} excessive_neighbor_gap={} selection_eligible={}",
        len(base), exclude_count, insufficient, excessive_gap, selected,
    )
    return summary


def _eligible(metrics, wf):
    required_active = min(
        max(1, int(metrics.get("subperiod_count", 1) or 1)),
        max(0, int(wf.min_active_subperiods)),
    )
    return (
        int(metrics.get("trade_count", 0)) >= int(wf.min_trades)
        and int(metrics.get("active_subperiod_count", 0)) >= required_active
        and (
            wf.max_drawdown is None
            or metrics.get("max_drawdown_on_target_capital", 0.0) >= -abs(float(wf.max_drawdown))
        )
    )


__all__ = ["run_numba_portfolio_optimizer", "_apply_overfit_filters"]
