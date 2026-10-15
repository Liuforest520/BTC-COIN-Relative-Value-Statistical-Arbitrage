"""Batch Numba optimizer for the cleaned single-pair TLS strategy."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from core.backtest.pair_data_cache import PairDataCache
from core.modules.estimator.adf_numba import adf_pvalue_c_lag1


@dataclass(frozen=True)
class CandidateGrid:
    parameters: list[dict]
    regression_method_code: np.ndarray
    model_lookback_bars: np.ndarray
    model_update_interval_bars: np.ndarray
    residual_adf_max_pvalue: np.ndarray
    entry_z: np.ndarray
    exit_z: np.ndarray
    reversion_ma_lookback_bars: np.ndarray
    reversion_ma_short_lookback_bars: np.ndarray
    pair_loss_stop_enabled: np.ndarray
    pair_loss_stop_return: np.ndarray

    @property
    def size(self):
        return len(self.parameters)


def build_candidate_grid(candidates: list[dict]) -> CandidateGrid:
    """Convert nested YAML patches into compact numeric candidate arrays."""
    clean = []
    for patch in candidates:
        estimator = dict(patch.get("estimator") or {})
        signal = dict(patch.get("signal") or {})
        protection = dict(patch.get("protection") or {})
        method = str(estimator.get("regression_method", "log_price")).lower()
        if method not in {"price", "log_price"}:
            raise ValueError(f"Numba TLS optimizer only supports price/log_price, got {method!r}")
        long_ma = int(signal.get("reversion_ma_lookback_bars", 60))
        short_ma = int(signal.get("reversion_ma_short_lookback_bars", 15))
        lookback = int(estimator.get("model_lookback_bars", 10080))
        update_interval = int(estimator.get("model_update_interval_bars", 240))
        if lookback < 5 * update_interval:
            continue
        if short_ma > long_ma:
            continue
        clean.append(patch)
    if not clean:
        raise ValueError("parameter grid produced no legal TLS candidates")

    def values(stage, key, default, caster):
        return np.ascontiguousarray([caster((p.get(stage) or {}).get(key, default)) for p in clean])

    return CandidateGrid(
        parameters=clean,
        regression_method_code=np.ascontiguousarray([
            0 if str((p.get("estimator") or {}).get("regression_method", "log_price")).lower() == "price" else 1
            for p in clean
        ], dtype=np.int64),
        model_lookback_bars=values("estimator", "model_lookback_bars", 10080, lambda v: max(10, int(v))),
        model_update_interval_bars=values("estimator", "model_update_interval_bars", 240, lambda v: max(1, int(v))),
        residual_adf_max_pvalue=values("estimator", "residual_adf_max_pvalue", 0.4, float),
        entry_z=values("signal", "entry_z", 2.0, float),
        exit_z=values("signal", "exit_z", 0.5, float),
        reversion_ma_lookback_bars=values("signal", "reversion_ma_lookback_bars", 60, lambda v: max(5, int(v))),
        reversion_ma_short_lookback_bars=values("signal", "reversion_ma_short_lookback_bars", 15, lambda v: max(1, int(v))),
        pair_loss_stop_enabled=values("protection", "pair_loss_stop_enabled", False, _as_bool),
        pair_loss_stop_return=values("protection", "pair_loss_stop_return", 0.05, float),
    )


def run_numba_pair_optimizer(
    data: PairDataCache,
    grid: CandidateGrid,
    train_start_ts: int,
    train_end_ts: int,
    initial_equity: float,
    target_capital: float,
    fee_rate: float,
    slippage_bps: float,
    hedge_beta_lookback_bars: int = 1440,
    hedge_beta_min_samples: int = 30,
    reversion_min_samples: int | None = None,
    pair_loss_stop_freeze_model_lookback_multiplier: float = 0.0,
    wait_for_model_update_after_non_z_exit: bool = False,
    min_trades: int = 20,
    max_drawdown: float | None = None,
    subperiod_count: int = 3,
    min_active_subperiods: int = 2,
    batch_size: int = 64,
    progress_callback=None,
    collect_daily_equity: bool = False,
) -> list[dict]:
    if data.size == 0:
        raise ValueError("cannot optimize an empty pair cache")
    start = int(np.searchsorted(data.timestamps, int(train_start_ts), side="left"))
    end = int(np.searchsorted(data.timestamps, int(train_end_ts), side="left"))
    if end <= start:
        raise ValueError("training window has no aligned bars")
    rows = []
    daily_batches = []
    daily_start_day = None
    batch_size = max(1, int(batch_size))
    subperiod_count = max(1, int(subperiod_count))
    min_active_subperiods = min(subperiod_count, max(0, int(min_active_subperiods)))
    for batch_start in range(0, grid.size, batch_size):
        batch_end = min(grid.size, batch_start + batch_size)
        batch = _slice_grid(grid, batch_start, batch_end)
        result = _run_batch(
            data.timestamps, data.x_open, data.x_high, data.x_low, data.x_close,
            data.y_open, data.y_high, data.y_low, data.y_close,
            data.funding_x, data.funding_y, data.funding_x_event, data.funding_y_event,
            start, end, int(train_start_ts), int(train_end_ts), subperiod_count,
            float(initial_equity), float(target_capital), float(fee_rate),
            float(slippage_bps) / 10000.0, int(hedge_beta_lookback_bars), int(hedge_beta_min_samples),
            int(reversion_min_samples or 0),
            float(pair_loss_stop_freeze_model_lookback_multiplier or 0.0),
            bool(wait_for_model_update_after_non_z_exit),
            batch.regression_method_code, batch.model_lookback_bars, batch.model_update_interval_bars,
            batch.residual_adf_max_pvalue, batch.entry_z, batch.exit_z,
            batch.reversion_ma_lookback_bars, batch.reversion_ma_short_lookback_bars,
            batch.pair_loss_stop_enabled, batch.pair_loss_stop_return,
        )
        if collect_daily_equity:
            daily_batches.append(result[18])
            daily_start_day = int(result[19])
        for local_index, params in enumerate(batch.parameters):
            metrics = {
                "initial_equity": float(result[0][local_index]),
                "final_equity": float(result[1][local_index]),
                "total_return": float(result[2][local_index]),
                "annualized_return": float(result[3][local_index]),
                "sharpe": float(result[4][local_index]),
                "max_drawdown": float(result[5][local_index]),
                "trade_count": int(result[6][local_index]),
                "win_rate": float(result[7][local_index]),
                "total_fee": float(result[8][local_index]),
                "total_slippage": float(result[9][local_index]),
                "funding_fee": float(result[10][local_index]),
                "net_pnl": float(result[11][local_index]),
                "max_drawdown_cash": float(result[12][local_index]),
                "return_on_target_capital": float(result[13][local_index]),
                "max_drawdown_on_target_capital": float(result[14][local_index]),
                "active_subperiod_count": int(result[15][local_index]),
                "profitable_subperiod_count": int(result[16][local_index]),
                "subperiod_return_std": float(result[17][local_index]),
                "subperiod_count": subperiod_count,
                "selection_score": 0.0,
            }
            components = score_candidate_metrics(metrics)
            own_score = components["own_score"]
            metrics.update(components)
            rows.append({
                "candidate_index": batch_start + local_index,
                "parameters": params,
                "metrics": metrics,
                "own_score": own_score,
                "selection_score": own_score,
                "eligible": _eligible(
                    metrics,
                    min_trades,
                    max_drawdown,
                    min_active_subperiods,
                    subperiod_count,
                ),
                "error": "",
            })
        if progress_callback is not None:
            progress_callback(batch_end, grid.size)
    _apply_neighbor_scores(rows, grid)
    for row in rows:
        row["metrics"]["selection_score"] = row["selection_score"]
    if collect_daily_equity:
        daily_equity = np.ascontiguousarray(np.vstack(daily_batches), dtype=np.float64)
        return rows, daily_equity, daily_start_day
    return rows


def _slice_grid(grid: CandidateGrid, start: int, end: int) -> CandidateGrid:
    """Return a compact candidate batch without changing candidate ordering."""
    return CandidateGrid(
        parameters=grid.parameters[start:end],
        regression_method_code=np.ascontiguousarray(grid.regression_method_code[start:end]),
        model_lookback_bars=np.ascontiguousarray(grid.model_lookback_bars[start:end]),
        model_update_interval_bars=np.ascontiguousarray(grid.model_update_interval_bars[start:end]),
        residual_adf_max_pvalue=np.ascontiguousarray(grid.residual_adf_max_pvalue[start:end]),
        entry_z=np.ascontiguousarray(grid.entry_z[start:end]),
        exit_z=np.ascontiguousarray(grid.exit_z[start:end]),
        reversion_ma_lookback_bars=np.ascontiguousarray(grid.reversion_ma_lookback_bars[start:end]),
        reversion_ma_short_lookback_bars=np.ascontiguousarray(grid.reversion_ma_short_lookback_bars[start:end]),
        pair_loss_stop_enabled=np.ascontiguousarray(grid.pair_loss_stop_enabled[start:end]),
        pair_loss_stop_return=np.ascontiguousarray(grid.pair_loss_stop_return[start:end]),
    )


@njit(cache=True)
def _run_batch(
    timestamps, x_open, x_high, x_low, x_close, y_open, y_high, y_low, y_close,
    funding_x, funding_y, funding_x_event, funding_y_event,
    start, end, train_start_ts, train_end_ts, subperiod_count,
    initial_equity, target_capital, fee_rate, slippage,
    hedge_beta_lookback_bars, hedge_beta_min_samples,
    configured_reversion_min_samples, stop_freeze_lookback_multiplier,
    wait_for_model_update_after_non_z_exit,
    regression_method_code, model_lookback, update_interval, adf_threshold,
    entry_z, exit_z, ma_long, ma_short, stop_enabled, stop_return,
):
    n_candidates = regression_method_code.size
    initial = np.full(n_candidates, initial_equity, dtype=np.float64)
    cash_pnl = np.zeros(n_candidates, dtype=np.float64)
    equity = np.full(n_candidates, initial_equity, dtype=np.float64)
    previous_equity = np.full(n_candidates, initial_equity, dtype=np.float64)
    peak = np.full(n_candidates, initial_equity, dtype=np.float64)
    max_dd = np.zeros(n_candidates, dtype=np.float64)
    max_dd_cash = np.zeros(n_candidates, dtype=np.float64)
    alpha = np.full(n_candidates, np.nan, dtype=np.float64)
    beta = np.full(n_candidates, np.nan, dtype=np.float64)
    spread_mean = np.full(n_candidates, np.nan, dtype=np.float64)
    spread_std = np.full(n_candidates, np.nan, dtype=np.float64)
    adf_p = np.full(n_candidates, 1.0, dtype=np.float64)
    next_update = np.full(n_candidates, -1, dtype=np.int64)
    position = np.zeros(n_candidates, dtype=np.int64)
    qx = np.zeros(n_candidates, dtype=np.float64)
    qy = np.zeros(n_candidates, dtype=np.float64)
    entry_x = np.zeros(n_candidates, dtype=np.float64)
    entry_y = np.zeros(n_candidates, dtype=np.float64)
    entry_gross = np.zeros(n_candidates, dtype=np.float64)
    entry_fee = np.zeros(n_candidates, dtype=np.float64)
    position_funding = np.zeros(n_candidates, dtype=np.float64)
    pending_action = np.zeros(n_candidates, dtype=np.int64)
    pending_side = np.zeros(n_candidates, dtype=np.int64)
    pending_qx = np.zeros(n_candidates, dtype=np.float64)
    pending_qy = np.zeros(n_candidates, dtype=np.float64)
    pending_stop_close = np.zeros(n_candidates, dtype=np.int64)
    freeze_until = np.full(n_candidates, -1, dtype=np.int64)
    reopen_lock_pending = np.zeros(n_candidates, dtype=np.int64)
    stop_close_bar = np.full(n_candidates, -1, dtype=np.int64)
    trades = np.zeros(n_candidates, dtype=np.int64)
    wins = np.zeros(n_candidates, dtype=np.int64)
    fees = np.zeros(n_candidates, dtype=np.float64)
    slippages = np.zeros(n_candidates, dtype=np.float64)
    funding_total = np.zeros(n_candidates, dtype=np.float64)
    daily_base = np.full(n_candidates, initial_equity, dtype=np.float64)
    daily_last = np.full(n_candidates, initial_equity, dtype=np.float64)
    daily_sum = np.zeros(n_candidates, dtype=np.float64)
    daily_sum2 = np.zeros(n_candidates, dtype=np.float64)
    daily_count = np.zeros(n_candidates, dtype=np.int64)
    start_day_key = int(timestamps[start] // 86400000)
    end_day_key = int(timestamps[end - 1] // 86400000)
    daily_equity = np.full((n_candidates, max(1, end_day_key - start_day_key + 1)), np.nan, dtype=np.float64)
    subperiod_start_equity = np.full((n_candidates, subperiod_count), np.nan, dtype=np.float64)
    subperiod_end_equity = np.full((n_candidates, subperiod_count), np.nan, dtype=np.float64)
    subperiod_trades = np.zeros((n_candidates, subperiod_count), dtype=np.int64)
    current_day = -1
    max_ma = 5
    min_samples = np.empty(n_candidates, dtype=np.int64)
    for c in range(n_candidates):
        required = ma_long[c]
        if configured_reversion_min_samples > required:
            required = configured_reversion_min_samples
        min_samples[c] = required
        if required > max_ma:
            max_ma = required
    z_hist = np.zeros((n_candidates, max_ma), dtype=np.float64)
    z_count = np.zeros(n_candidates, dtype=np.int64)
    z_pos = np.zeros(n_candidates, dtype=np.int64)
    max_lookback = 10
    for value in model_lookback:
        if value > max_lookback:
            max_lookback = value
    residual_buf = np.empty(max_lookback + 4, dtype=np.float64)

    # Warmup bars update estimator and signal state but never create orders or
    # enter training metrics.  This mirrors the complete Backtest boundary.
    for i in range(0, end):
        duration = max(1, train_end_ts - train_start_ts)
        segment = int(((timestamps[i] - train_start_ts) * subperiod_count) // duration)
        if segment < 0:
            segment = 0
        elif segment >= subperiod_count:
            segment = subperiod_count - 1
        if i >= start:
            for c in range(n_candidates):
                if not math.isfinite(subperiod_start_equity[c, segment]):
                    subperiod_start_equity[c, segment] = previous_equity[c]
        # Funding and all pending orders happen at the current executable open.
        for c in range(n_candidates):
            if position[c] != 0:
                if funding_x_event[i]:
                    payment = qx[c] * x_open[i] * funding_x[i]
                    cash_pnl[c] -= payment
                    position_funding[c] += payment
                    funding_total[c] += payment
                if funding_y_event[i]:
                    payment = qy[c] * y_open[i] * funding_y[i]
                    cash_pnl[c] -= payment
                    position_funding[c] += payment
                    funding_total[c] += payment
            action = pending_action[c]
            if action == 1 and position[c] == 0:
                side = pending_side[c]
                fill_x = x_open[i] * (1.0 + slippage if pending_qx[c] > 0 else 1.0 - slippage)
                fill_y = y_open[i] * (1.0 + slippage if pending_qy[c] > 0 else 1.0 - slippage)
                qx[c] = pending_qx[c]
                qy[c] = pending_qy[c]
                entry_x[c] = fill_x
                entry_y[c] = fill_y
                entry_gross[c] = abs(qx[c] * fill_x) + abs(qy[c] * fill_y)
                entry_fee[c] = entry_gross[c] * fee_rate
                fees[c] += entry_fee[c]
                slippages[c] += abs(fill_x - x_open[i]) * abs(qx[c]) + abs(fill_y - y_open[i]) * abs(qy[c])
                cash_pnl[c] -= entry_fee[c]
                position[c] = side
            elif action == -1 and position[c] != 0:
                fill_x = x_open[i] * (1.0 - slippage if qx[c] > 0 else 1.0 + slippage)
                fill_y = y_open[i] * (1.0 - slippage if qy[c] > 0 else 1.0 + slippage)
                gross = qx[c] * (fill_x - entry_x[c]) + qy[c] * (fill_y - entry_y[c])
                close_notional = abs(qx[c] * fill_x) + abs(qy[c] * fill_y)
                close_fee = close_notional * fee_rate
                close_slip = abs(fill_x - x_open[i]) * abs(qx[c]) + abs(fill_y - y_open[i]) * abs(qy[c])
                cash_pnl[c] += gross - close_fee
                fees[c] += close_fee
                slippages[c] += close_slip
                trades[c] += 1
                if i >= start:
                    subperiod_trades[c, segment] += 1
                if gross - entry_fee[c] - close_fee - position_funding[c] > 0:
                    wins[c] += 1
                if pending_stop_close[c] != 0:
                    freeze_bars = int(math.ceil(model_lookback[c] * stop_freeze_lookback_multiplier))
                    if freeze_bars > 0:
                        freeze_until[c] = i + freeze_bars
                    if wait_for_model_update_after_non_z_exit:
                        reopen_lock_pending[c] = 1
                        stop_close_bar[c] = i
                position[c] = 0
                qx[c] = qy[c] = 0.0
                position_funding[c] = 0.0
                entry_fee[c] = 0.0
                entry_gross[c] = 0.0
            pending_action[c] = 0
            pending_stop_close[c] = 0

        # Mark the account at the current close before generating a signal.
        for c in range(n_candidates):
            unrealized = 0.0
            if position[c] != 0:
                unrealized = qx[c] * (x_close[i] - entry_x[c]) + qy[c] * (y_close[i] - entry_y[c])
            equity[c] = initial_equity + cash_pnl[c] + unrealized
            if i >= start:
                if equity[c] > peak[c]:
                    peak[c] = equity[c]
                drawdown = equity[c] / max(peak[c], 1e-12) - 1.0
                if drawdown < max_dd[c]:
                    max_dd[c] = drawdown
                drawdown_cash = equity[c] - peak[c]
                if drawdown_cash < max_dd_cash[c]:
                    max_dd_cash[c] = drawdown_cash
                subperiod_end_equity[c, segment] = equity[c]
            previous_equity[c] = equity[c]

        day = timestamps[i] // 86400000
        if i >= start and current_day < 0:
            current_day = day
            for c in range(n_candidates):
                daily_base[c] = equity[c]
                daily_last[c] = equity[c]
        elif i >= start and day != current_day:
            for c in range(n_candidates):
                daily_equity[c, int(current_day - start_day_key)] = daily_last[c]
                if abs(daily_base[c]) > 1e-12:
                    daily_return = daily_last[c] / daily_base[c] - 1.0
                    if math.isfinite(daily_return):
                        daily_sum[c] += daily_return
                        daily_sum2[c] += daily_return * daily_return
                        daily_count[c] += 1
                daily_base[c] = daily_last[c]
                daily_last[c] = equity[c]
            current_day = day
        elif i >= start:
            for c in range(n_candidates):
                daily_last[c] = equity[c]

        for c in range(n_candidates):
            lookback = model_lookback[c]
            if i + 1 < lookback:
                continue
            has_model = math.isfinite(beta[c])
            latest_spread = 0.0
            if has_model:
                tx = math.log(max(x_close[i], 1e-12)) if regression_method_code[c] == 1 else x_close[i]
                ty = math.log(max(y_close[i], 1e-12)) if regression_method_code[c] == 1 else y_close[i]
                latest_spread = ty - (alpha[c] + beta[c] * tx)
                z = (latest_spread - spread_mean[c]) / max(spread_std[c], 1e-12)
                pos = z_pos[c]
                z_hist[c, pos] = z
                z_pos[c] = (pos + 1) % max_ma
                if z_count[c] < max_ma:
                    z_count[c] += 1
                long_n = min(ma_long[c], z_count[c])
                short_n = min(ma_short[c], z_count[c])
                if position[c] != 0:
                    should_close = abs(z) <= exit_z[c]
                    if not should_close and position[c] == 1 and z <= 0.0 and exit_z[c] >= 0.0:
                        should_close = True
                    if not should_close and position[c] == -1 and z >= 0.0 and exit_z[c] >= 0.0:
                        should_close = True
                    if stop_enabled[c] != 0:
                        gross_mark = qx[c] * (x_close[i] - entry_x[c]) + qy[c] * (y_close[i] - entry_y[c])
                        close_cost = (abs(qx[c] * x_close[i]) + abs(qy[c] * y_close[i])) * (fee_rate + slippage)
                        net_return = (gross_mark - entry_fee[c] - position_funding[c] - close_cost) / max(entry_gross[c], 1e-12)
                        if net_return <= -stop_return[c]:
                            should_close = True
                            pending_stop_close[c] = 1
                    if should_close and pending_action[c] == 0:
                        pending_action[c] = -1
                else:
                    if z_count[c] >= min_samples[c] and long_n >= 1:
                        long_sum = 0.0
                        short_sum = 0.0
                        for j in range(long_n):
                            index = (z_pos[c] - 1 - j) % max_ma
                            long_sum += z_hist[c, index]
                        for j in range(short_n):
                            index = (z_pos[c] - 1 - j) % max_ma
                            short_sum += z_hist[c, index]
                        gap = short_sum / short_n - long_sum / long_n
                        side = 0
                        if z >= entry_z[c] and gap < 0.0:
                            side = 1
                        elif z <= -entry_z[c] and gap > 0.0:
                            side = -1
                        can_reopen = (freeze_until[c] < 0 or i >= freeze_until[c]) and reopen_lock_pending[c] == 0
                        if side != 0 and adf_p[c] <= adf_threshold[c] and i >= start and can_reopen and pending_action[c] == 0:
                            bx = _returns_beta(x_close, y_close, i, hedge_beta_lookback_bars, hedge_beta_min_samples)
                            if not math.isfinite(bx) or bx <= 1e-12:
                                bx = abs(beta[c]) if math.isfinite(beta[c]) and abs(beta[c]) > 1e-12 else 1.0
                            x_weight = bx / (1.0 + bx)
                            y_weight = 1.0 / (1.0 + bx)
                            qx_abs = target_capital * x_weight / max(x_close[i], 1e-12)
                            qy_abs = target_capital * y_weight / max(y_close[i], 1e-12)
                            pending_side[c] = side
                            pending_qx[c] = qx_abs if side == 1 else -qx_abs
                            pending_qy[c] = -qy_abs if side == 1 else qy_abs
                            pending_action[c] = 1

            # Fit after signal generation, matching TLSEstimator.update(): the
            # current close enters the new model for the next bar.
            if not has_model or (next_update[c] >= 0 and i >= next_update[c]):
                if position[c] == 0 or not has_model:
                    ok = _fit_tls(
                        x_close, y_close, i, lookback, regression_method_code[c], residual_buf,
                    )
                    if ok:
                        alpha[c], beta[c], spread_mean[c], spread_std[c] = residual_buf[lookback], residual_buf[lookback + 1], residual_buf[lookback + 2], residual_buf[lookback + 3]
                        adf_p[c] = adf_pvalue_c_lag1(residual_buf[:lookback])
                        next_update[c] = i + update_interval[c]
                        if reopen_lock_pending[c] != 0 and i > stop_close_bar[c]:
                            reopen_lock_pending[c] = 0
                        # ADF is a gate for the next signal, as in the full TLS path.
                    elif not has_model:
                        next_update[c] = i + update_interval[c]
                else:
                    # TLS ``freeze`` skips a due refit while a position is
                    # open and schedules the next attempt from this bar.
                    next_update[c] = i + update_interval[c]

    total_return = np.empty(n_candidates, dtype=np.float64)
    annualized = np.empty(n_candidates, dtype=np.float64)
    sharpe = np.empty(n_candidates, dtype=np.float64)
    win_rate = np.empty(n_candidates, dtype=np.float64)
    net_pnl = np.empty(n_candidates, dtype=np.float64)
    return_on_target = np.empty(n_candidates, dtype=np.float64)
    max_dd_on_target = np.empty(n_candidates, dtype=np.float64)
    active_subperiods = np.zeros(n_candidates, dtype=np.int64)
    profitable_subperiods = np.zeros(n_candidates, dtype=np.int64)
    subperiod_return_std = np.zeros(n_candidates, dtype=np.float64)
    for c in range(n_candidates):
        if current_day >= 0:
            daily_equity[c, int(current_day - start_day_key)] = daily_last[c]
        net_pnl[c] = equity[c] - initial_equity
        return_on_target[c] = net_pnl[c] / max(target_capital, 1e-12)
        max_dd_on_target[c] = max_dd_cash[c] / max(target_capital, 1e-12)
        total_return[c] = equity[c] / max(initial_equity, 1e-12) - 1.0
        years = max(1e-12, (end - start) / 525600.0)
        annualized[c] = max(0.0, equity[c] / max(initial_equity, 1e-12)) ** (1.0 / years) - 1.0 if equity[c] > 0 else -1.0
        if current_day >= 0:
            if abs(daily_base[c]) > 1e-12:
                daily_return = daily_last[c] / daily_base[c] - 1.0
                if math.isfinite(daily_return):
                    daily_sum[c] += daily_return
                    daily_sum2[c] += daily_return * daily_return
                    daily_count[c] += 1
            if daily_count[c] > 1:
                mean = daily_sum[c] / daily_count[c]
                variance = max(0.0, daily_sum2[c] / daily_count[c] - mean * mean)
                sharpe[c] = mean / math.sqrt(variance) * math.sqrt(252.0) if variance > 1e-20 else 0.0
            else:
                sharpe[c] = 0.0
        else:
            sharpe[c] = 0.0
        win_rate[c] = wins[c] / trades[c] if trades[c] > 0 else 0.0
        subperiod_sum = 0.0
        subperiod_sum2 = 0.0
        observed_subperiods = 0
        for segment_index in range(subperiod_count):
            first_equity = subperiod_start_equity[c, segment_index]
            last_equity = subperiod_end_equity[c, segment_index]
            if math.isfinite(first_equity) and math.isfinite(last_equity):
                segment_return = (last_equity - first_equity) / max(target_capital, 1e-12)
                subperiod_sum += segment_return
                subperiod_sum2 += segment_return * segment_return
                observed_subperiods += 1
            if subperiod_trades[c, segment_index] > 0:
                active_subperiods[c] += 1
                if math.isfinite(first_equity) and math.isfinite(last_equity) and last_equity > first_equity:
                    profitable_subperiods[c] += 1
        if observed_subperiods > 1:
            mean_subperiod_return = subperiod_sum / observed_subperiods
            variance = max(
                0.0,
                subperiod_sum2 / observed_subperiods - mean_subperiod_return * mean_subperiod_return,
            )
            subperiod_return_std[c] = math.sqrt(variance)
    return (initial, equity, total_return, annualized, sharpe, max_dd, trades,
            win_rate, fees, slippages, funding_total, net_pnl, max_dd_cash,
            return_on_target, max_dd_on_target, active_subperiods,
            profitable_subperiods, subperiod_return_std, daily_equity,
            start_day_key)


@njit(cache=True)
def _fit_tls(x_close, y_close, index, lookback, method, residual_buf):
    x_mean = 0.0
    y_mean = 0.0
    for j in range(index - lookback + 1, index + 1):
        x = math.log(max(x_close[j], 1e-12)) if method == 1 else x_close[j]
        y = math.log(max(y_close[j], 1e-12)) if method == 1 else y_close[j]
        x_mean += x
        y_mean += y
    x_mean /= lookback
    y_mean /= lookback
    sxx = syy = sxy = 0.0
    for j in range(index - lookback + 1, index + 1):
        x = math.log(max(x_close[j], 1e-12)) if method == 1 else x_close[j]
        y = math.log(max(y_close[j], 1e-12)) if method == 1 else y_close[j]
        dx = x - x_mean
        dy = y - y_mean
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    if lookback <= 1:
        return False
    sxx /= (lookback - 1)
    syy /= (lookback - 1)
    sxy /= (lookback - 1)
    if sxx <= 1e-12 or abs(sxy) <= 1e-12:
        return False
    discriminant = (syy - sxx) ** 2 + 4.0 * sxy ** 2
    b = ((syy - sxx) + math.sqrt(max(0.0, discriminant))) / (2.0 * sxy)
    a = y_mean - b * x_mean
    mean = 0.0
    for k, j in enumerate(range(index - lookback + 1, index + 1)):
        x = math.log(max(x_close[j], 1e-12)) if method == 1 else x_close[j]
        y = math.log(max(y_close[j], 1e-12)) if method == 1 else y_close[j]
        residual_buf[k] = y - (a + b * x)
        mean += residual_buf[k]
    mean /= lookback
    variance = 0.0
    for k in range(lookback):
        variance += (residual_buf[k] - mean) ** 2
    std = math.sqrt(max(variance / max(1, lookback - 1), 1e-16))
    residual_buf[lookback] = a
    residual_buf[lookback + 1] = b
    residual_buf[lookback + 2] = mean
    residual_buf[lookback + 3] = std
    return True


@njit(cache=True)
def _returns_beta(x_close, y_close, index, lookback, min_samples):
    first = max(1, index - lookback + 1)
    samples = index - first
    if samples < min_samples:
        return np.nan
    mx = my = 0.0
    for j in range(first, index + 1):
        xr = math.log(max(x_close[j], 1e-12)) - math.log(max(x_close[j - 1], 1e-12))
        yr = math.log(max(y_close[j], 1e-12)) - math.log(max(y_close[j - 1], 1e-12))
        mx += xr
        my += yr
    mx /= samples
    my /= samples
    vx = cov = 0.0
    for j in range(first, index + 1):
        xr = math.log(max(x_close[j], 1e-12)) - math.log(max(x_close[j - 1], 1e-12)) - mx
        yr = math.log(max(y_close[j], 1e-12)) - math.log(max(y_close[j - 1], 1e-12)) - my
        vx += xr * xr
        cov += xr * yr
    if vx <= 1e-18:
        return np.nan
    return cov / vx


def add_neighbor_scores(rows: list[dict], grid: CandidateGrid) -> None:
    _apply_neighbor_scores(rows, grid)


def _apply_neighbor_scores(rows, grid):
    keys = (
        "regression_method", "model_lookback_bars", "model_update_interval_bars",
        "residual_adf_max_pvalue", "entry_z", "exit_z",
        "reversion_ma_lookback_bars", "reversion_ma_short_lookback_bars",
        "pair_loss_stop_enabled", "pair_loss_stop_return",
    )
    values = [
        [p.get("estimator", {}).get("regression_method", "log_price"),
         p.get("estimator", {}).get("model_lookback_bars", 10080),
         p.get("estimator", {}).get("model_update_interval_bars", 240),
         p.get("estimator", {}).get("residual_adf_max_pvalue", 0.4),
         p.get("signal", {}).get("entry_z", 2.0), p.get("signal", {}).get("exit_z", 0.5),
         p.get("signal", {}).get("reversion_ma_lookback_bars", 60),
         p.get("signal", {}).get("reversion_ma_short_lookback_bars", 15),
         p.get("protection", {}).get("pair_loss_stop_enabled", False),
         p.get("protection", {}).get("pair_loss_stop_return", 0.05)]
        for p in grid.parameters
    ]
    dimensions = []
    for field in range(len(keys)):
        distinct_values = {values[x][field] for x in range(len(values))}
        if all(isinstance(value, (int, float, np.integer, np.floating, bool)) for value in distinct_values):
            ordered = sorted(distinct_values, key=float)
        else:
            ordered = sorted(distinct_values, key=lambda value: str(value))
        dimensions.append((ordered, {value: index for index, value in enumerate(ordered)}))
    row_by_key = {}
    for index, item in enumerate(values):
        row_by_key.setdefault(tuple(item), index)

    for i, row in enumerate(rows):
        neighbor_indices = set()
        base_key = list(values[i])
        for field, (ordered, positions) in enumerate(dimensions):
            position = positions[base_key[field]]
            for direction in (-1, 1):
                neighbor_position = position + direction
                if 0 <= neighbor_position < len(ordered):
                    candidate_key = list(base_key)
                    candidate_key[field] = ordered[neighbor_position]
                    neighbor_index = row_by_key.get(tuple(candidate_key))
                    if neighbor_index is not None and neighbor_index != i:
                        neighbor_indices.add(neighbor_index)
        neighbor_scores = [
            float(rows[index]["own_score"])
            for index in sorted(neighbor_indices)
            if rows[index].get("eligible") and math.isfinite(float(rows[index].get("own_score", math.nan)))
        ]
        row["neighbor_count"] = len(neighbor_indices)
        row["valid_neighbor_count"] = len(neighbor_scores)
        if neighbor_scores:
            mean = sum(neighbor_scores) / len(neighbor_scores)
            std = math.sqrt(sum((score - mean) ** 2 for score in neighbor_scores) / len(neighbor_scores))
            row["neighbor_mean_score"] = mean
            row["neighbor_score_std"] = std
            row["selection_score"] = 0.75 * row["own_score"] + 0.20 * mean - 0.05 * std
        else:
            row["neighbor_mean_score"] = 0.0
            row["neighbor_score_std"] = 0.0
            row["selection_score"] = 0.75 * row["own_score"]
        row["final_selection_score"] = row["selection_score"]
        row["selection_reason"] = (
            "own score 75% + eligible adjacent single-parameter neighborhood 25%"
            if neighbor_scores else
            "no eligible adjacent candidate; own score weighted 75%"
        )


def score_candidate_metrics(metrics: dict) -> dict:
    """Return bounded score components on the pair's deployed capital."""
    pair_return = _finite_number(metrics.get("return_on_target_capital"), 0.0)
    sharpe = _finite_number(metrics.get("sharpe"), 0.0)
    win_rate = _finite_number(metrics.get("win_rate"), 0.0)
    pair_drawdown = _finite_number(metrics.get("max_drawdown_on_target_capital"), 0.0)
    subperiod_count = max(1, int(metrics.get("subperiod_count", 1) or 1))
    profitable_subperiods = max(0, int(metrics.get("profitable_subperiod_count", 0) or 0))

    return_score = _clip(pair_return / 0.20, -1.0, 1.0)
    sharpe_score = _clip(sharpe / 3.0, -1.0, 1.0)
    win_rate_score = _clip((win_rate - 0.50) / 0.25, -1.0, 1.0)
    drawdown_score = -_clip(abs(pair_drawdown) / 0.20, 0.0, 1.0)
    subperiod_score = min(profitable_subperiods, subperiod_count) / subperiod_count
    own_score = (
        0.45 * return_score
        + 0.15 * sharpe_score
        + 0.10 * win_rate_score
        + 0.20 * drawdown_score
        + 0.10 * subperiod_score
    )
    return {
        "return_score": float(return_score),
        "sharpe_score": float(sharpe_score),
        "win_rate_score": float(win_rate_score),
        "drawdown_score": float(drawdown_score),
        "subperiod_score": float(subperiod_score),
        "own_score": float(own_score),
    }


def _eligible(metrics, min_trades, max_drawdown, min_active_subperiods, subperiod_count):
    required_active = min(max(1, int(subperiod_count)), max(0, int(min_active_subperiods)))
    drawdown = _finite_number(metrics.get("max_drawdown_on_target_capital"), 0.0)
    return (
        int(metrics.get("trade_count", 0) or 0) >= int(min_trades)
        and int(metrics.get("active_subperiod_count", 0) or 0) >= required_active
        and (max_drawdown is None or drawdown >= -abs(float(max_drawdown)))
    )


def _clip(value, lower, upper):
    return max(lower, min(upper, value))


def _finite_number(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _as_bool(value):
    if isinstance(value, str):
        return 1 if value.strip().lower() in {"1", "true", "yes", "y", "on"} else 0
    return 1 if bool(value) else 0


__all__ = [
    "CandidateGrid",
    "build_candidate_grid",
    "run_numba_pair_optimizer",
    "add_neighbor_scores",
    "score_candidate_metrics",
]
