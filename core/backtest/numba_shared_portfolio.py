"""Numba kernel for synchronized multi-pair shared-account training."""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from core.modules.estimator.adf_numba import adf_pvalue_c_lag1


@njit(cache=True)
def run_shared_portfolio_batch(
    timestamps, bar_indices, values, offsets,
    target_capital, beta_lookback, beta_min_samples, reversion_min_samples,
    stop_freeze_multiplier, wait_after_stop,
    train_start_ts, train_end_ts, subperiod_count,
    initial_equity, fee_rate, slippage, max_leverage,
    regression_method, model_lookback, update_interval, adf_threshold,
    entry_z, exit_z, ma_long, ma_short, stop_enabled, stop_return,
):
    n_times, n_pairs = bar_indices.shape
    n_candidates = regression_method.size
    measurement_start = np.searchsorted(timestamps, train_start_ts)
    measurement_end = np.searchsorted(timestamps, train_end_ts)
    max_ma = 5
    for value in ma_long:
        if value > max_ma:
            max_ma = value
    for value in reversion_min_samples:
        if value > max_ma:
            max_ma = value
    max_lookback = 10
    for value in model_lookback:
        if value > max_lookback:
            max_lookback = value

    alpha = np.full((n_candidates, n_pairs), np.nan)
    beta = np.full((n_candidates, n_pairs), np.nan)
    spread_mean = np.full((n_candidates, n_pairs), np.nan)
    spread_std = np.full((n_candidates, n_pairs), np.nan)
    adf_p = np.ones((n_candidates, n_pairs))
    next_update = np.full((n_candidates, n_pairs), -1, dtype=np.int64)
    position = np.zeros((n_candidates, n_pairs), dtype=np.int8)
    qx = np.zeros((n_candidates, n_pairs))
    qy = np.zeros((n_candidates, n_pairs))
    entry_x = np.zeros((n_candidates, n_pairs))
    entry_y = np.zeros((n_candidates, n_pairs))
    entry_gross = np.zeros((n_candidates, n_pairs))
    entry_fee = np.zeros((n_candidates, n_pairs))
    entry_reserve = np.zeros((n_candidates, n_pairs))
    position_capital = np.zeros((n_candidates, n_pairs))
    position_funding = np.zeros((n_candidates, n_pairs))
    pending_action = np.zeros((n_candidates, n_pairs), dtype=np.int8)
    pending_side = np.zeros((n_candidates, n_pairs), dtype=np.int8)
    pending_qx = np.zeros((n_candidates, n_pairs))
    pending_qy = np.zeros((n_candidates, n_pairs))
    pending_stop = np.zeros((n_candidates, n_pairs), dtype=np.int8)
    freeze_until = np.full((n_candidates, n_pairs), -1, dtype=np.int64)
    reopen_lock = np.zeros((n_candidates, n_pairs), dtype=np.int8)
    stop_close_bar = np.full((n_candidates, n_pairs), -1, dtype=np.int64)
    z_hist = np.zeros((n_candidates, n_pairs, max_ma))
    z_count = np.zeros((n_candidates, n_pairs), dtype=np.int64)
    z_pos = np.zeros((n_candidates, n_pairs), dtype=np.int64)

    available = np.full(n_candidates, initial_equity)
    equity = np.full(n_candidates, initial_equity)
    previous_equity = np.full(n_candidates, initial_equity)
    peak = np.full(n_candidates, initial_equity)
    max_dd = np.zeros(n_candidates)
    max_dd_cash = np.zeros(n_candidates)
    trades = np.zeros(n_candidates, dtype=np.int64)
    wins = np.zeros(n_candidates, dtype=np.int64)
    fees = np.zeros(n_candidates)
    slippages = np.zeros(n_candidates)
    funding_total = np.zeros(n_candidates)
    rejected_opens = np.zeros(n_candidates, dtype=np.int64)

    pair_realized = np.zeros((n_candidates, n_pairs))
    pair_trades = np.zeros((n_candidates, n_pairs), dtype=np.int64)
    pair_wins = np.zeros((n_candidates, n_pairs), dtype=np.int64)
    pair_peak = np.zeros((n_candidates, n_pairs))
    pair_max_dd = np.zeros((n_candidates, n_pairs))
    pair_fees = np.zeros((n_candidates, n_pairs))
    pair_funding = np.zeros((n_candidates, n_pairs))

    start_day = int(train_start_ts // 86400000)
    end_day = int((train_end_ts - 1) // 86400000)
    daily_equity = np.full((n_candidates, max(1, end_day - start_day + 1)), np.nan)
    daily_base = np.full(n_candidates, initial_equity)
    daily_last = np.full(n_candidates, initial_equity)
    daily_sum = np.zeros(n_candidates)
    daily_sum2 = np.zeros(n_candidates)
    daily_count = np.zeros(n_candidates, dtype=np.int64)
    current_day = -1
    sub_start = np.full((n_candidates, subperiod_count), np.nan)
    sub_end = np.full((n_candidates, subperiod_count), np.nan)
    sub_trades = np.zeros((n_candidates, subperiod_count), dtype=np.int64)
    last_x_close = np.zeros(n_pairs)
    last_y_close = np.zeros(n_pairs)
    residual_buf = np.empty(max_lookback + 4)

    for t in range(measurement_end):
        ts = timestamps[t]
        measuring = t >= measurement_start
        segment = 0
        if measuring:
            segment = int(((ts - train_start_ts) * subperiod_count) // max(1, train_end_ts - train_start_ts))
            segment = min(subperiod_count - 1, max(0, segment))

        # Current closes are the mark and signal prices after open execution.
        for p in range(n_pairs):
            local = bar_indices[t, p]
            if local >= 0:
                row = offsets[p] + local
                last_x_close[p] = values[row, 1]
                last_y_close[p] = values[row, 3]

        # Funding is settled at the executable open.
        for p in range(n_pairs):
            local = bar_indices[t, p]
            if local < 0:
                continue
            row = offsets[p] + local
            xo, yo = values[row, 0], values[row, 2]
            for c in range(n_candidates):
                if position[c, p] == 0:
                    continue
                payment = 0.0
                if values[row, 6] != 0.0:
                    payment += qx[c, p] * xo * values[row, 4]
                if values[row, 7] != 0.0:
                    payment += qy[c, p] * yo * values[row, 5]
                if payment != 0.0:
                    position_capital[c, p] -= payment
                    position_funding[c, p] += payment
                    funding_total[c] += payment
                    pair_funding[c, p] += payment

        # Forced liquidation and pending closes release capital before opens.
        for p in range(n_pairs):
            local = bar_indices[t, p]
            if local < 0:
                continue
            row = offsets[p] + local
            xo, yo = values[row, 0], values[row, 2]
            for c in range(n_candidates):
                if position[c, p] == 0:
                    continue
                open_unrealized = qx[c, p] * (xo - entry_x[c, p]) + qy[c, p] * (yo - entry_y[c, p])
                forced = position_capital[c, p] + open_unrealized <= 1e-9
                if pending_action[c, p] != -1 and not forced:
                    continue
                fill_x = xo * (1.0 - slippage if qx[c, p] > 0 else 1.0 + slippage)
                fill_y = yo * (1.0 - slippage if qy[c, p] > 0 else 1.0 + slippage)
                gross = qx[c, p] * (fill_x - entry_x[c, p]) + qy[c, p] * (fill_y - entry_y[c, p])
                close_notional = abs(qx[c, p] * fill_x) + abs(qy[c, p] * fill_y)
                close_fee = close_notional * fee_rate
                close_slip = abs(fill_x - xo) * abs(qx[c, p]) + abs(fill_y - yo) * abs(qy[c, p])
                released = position_capital[c, p] + gross - close_fee
                available[c] += released
                pair_realized[c, p] += released - entry_reserve[c, p]
                fees[c] += close_fee
                pair_fees[c, p] += close_fee
                slippages[c] += close_slip
                if measuring:
                    trades[c] += 1
                    pair_trades[c, p] += 1
                    sub_trades[c, segment] += 1
                    if gross - entry_fee[c, p] - close_fee - position_funding[c, p] > 0.0:
                        wins[c] += 1
                        pair_wins[c, p] += 1
                protective = forced or pending_stop[c, p] != 0
                if protective:
                    freeze_bars = int(math.ceil(model_lookback[c] * stop_freeze_multiplier[p]))
                    if freeze_bars > 0:
                        freeze_until[c, p] = local + freeze_bars
                    if wait_after_stop[p] != 0:
                        reopen_lock[c, p] = 1
                        stop_close_bar[c, p] = local
                position[c, p] = 0
                qx[c, p] = qy[c, p] = 0.0
                entry_fee[c, p] = entry_gross[c, p] = entry_reserve[c, p] = 0.0
                position_capital[c, p] = position_funding[c, p] = 0.0
                pending_action[c, p] = pending_stop[c, p] = 0

        # Pending opens compete for the same candidate-level free balance.
        for p in range(n_pairs):
            local = bar_indices[t, p]
            if local < 0:
                continue
            row = offsets[p] + local
            xo, yo = values[row, 0], values[row, 2]
            for c in range(n_candidates):
                if pending_action[c, p] != 1 or position[c, p] != 0:
                    continue
                fill_x = xo * (1.0 + slippage if pending_qx[c, p] > 0 else 1.0 - slippage)
                fill_y = yo * (1.0 + slippage if pending_qy[c, p] > 0 else 1.0 - slippage)
                requested_gross = abs(pending_qx[c, p] * fill_x) + abs(pending_qy[c, p] * fill_y)
                requested_reserve = requested_gross / max(max_leverage, 1e-12)
                scale = min(1.0, available[c] / max(requested_reserve, 1e-12))
                if scale <= 1e-12:
                    rejected_opens[c] += 1
                    pending_action[c, p] = 0
                    continue
                qx[c, p] = pending_qx[c, p] * scale
                qy[c, p] = pending_qy[c, p] * scale
                entry_x[c, p], entry_y[c, p] = fill_x, fill_y
                gross = requested_gross * scale
                reserve = requested_reserve * scale
                open_fee = gross * fee_rate
                available[c] -= reserve
                position_capital[c, p] = reserve - open_fee
                entry_reserve[c, p] = reserve
                entry_gross[c, p] = gross
                entry_fee[c, p] = open_fee
                position_funding[c, p] = 0.0
                position[c, p] = pending_side[c, p]
                fees[c] += open_fee
                pair_fees[c, p] += open_fee
                slip_cost = abs(fill_x - xo) * abs(qx[c, p]) + abs(fill_y - yo) * abs(qy[c, p])
                slippages[c] += slip_cost
                pending_action[c, p] = 0

        # Mark the real shared account after every timestamp.
        for c in range(n_candidates):
            marked = available[c]
            for p in range(n_pairs):
                if position[c, p] != 0:
                    unrealized = qx[c, p] * (last_x_close[p] - entry_x[c, p]) + qy[c, p] * (last_y_close[p] - entry_y[c, p])
                    marked += position_capital[c, p] + unrealized
                    pair_mark = pair_realized[c, p] + position_capital[c, p] + unrealized - entry_reserve[c, p]
                else:
                    pair_mark = pair_realized[c, p]
                if pair_mark > pair_peak[c, p]:
                    pair_peak[c, p] = pair_mark
                pair_dd = pair_mark - pair_peak[c, p]
                if pair_dd < pair_max_dd[c, p]:
                    pair_max_dd[c, p] = pair_dd
            equity[c] = marked
            if measuring:
                if not math.isfinite(sub_start[c, segment]):
                    sub_start[c, segment] = previous_equity[c]
                sub_end[c, segment] = marked
                if marked > peak[c]:
                    peak[c] = marked
                dd_cash = marked - peak[c]
                if dd_cash < max_dd_cash[c]:
                    max_dd_cash[c] = dd_cash
                dd = marked / max(peak[c], 1e-12) - 1.0
                if dd < max_dd[c]:
                    max_dd[c] = dd
                previous_equity[c] = marked

        if measuring:
            day = ts // 86400000
            if current_day < 0:
                current_day = day
                for c in range(n_candidates):
                    daily_base[c] = equity[c]
                    daily_last[c] = equity[c]
            elif day != current_day:
                for c in range(n_candidates):
                    daily_equity[c, int(current_day - start_day)] = daily_last[c]
                    daily_sum[c], daily_sum2[c], daily_count[c] = _add_return(
                        daily_sum[c], daily_sum2[c], daily_count[c], daily_last[c], daily_base[c]
                    )
                    daily_base[c] = daily_last[c]
                    daily_last[c] = equity[c]
                current_day = day
            else:
                for c in range(n_candidates):
                    daily_last[c] = equity[c]

        # Signals and model updates are pair-local, while orders remain shared.
        for p in range(n_pairs):
            local = bar_indices[t, p]
            if local < 0:
                continue
            row = offsets[p] + local
            xc, yc = values[row, 1], values[row, 3]
            for c in range(n_candidates):
                lookback = model_lookback[c]
                has_model = math.isfinite(beta[c, p])
                if has_model:
                    tx = math.log(max(xc, 1e-12)) if regression_method[c] == 1 else xc
                    ty = math.log(max(yc, 1e-12)) if regression_method[c] == 1 else yc
                    spread = ty - (alpha[c, p] + beta[c, p] * tx)
                    z = (spread - spread_mean[c, p]) / max(spread_std[c, p], 1e-12)
                    pos = z_pos[c, p]
                    z_hist[c, p, pos] = z
                    z_pos[c, p] = (pos + 1) % max_ma
                    if z_count[c, p] < max_ma:
                        z_count[c, p] += 1
                    _schedule_pair_action(
                        c, p, t, local, row, values, offsets, target_capital,
                        beta_lookback, beta_min_samples, reversion_min_samples,
                        position, qx, qy, entry_x, entry_y, entry_gross,
                        entry_fee, position_funding, pending_action, pending_side,
                        pending_qx, pending_qy, pending_stop, freeze_until,
                        reopen_lock, z_hist, z_count, z_pos, max_ma, beta,
                        adf_p, adf_threshold, entry_z, exit_z, ma_long, ma_short,
                        stop_enabled, stop_return, fee_rate, slippage, z, measuring,
                    )

                if local + 1 >= lookback and (not has_model or (next_update[c, p] >= 0 and local >= next_update[c, p])):
                    if position[c, p] == 0 or not has_model:
                        ok = _fit_tls_flat(values, offsets[p], local, lookback, regression_method[c], residual_buf)
                        if ok:
                            alpha[c, p] = residual_buf[lookback]
                            beta[c, p] = residual_buf[lookback + 1]
                            spread_mean[c, p] = residual_buf[lookback + 2]
                            spread_std[c, p] = residual_buf[lookback + 3]
                            adf_p[c, p] = adf_pvalue_c_lag1(residual_buf[:lookback])
                            next_update[c, p] = local + update_interval[c]
                            if reopen_lock[c, p] != 0 and local > stop_close_bar[c, p]:
                                reopen_lock[c, p] = 0
                        elif not has_model:
                            next_update[c, p] = local + update_interval[c]
                    else:
                        next_update[c, p] = local + update_interval[c]

    return _finish_results(
        timestamps, measurement_start, measurement_end, subperiod_count,
        initial_equity, target_capital, equity, position, qx, qy, entry_x,
        entry_y, entry_reserve, position_capital, pair_realized, last_x_close,
        last_y_close, peak, max_dd, max_dd_cash, trades, wins, fees, slippages,
        funding_total, rejected_opens, daily_equity, daily_base, daily_last,
        daily_sum, daily_sum2, daily_count, current_day, start_day, sub_start,
        sub_end, sub_trades, pair_trades, pair_wins, pair_max_dd, pair_fees,
        pair_funding,
    )


@njit(cache=True)
def _schedule_pair_action(
    c, p, t, local, row, values, offsets, target_capital,
    beta_lookback, beta_min_samples, reversion_min_samples,
    position, qx, qy, entry_x, entry_y, entry_gross, entry_fee,
    position_funding, pending_action, pending_side, pending_qx, pending_qy,
    pending_stop, freeze_until, reopen_lock, z_hist, z_count, z_pos, max_ma,
    beta, adf_p, adf_threshold, entry_z, exit_z, ma_long, ma_short,
    stop_enabled, stop_return, fee_rate, slippage, z, measuring,
):
    xc, yc = values[row, 1], values[row, 3]
    if position[c, p] != 0:
        should_close = abs(z) <= exit_z[c]
        if not should_close and position[c, p] == 1 and z <= 0.0:
            should_close = True
        if not should_close and position[c, p] == -1 and z >= 0.0:
            should_close = True
        if stop_enabled[c] != 0:
            gross_mark = qx[c, p] * (xc - entry_x[c, p]) + qy[c, p] * (yc - entry_y[c, p])
            close_cost = (abs(qx[c, p] * xc) + abs(qy[c, p] * yc)) * (fee_rate + slippage)
            net_return = (
                gross_mark - entry_fee[c, p] - position_funding[c, p] - close_cost
            ) / max(entry_gross[c, p], 1e-12)
            if net_return <= -stop_return[c]:
                should_close = True
                pending_stop[c, p] = 1
        if should_close and pending_action[c, p] == 0:
            pending_action[c, p] = -1
        return

    required = max(ma_long[c], reversion_min_samples[p])
    if not measuring or z_count[c, p] < required:
        return
    long_n = min(ma_long[c], z_count[c, p])
    short_n = min(ma_short[c], z_count[c, p])
    long_sum = 0.0
    short_sum = 0.0
    for j in range(long_n):
        index = (z_pos[c, p] - 1 - j) % max_ma
        long_sum += z_hist[c, p, index]
    for j in range(short_n):
        index = (z_pos[c, p] - 1 - j) % max_ma
        short_sum += z_hist[c, p, index]
    gap = short_sum / max(short_n, 1) - long_sum / max(long_n, 1)
    side = 0
    if z >= entry_z[c] and gap < 0.0:
        side = 1
    elif z <= -entry_z[c] and gap > 0.0:
        side = -1
    if (
        side == 0 or adf_p[c, p] > adf_threshold[c]
        or pending_action[c, p] != 0
        or (freeze_until[c, p] >= 0 and local < freeze_until[c, p])
        or reopen_lock[c, p] != 0
    ):
        return
    hedge = _returns_beta_flat(
        values, offsets[p], local, beta_lookback[p], beta_min_samples[p]
    )
    if not math.isfinite(hedge) or hedge <= 1e-12:
        hedge = abs(beta[c, p]) if math.isfinite(beta[c, p]) and abs(beta[c, p]) > 1e-12 else 1.0
    x_weight = hedge / (1.0 + hedge)
    y_weight = 1.0 / (1.0 + hedge)
    qx_abs = target_capital[p] * x_weight / max(xc, 1e-12)
    qy_abs = target_capital[p] * y_weight / max(yc, 1e-12)
    pending_side[c, p] = side
    pending_qx[c, p] = qx_abs if side == 1 else -qx_abs
    pending_qy[c, p] = -qy_abs if side == 1 else qy_abs
    pending_action[c, p] = 1


@njit(cache=True)
def _fit_tls_flat(values, offset, local, lookback, method, residual_buf):
    first = offset + local - lookback + 1
    last = offset + local + 1
    x_mean = 0.0
    y_mean = 0.0
    for row in range(first, last):
        x = math.log(max(values[row, 1], 1e-12)) if method == 1 else values[row, 1]
        y = math.log(max(values[row, 3], 1e-12)) if method == 1 else values[row, 3]
        x_mean += x
        y_mean += y
    x_mean /= lookback
    y_mean /= lookback
    sxx = syy = sxy = 0.0
    for row in range(first, last):
        x = math.log(max(values[row, 1], 1e-12)) if method == 1 else values[row, 1]
        y = math.log(max(values[row, 3], 1e-12)) if method == 1 else values[row, 3]
        dx, dy = x - x_mean, y - y_mean
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    sxx /= max(1, lookback - 1)
    syy /= max(1, lookback - 1)
    sxy /= max(1, lookback - 1)
    if sxx <= 1e-12 or abs(sxy) <= 1e-12:
        return False
    b = ((syy - sxx) + math.sqrt(max(0.0, (syy - sxx) ** 2 + 4.0 * sxy ** 2))) / (2.0 * sxy)
    a = y_mean - b * x_mean
    mean = 0.0
    for k, row in enumerate(range(first, last)):
        x = math.log(max(values[row, 1], 1e-12)) if method == 1 else values[row, 1]
        y = math.log(max(values[row, 3], 1e-12)) if method == 1 else values[row, 3]
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
def _returns_beta_flat(values, offset, local, lookback, min_samples):
    first_local = max(1, local - lookback + 1)
    samples = local - first_local + 1
    if samples < min_samples:
        return np.nan
    mx = my = 0.0
    for local_index in range(first_local, local + 1):
        row = offset + local_index
        previous = row - 1
        xr = math.log(max(values[row, 1], 1e-12)) - math.log(max(values[previous, 1], 1e-12))
        yr = math.log(max(values[row, 3], 1e-12)) - math.log(max(values[previous, 3], 1e-12))
        mx += xr
        my += yr
    mx /= samples
    my /= samples
    vx = cov = 0.0
    for local_index in range(first_local, local + 1):
        row = offset + local_index
        previous = row - 1
        xr = math.log(max(values[row, 1], 1e-12)) - math.log(max(values[previous, 1], 1e-12)) - mx
        yr = math.log(max(values[row, 3], 1e-12)) - math.log(max(values[previous, 3], 1e-12)) - my
        vx += xr * xr
        cov += xr * yr
    return cov / vx if vx > 1e-18 else np.nan


@njit(cache=True)
def _add_return(total, total2, count, last, base):
    if abs(base) > 1e-12:
        value = last / base - 1.0
        if math.isfinite(value):
            return total + value, total2 + value * value, count + 1
    return total, total2, count


@njit(cache=True)
def _finish_results(
    timestamps, measurement_start, measurement_end, subperiod_count,
    initial_equity, target_capital, equity, position, qx, qy, entry_x,
    entry_y, entry_reserve, position_capital, pair_realized, last_x_close,
    last_y_close, peak, max_dd, max_dd_cash, trades, wins, fees, slippages,
    funding_total, rejected_opens, daily_equity, daily_base, daily_last,
    daily_sum, daily_sum2, daily_count, current_day, start_day, sub_start,
    sub_end, sub_trades, pair_trades, pair_wins, pair_max_dd, pair_fees,
    pair_funding,
):
    n_candidates, n_pairs = position.shape
    total_return = np.zeros(n_candidates)
    annualized = np.zeros(n_candidates)
    sharpe = np.zeros(n_candidates)
    win_rate = np.zeros(n_candidates)
    net_pnl = np.zeros(n_candidates)
    active_subperiods = np.zeros(n_candidates, dtype=np.int64)
    profitable_subperiods = np.zeros(n_candidates, dtype=np.int64)
    subperiod_std = np.zeros(n_candidates)
    pair_net = np.zeros((n_candidates, n_pairs))

    for c in range(n_candidates):
        if current_day >= 0:
            daily_equity[c, int(current_day - start_day)] = daily_last[c]
            daily_sum[c], daily_sum2[c], daily_count[c] = _add_return(
                daily_sum[c], daily_sum2[c], daily_count[c], daily_last[c], daily_base[c]
            )
        net_pnl[c] = equity[c] - initial_equity
        total_return[c] = net_pnl[c] / max(initial_equity, 1e-12)
        duration_years = max(
            1e-12,
            (timestamps[measurement_end - 1] - timestamps[measurement_start] + 60000)
            / 31_536_000_000.0,
        )
        annualized[c] = (
            max(0.0, equity[c] / max(initial_equity, 1e-12)) ** (1.0 / duration_years) - 1.0
            if equity[c] > 0 else -1.0
        )
        if daily_count[c] > 1:
            mean = daily_sum[c] / daily_count[c]
            variance = max(0.0, daily_sum2[c] / daily_count[c] - mean * mean)
            sharpe[c] = mean / math.sqrt(variance) * math.sqrt(252.0) if variance > 1e-20 else 0.0
        win_rate[c] = wins[c] / trades[c] if trades[c] > 0 else 0.0

        sub_sum = 0.0
        sub_sum2 = 0.0
        observed = 0
        for segment in range(subperiod_count):
            first = sub_start[c, segment]
            last = sub_end[c, segment]
            if math.isfinite(first) and math.isfinite(last):
                value = (last - first) / max(initial_equity, 1e-12)
                sub_sum += value
                sub_sum2 += value * value
                observed += 1
            if sub_trades[c, segment] > 0:
                active_subperiods[c] += 1
                if math.isfinite(first) and math.isfinite(last) and last > first:
                    profitable_subperiods[c] += 1
        if observed > 1:
            mean = sub_sum / observed
            subperiod_std[c] = math.sqrt(max(0.0, sub_sum2 / observed - mean * mean))

        for p in range(n_pairs):
            value = pair_realized[c, p]
            if position[c, p] != 0:
                unrealized = qx[c, p] * (last_x_close[p] - entry_x[c, p]) + qy[c, p] * (last_y_close[p] - entry_y[c, p])
                value += position_capital[c, p] + unrealized - entry_reserve[c, p]
            pair_net[c, p] = value

    return (
        equity, total_return, annualized, sharpe, max_dd, trades, win_rate,
        fees, slippages, funding_total, net_pnl, max_dd_cash,
        active_subperiods, profitable_subperiods, subperiod_std, daily_equity,
        start_day, pair_net, pair_trades, pair_wins, pair_max_dd, pair_fees,
        pair_funding, rejected_opens,
    )


__all__ = ["run_shared_portfolio_batch"]
