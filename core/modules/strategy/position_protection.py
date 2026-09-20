"""Protective exits for frozen Price and Log-Price pair positions.

The module contains no exchange side effects.  It only turns a frozen model
snapshot, actual filled quantities and current prices into a theoretical
zero-net-PnL X boundary or a pair-level mark-to-market return decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log


@dataclass(frozen=True)
class StopLossBoundary:
    x_price: float | None
    target_residual: float | None
    reason: str
    trigger_direction: str | None = None


def supported(regression_method: str, position_update_policy: str, enabled: bool) -> bool:
    return bool(
        enabled
        and str(position_update_policy).lower() == "freeze"
        and str(regression_method).lower() in {"price", "log_price"}
    )


def signed_quantities(side: str, x_quantity: float, y_quantity: float) -> tuple[float, float]:
    """Return signed X/Y quantities for a pair position."""
    if side == "long_x":
        return abs(float(x_quantity)), -abs(float(y_quantity))
    if side == "short_x":
        return -abs(float(x_quantity)), abs(float(y_quantity))
    return 0.0, 0.0


def target_residual(side: str, exit_z: float | None, spread_mean: float | None,
                    spread_std: float | None) -> float | None:
    if spread_mean is None or spread_std is None:
        return None
    z = float(exit_z or 0.0)
    direction = 1.0 if side == "long_x" else -1.0 if side == "short_x" else 0.0
    return float(spread_mean) + direction * z * float(spread_std)


def theoretical_y_price(regression_method: str, alpha: float, beta: float,
                         x_price: float, residual: float) -> float | None:
    try:
        x = float(x_price)
        if x <= 0 or not all(isfinite(float(v)) for v in (alpha, beta, residual)):
            return None
        if regression_method == "log_price":
            return float(exp(float(alpha) + float(beta) * log(x) + float(residual)))
        if regression_method == "price":
            return float(float(alpha) + float(beta) * x + float(residual))
    except (OverflowError, ValueError):
        return None
    return None


def close_cost(x_price: float, y_price: float, x_quantity: float, y_quantity: float,
               fee_rate: float, slippage_rate: float) -> float:
    notional = abs(float(x_price) * float(x_quantity)) + abs(float(y_price) * float(y_quantity))
    return notional * max(0.0, float(fee_rate) + float(slippage_rate))


def gross_pnl(side: str, x_price: float, y_price: float,
              entry_x_price: float, entry_y_price: float,
              x_quantity: float, y_quantity: float) -> float:
    qx, qy = signed_quantities(side, x_quantity, y_quantity)
    return qx * (float(x_price) - float(entry_x_price)) + qy * (float(y_price) - float(entry_y_price))


def net_pnl_at_x(
    regression_method: str,
    alpha: float,
    beta: float,
    residual: float,
    x_price: float,
    entry_x_price: float,
    entry_y_price: float,
    side: str,
    x_quantity: float,
    y_quantity: float,
    entry_cost: float,
    fee_rate: float,
    slippage_rate: float,
) -> float | None:
    y_price = theoretical_y_price(regression_method, alpha, beta, x_price, residual)
    if y_price is None or y_price <= 0:
        return None
    return gross_pnl(
        side, x_price, y_price, entry_x_price, entry_y_price, x_quantity, y_quantity
    ) - float(entry_cost) - close_cost(
        x_price, y_price, x_quantity, y_quantity, fee_rate, slippage_rate
    )


def solve_zero_net_x_price(
    regression_method: str,
    alpha: float | None,
    beta: float | None,
    spread_mean: float | None,
    spread_std: float | None,
    exit_z: float | None,
    side: str,
    entry_x_price: float | None,
    entry_y_price: float | None,
    x_quantity: float,
    y_quantity: float,
    entry_cost: float,
    fee_rate: float,
    slippage_rate: float,
) -> StopLossBoundary:
    """Solve the nearest adverse X price where frozen-target net PnL is zero.

    A logarithmic grid is used before bisection.  This works for both the
    linear Price relation and the nonlinear Log-Price relation and avoids a
    fragile analytic root when costs are included.
    """
    values = (alpha, beta, spread_mean, spread_std, entry_x_price, entry_y_price)
    fallback_direction = "lower" if side == "long_x" else "upper"
    if side not in {"long_x", "short_x"} or any(v is None for v in values):
        return StopLossBoundary(None, None, "missing frozen model or entry data", fallback_direction)
    x0 = float(entry_x_price)
    y0 = float(entry_y_price)
    if x0 <= 0 or y0 <= 0 or x_quantity <= 0 or y_quantity <= 0:
        return StopLossBoundary(None, None, "invalid entry data", fallback_direction)
    residual = target_residual(side, exit_z, spread_mean, spread_std)
    if residual is None:
        return StopLossBoundary(None, None, "missing exit target", fallback_direction)

    def f(x):
        return net_pnl_at_x(
            regression_method, float(alpha), float(beta), residual, x, x0, y0,
            side, x_quantity, y_quantity, entry_cost, fee_rate, slippage_rate,
        )

    # The target at the entry X price must offer positive net expectancy. If
    # it does not, there is no meaningful zero-profit stop boundary.
    f0 = f(x0)
    if f0 is None or not isfinite(float(f0)) or f0 <= 0:
        return StopLossBoundary(
            None, residual, "frozen target is not net profitable at entry", fallback_direction
        )

    def _multipliers(direction: str) -> list[float]:
        # Include a dense near-entry grid.  The old decade grid had its first
        # point roughly 11% away from entry, so ordinary fee/slippage-sized
        # stop boundaries were never bracketed.
        fractions = [
            min(0.999999999, 10 ** (-8 + 8 * i / 240))
            for i in range(241)
        ]
        if direction == "lower":
            values = [1.0 - fraction for fraction in fractions]
            values.extend(10 ** exponent for exponent in range(-10, 0))
            return sorted({value for value in values if value > 0}, reverse=True)
        values = [1.0 + fraction for fraction in fractions]
        values.extend((2.0, 3.0, 5.0, 10.0, 100.0, 1000.0, 10000.0))
        return sorted(set(values))

    probe_fraction = 1e-4
    lower_probe = f(x0 * (1.0 - probe_fraction))
    upper_probe = f(x0 * (1.0 + probe_fraction))
    if (
        lower_probe is not None
        and upper_probe is not None
        and isfinite(float(lower_probe))
        and isfinite(float(upper_probe))
        and abs(float(lower_probe) - float(upper_probe)) > max(1e-10, abs(f0) * 1e-10)
    ):
        trigger_direction = "lower" if float(lower_probe) < float(upper_probe) else "upper"
    else:
        trigger_direction = fallback_direction

    def _find_root(direction: str) -> float | None:
        previous_x = x0
        previous_value = float(f0)
        for multiplier in _multipliers(direction):
            if abs(multiplier - 1.0) <= 1e-15:
                continue
            x = x0 * multiplier
            current = f(x)
            if current is None or not isfinite(float(current)):
                continue
            current = float(current)
            if previous_value == 0.0 or current == 0.0 or previous_value * current < 0.0:
                lo, hi = (x, previous_x) if x < previous_x else (previous_x, x)
                flo, fhi = f(lo), f(hi)
                if flo is None or fhi is None:
                    continue
                flo, fhi = float(flo), float(fhi)
                if abs(flo) <= max(1e-10, abs(f0) * 1e-10):
                    return lo
                if abs(fhi) <= max(1e-10, abs(f0) * 1e-10):
                    return hi
                for _ in range(80):
                    mid = (lo + hi) / 2.0
                    fmid = f(mid)
                    if fmid is None or not isfinite(float(fmid)):
                        break
                    fmid = float(fmid)
                    if abs(fmid) <= max(1e-10, abs(f0) * 1e-10):
                        return mid
                    if flo * fmid <= 0.0:
                        hi, fhi = mid, fmid
                    else:
                        lo, flo = mid, fmid
                return (lo + hi) / 2.0
            previous_x, previous_value = x, current
        return None

    root = _find_root(trigger_direction)
    if root is not None:
        return StopLossBoundary(float(root), residual, "zero_net_pnl", trigger_direction)
    return StopLossBoundary(None, residual, "no adverse zero_net_pnl root", trigger_direction)


def mark_net_pnl(
    side: str,
    x_price: float,
    y_price: float,
    entry_x_price: float,
    entry_y_price: float,
    x_quantity: float,
    y_quantity: float,
    entry_cost: float,
    funding_cost: float,
    fee_rate: float,
    slippage_rate: float,
) -> float:
    return gross_pnl(
        side, x_price, y_price, entry_x_price, entry_y_price, x_quantity, y_quantity
    ) - float(entry_cost) - float(funding_cost) - close_cost(
        x_price, y_price, x_quantity, y_quantity, fee_rate, slippage_rate
    )
