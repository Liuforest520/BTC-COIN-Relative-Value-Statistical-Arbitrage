"""Numba implementation of the fixed ADF test used by TLS.

The fast walk-forward path intentionally implements the exact fixed
``maxlag=1, regression='c', autolag=None`` contract used by the production
TLS estimator.  The MacKinnon 1994 response-surface coefficients below are
the same coefficients used by statsmodels for the p-value approximation.
"""
from __future__ import annotations

from math import erf, isfinite, sqrt

import numpy as np
from numba import njit


_SMALL = np.asarray((
    (2.1659, 1.4412, 0.038269),
), dtype=np.float64)
_LARGE = np.asarray((
    (1.7339, 0.93202, -0.12745, -0.010368),
), dtype=np.float64)
_MIN_STAT = -18.83
_MAX_STAT = 2.74
_STAR_STAT = -1.61


@njit(cache=True)
def _normal_cdf(value):
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


@njit(cache=True)
def mackinnon_pvalue_c(adf_stat):
    """Approximate MacKinnon p-value for one-series ADF with a constant."""
    if not isfinite(adf_stat):
        return 1.0
    if adf_stat > _MAX_STAT:
        return 1.0
    if adf_stat < _MIN_STAT:
        return 0.0
    if adf_stat <= _STAR_STAT:
        c = _SMALL[0]
        z = c[0] + c[1] * adf_stat + c[2] * adf_stat * adf_stat
    else:
        c = _LARGE[0]
        z = c[0] + c[1] * adf_stat + c[2] * adf_stat * adf_stat + c[3] * adf_stat ** 3
    pvalue = _normal_cdf(z)
    return min(1.0, max(0.0, pvalue))


@njit(cache=True)
def adf_statistic_c_lag1(residuals):
    """Return the ADF t-statistic for ``c, maxlag=1, autolag=None``."""
    n = residuals.size
    if n < 8:
        return np.nan

    # X=[constant, residual[t-1], delta_residual[t-1]],
    # y=delta_residual[t], t=2..n-1.  This is statsmodels' lagmat trim='both'
    # layout for maxlag=1.
    observations = n - 2
    s00 = float(observations)
    s01 = s02 = s11 = s12 = s22 = 0.0
    b0 = b1 = b2 = 0.0
    for t in range(2, n):
        level = residuals[t - 1]
        lag_diff = residuals[t - 1] - residuals[t - 2]
        current_diff = residuals[t] - residuals[t - 1]
        s01 += level
        s02 += lag_diff
        s11 += level * level
        s12 += level * lag_diff
        s22 += lag_diff * lag_diff
        b0 += current_diff
        b1 += level * current_diff
        b2 += lag_diff * current_diff

    determinant = (
        s00 * (s11 * s22 - s12 * s12)
        - s01 * (s01 * s22 - s12 * s02)
        + s02 * (s01 * s12 - s11 * s02)
    )
    if not isfinite(determinant) or abs(determinant) <= 1e-24:
        return np.nan

    i00 = (s11 * s22 - s12 * s12) / determinant
    i11 = (s00 * s22 - s02 * s02) / determinant
    i22 = (s00 * s11 - s01 * s01) / determinant
    i01 = (s02 * s12 - s01 * s22) / determinant
    i02 = (s01 * s12 - s02 * s11) / determinant
    i12 = (s01 * s02 - s00 * s12) / determinant

    beta0 = i00 * b0 + i01 * b1 + i02 * b2
    beta1 = i01 * b0 + i11 * b1 + i12 * b2
    beta2 = i02 * b0 + i12 * b1 + i22 * b2
    sse = 0.0
    for t in range(2, n):
        level = residuals[t - 1]
        lag_diff = residuals[t - 1] - residuals[t - 2]
        current_diff = residuals[t] - residuals[t - 1]
        error = current_diff - beta0 - beta1 * level - beta2 * lag_diff
        sse += error * error
    dof = observations - 3
    if dof <= 0 or not isfinite(sse):
        return np.nan
    variance = sse / dof
    standard_error = sqrt(max(0.0, variance * i11))
    if standard_error <= 1e-15 or not isfinite(standard_error):
        return np.nan
    return beta1 / standard_error


@njit(cache=True)
def adf_pvalue_c_lag1(residuals):
    stat = adf_statistic_c_lag1(residuals)
    return mackinnon_pvalue_c(stat)


def adf_reference(residuals):
    """Reference helper used by tests; the fast path never calls it."""
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(np.asarray(residuals, dtype=np.float64), maxlag=1, regression="c", autolag=None)
    return float(result[0]), float(result[1])


__all__ = [
    "adf_statistic_c_lag1",
    "adf_pvalue_c_lag1",
    "mackinnon_pvalue_c",
    "adf_reference",
]
