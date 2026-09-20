"""POR decisions maximizing the supplied discrete distribution's official utility.

Optimality is for point masses at class representatives, not the unknown true
conditional distribution. All descending slope changes occur at those masses;
a maximum of the piecewise-linear triangular utility occurs at a mass.
"""
from __future__ import annotations
import numpy as np


def utility_action(probabilities, support, *, chunk_size=4096):
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(support, dtype=np.float64)
    if p.ndim != 2 or y.ndim != 1 or p.shape[1] != len(y) or not len(y):
        raise ValueError('Expected probabilities (rows, classes) and nonempty support (classes,)')
    if chunk_size < 1 or not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError('Invalid support or chunk size')
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p.sum(axis=1) <= 0):
        raise ValueError('Probabilities must be finite, nonnegative, with positive row sums')
    # K[action, true class]; the tolerance denominator uses the TRUE value.
    kernel = np.maximum(0., 1. - np.abs(y[:, None] - y[None, :]) /
                        (.08 * (np.abs(y[None, :]) + 1e-5)))
    result = np.empty(len(p), dtype=np.float64)
    for start in range(0, len(p), chunk_size):
        block = p[start:start + chunk_size]
        result[start:start + len(block)] = y[np.argmax(block @ kernel.T, axis=1)]
    return result


def por_prediction(probabilities, default_probability, centers_log1p, calibration,
                   bounds=(0., 50.), default_value=.1, mode='utility'):
    p = np.asarray(probabilities, dtype=np.float64)
    d = np.asarray(default_probability, dtype=np.float64)
    centers = np.asarray(centers_log1p, dtype=np.float64)
    if p.ndim != 2 or centers.shape != (p.shape[1],) or d.shape != (len(p),):
        raise ValueError('POR probability/center dimensions differ')
    if (not np.isfinite(p).all() or np.any(p < 0) or
            not np.allclose(p.sum(axis=1), 1., atol=1e-5) or
            not np.isfinite(d).all() or np.any((d < 0) | (d > 1))):
        raise ValueError('Invalid model probabilities')
    if not np.isfinite(centers).all():
        raise ValueError('Nonfinite POR centers')
    support = np.expm1(centers)
    if mode == 'argmax':
        continuous = support[np.argmax(p, axis=1)]
    elif mode == 'utility':
        continuous = utility_action(p, support)
    elif mode == 'joint':
        # Joint decision uses raw calibrated probabilities; no historical gate/scale.
        joint_p = np.column_stack((d, (1. - d[:, None]) * p))
        return np.clip(utility_action(joint_p, np.r_[default_value, support]), *bounds)
    else:
        raise ValueError(f'Unknown POR decision: {mode}')
    continuous = np.clip(np.expm1(np.log1p(continuous) * calibration['scale'] +
                                calibration['offset']), *bounds)
    return np.where(d >= calibration['threshold'], default_value, continuous)
