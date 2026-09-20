from __future__ import annotations

import numpy as np


def official_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute the official POR/PERM/SW accuracy and weighted total score."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape or y_true.shape[1] != 3:
        raise ValueError("Expected arrays with shape (n_samples, 3)")

    por_true, perm_true, sw_true = y_true.T
    por_pred, perm_pred, sw_pred = y_pred.T
    epsilon = 1e-5

    por_accuracy = np.maximum(
        0.0,
        1.0 - np.abs(por_pred - por_true) / (0.08 * (np.abs(por_true) + epsilon)),
    )
    perm_ratio = np.maximum(perm_pred / perm_true, epsilon)
    perm_accuracy = np.maximum(0.0, 1.0 - np.abs(np.log10(perm_ratio)))
    sw_accuracy = np.maximum(
        0.0,
        1.0 - np.abs(sw_pred - sw_true) / (0.05 * (np.abs(sw_true) + epsilon)),
    )

    por_score = float(np.mean(por_accuracy))
    perm_score = float(np.mean(perm_accuracy))
    sw_score = float(np.mean(sw_accuracy))
    total_score = (0.3 * por_score + 0.35 * perm_score + 0.35 * sw_score) * 100.0
    return {
        "POR": por_score,
        "PERM": perm_score,
        "SW": sw_score,
        "TOTAL": total_score,
    }


def official_target_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target: str,
) -> float:
    """Compute the official score for one target without evaluating the others."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError("Expected one-dimensional arrays with equal length")

    epsilon = 1e-5
    if target == "POR":
        accuracy = np.maximum(
            0.0,
            1.0 - np.abs(y_pred - y_true) / (0.08 * (np.abs(y_true) + epsilon)),
        )
    elif target == "PERM":
        ratio = np.maximum(y_pred / np.maximum(y_true, epsilon), epsilon)
        accuracy = np.maximum(0.0, 1.0 - np.abs(np.log10(ratio)))
    elif target == "SW":
        accuracy = np.maximum(
            0.0,
            1.0 - np.abs(y_pred - y_true) / (0.05 * (np.abs(y_true) + epsilon)),
        )
    else:
        raise ValueError(f"Unknown target: {target}")
    return float(np.mean(accuracy))
