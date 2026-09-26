import math
import numpy as np


DEFAULTS = {"POR": 0.1, "PERM": 0.01, "SW": 99.9}

BOUNDS = {"POR": (0.0, 50.0), "PERM": (0.01, 10000.0), "SW": (8.305, 99.9)}

def kernel_matrix(actions: np.ndarray, support: np.ndarray, target: str) -> np.ndarray:
    """K[i, j] = official accuracy of action a_i when the truth is y_j."""
    a, y = actions[:, None], support[None, :]
    if target == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(a, 1e-5)) - np.log10(np.maximum(y, 1e-5))))
    tol = 0.08 if target == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(a - y) / (tol * (np.abs(y) + 1e-5)))

def make_rows(depth, pred, log_id):
    rows = []
    for i in range(len(depth)):
        row = {"DEPTH": round(float(depth[i]), 6), "POR": round(float(pred[i, 0]), 6),
               "PERM": round(float(pred[i, 1]), 6), "SW": round(float(pred[i, 2]), 6)}
        if not all(math.isfinite(row[k]) for k in ("POR", "PERM", "SW")):
            raise RuntimeError(f"non-finite prediction {log_id} row {i}")
        rows.append(row)
    return rows
