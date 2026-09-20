"""Decision-layer utilities: official-metric kernels, Bayes-action grids, HMM gate smoothing, config, output."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

DEFAULTS = {"POR": 0.1, "PERM": 0.01, "SW": 99.9}
BOUNDS = {"POR": (0.0, 50.0), "PERM": (0.01, 10000.0), "SW": (8.305, 99.9)}
TI = {"POR": 0, "PERM": 1, "SW": 2}
TINV = {"POR": np.expm1, "PERM": lambda v: 10 ** v, "SW": lambda v: v}


def kernel_matrix(actions: np.ndarray, support: np.ndarray, target: str) -> np.ndarray:
    """K[i, j] = official accuracy of action a_i when the truth is y_j."""
    a, y = actions[:, None], support[None, :]
    if target == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(a, 1e-5)) - np.log10(np.maximum(y, 1e-5))))
    tol = 0.08 if target == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(a - y) / (tol * (np.abs(y) + 1e-5)))


def dense_actions(sup, t):
    """800-point action grid spanning the bin support (uniform for SW, log-uniform for POR/PERM)."""
    return np.linspace(sup[0], sup[-1], 800) if t == "SW" else np.exp(np.linspace(np.log(sup[0] * 0.9), np.log(sup[-1] * 1.1), 800))


def hmm_posterior(p, stay=0.98, prior=None):
    """Two-state forward-backward smoothing of a default-probability sequence (optional in the decision block)."""
    n = len(p)
    p = np.clip(p, 1e-4, 1 - 1e-4)
    e = np.stack([1 - p, p], 1)
    A = np.array([[stay, 1 - stay], [1 - stay, stay]])
    alpha = np.zeros((n, 2)); beta = np.ones((n, 2))
    alpha[0] = e[0] * (0.5 if prior is None else prior); alpha[0] /= alpha[0].sum()
    for t in range(1, n):
        alpha[t] = e[t] * (alpha[t - 1] @ A); alpha[t] /= alpha[t].sum()
    for t in range(n - 2, -1, -1):
        beta[t] = A @ (e[t + 1] * beta[t + 1]); beta[t] /= beta[t].sum()
    post = alpha * beta; post /= post.sum(1, keepdims=True)
    return post[:, 1]


def load_config(path: Path) -> dict:
    """Minimal reader for the flat `key: value` config.yaml (no PyYAML dependency)."""
    cfg = {}
    if path is None or not Path(path).exists():
        return cfg
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        cfg[key.strip()] = value.strip().strip("'\"")
    return cfg


def select_device(name: str = "auto"):
    import torch
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_rows(depth, pred, log_id):
    rows = []
    for i in range(len(depth)):
        row = {"depth": round(float(depth[i]), 6), "POR": round(float(pred[i, 0]), 6),
               "PERM": round(float(pred[i, 1]), 6), "SW": round(float(pred[i, 2]), 6)}
        if not all(math.isfinite(row[k]) for k in ("POR", "PERM", "SW")):
            raise RuntimeError(f"non-finite prediction {log_id} row {i}")
        rows.append(row)
    return rows


def write_result(path: Path, result_data, model_name: str, version: str = "3.0"):
    out = {"modelId": "", "modelName": model_name, "version": version, "resultData": result_data}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
