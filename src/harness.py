"""Shared data/feature/fold/metric utilities for the v3 experiments.

Reuses the v2 loader and FeatureBuilder unchanged so that every experiment is
measured against the same 113/237-dimensional inputs and the same well folds.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ref" / "solution_v2"))
from src.data import TARGET_COLUMNS, load_split  # noqa: E402
from src.features import FeatureBuilder  # noqa: E402

DEFAULTS = {"POR": 0.1, "PERM": 0.01, "SW": 99.9}
BOUNDS = {"POR": (0.0, 50.0), "PERM": (0.01, 10000.0), "SW": (8.305, 99.9)}
V2_CAL = {
    "POR": {"threshold": 0.2613781234057015, "scale": 1.0, "offset": 0.0},
    "PERM": {"threshold": 0.37384488065262944, "scale": 1.03, "offset": 0.0},
    "SW": {"threshold": 0.08702066266867681, "scale": 1.05, "offset": -3.0},
}
WEIGHTS = {"POR": 0.3, "PERM": 0.35, "SW": 0.35}
N_FOLDS = 5
RAW_COLS = ["GR", "PE", "SP", "CAL", "AC", "DEN", "CNL", "RXO", "RT", "DEVI", "AZIM", "BIT"]


def default_mask(values: np.ndarray, target: str) -> np.ndarray:
    return values >= DEFAULTS["SW"] if target == "SW" else values <= DEFAULTS[target]


def acc_vec(pred: np.ndarray, true: np.ndarray, target: str) -> np.ndarray:
    eps = 1e-5
    if target == "PERM":
        ratio = np.maximum(pred / np.maximum(true, eps), eps)
        return np.maximum(0.0, 1.0 - np.abs(np.log10(ratio)))
    tol = 0.08 if target == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(pred - true) / (tol * (np.abs(true) + eps)))


def score(pred: np.ndarray, true: np.ndarray, target: str) -> float:
    return float(acc_vec(pred, true, target).mean())


def total(scores: dict) -> float:
    return 100.0 * sum(WEIGHTS[t] * scores[t] for t in TARGET_COLUMNS)


def breakdown(pred, true, target, wells, folds) -> dict:
    a = acc_vec(pred, true, target)
    d = default_mask(true, target)
    per_well = {int(w): float(a[wells == w].mean()) for w in np.unique(wells)}
    pw = np.array(list(per_well.values()))
    return {
        "micro": float(a.mean()),
        "macro": float(pw.mean()),
        "worst5_mean": float(np.sort(pw)[:5].mean()),
        "fold_scores": [float(a[folds == f].mean()) for f in range(N_FOLDS)],
        "default_seg": float(a[d].mean()),
        "nondefault_seg": float(a[~d].mean()),
        "nondefault_frac": float((~d).mean()),
        "per_well": per_well,
    }


def paired_well_bootstrap(acc_new, acc_old, wells, n_boot=2000, seed=941):
    """Paired bootstrap over wells of the micro-score difference."""
    ws = np.unique(wells)
    cnt = np.array([(wells == w).sum() for w in ws], dtype=np.float64)
    d = np.array([(acc_new[wells == w] - acc_old[wells == w]).sum() for w in ws])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ws), (n_boot, len(ws)))
    boot = d[idx].sum(1) / cnt[idx].sum(1)
    return {"delta_micro": float(d.sum() / cnt.sum()), "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "well_wins": int((d > 0).sum()), "wells": int(len(ws))}


def load_or_build_cache(data_dir: Path, cache_dir: Path) -> dict:
    """Build (once) and load the training/test feature matrices used by v2."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        out = {k: np.load(cache_dir / f"{k}.npy") for k in meta["arrays"]}
        out["meta"] = meta
        return out
    t0 = time.time()
    train_records = load_split(data_dir, "train")
    test_records = load_split(data_dir, "test")
    builder = FeatureBuilder().fit([f for _, f in train_records])
    tr_base, tr_enh, tr_y, tr_well, tr_depth, tr_raw, tr_pos = [], [], [], [], [], [], []
    for wi, (wid, frame) in enumerate(train_records):
        valid = frame[TARGET_COLUMNS].notna().all(axis=1).to_numpy()
        m = builder.transform(frame)
        tr_base.append(m.base[valid]); tr_enh.append(m.enhanced[valid])
        tr_y.append(frame.loc[valid, TARGET_COLUMNS].to_numpy(np.float64))
        tr_well.append(np.full(valid.sum(), wi, np.int32))
        tr_depth.append(frame.loc[valid, "DEPTH"].to_numpy(np.float64))
        tr_raw.append(frame.loc[valid, RAW_COLS].to_numpy(np.float64))
        tr_pos.append(np.flatnonzero(valid).astype(np.int32))
    te_base, te_enh, te_well, te_depth, te_raw = [], [], [], [], []
    for wi, (wid, frame) in enumerate(test_records):
        m = builder.transform(frame)
        te_base.append(m.base); te_enh.append(m.enhanced)
        te_well.append(np.full(len(frame), wi, np.int32))
        te_depth.append(frame["DEPTH"].to_numpy(np.float64))
        te_raw.append(frame[RAW_COLS].to_numpy(np.float64))
    arrays = {
        "tr_base": np.concatenate(tr_base), "tr_enh": np.concatenate(tr_enh),
        "tr_y": np.concatenate(tr_y), "tr_well": np.concatenate(tr_well),
        "tr_depth": np.concatenate(tr_depth), "tr_raw": np.concatenate(tr_raw), "tr_pos": np.concatenate(tr_pos),
        "te_base": np.concatenate(te_base), "te_enh": np.concatenate(te_enh),
        "te_well": np.concatenate(te_well), "te_depth": np.concatenate(te_depth),
        "te_raw": np.concatenate(te_raw),
    }
    folds = np.full(len(arrays["tr_y"]), -1, np.int8)
    for f, (_, vi) in enumerate(GroupKFold(N_FOLDS).split(arrays["tr_y"], groups=arrays["tr_well"])):
        folds[vi] = f
    arrays["tr_fold"] = folds
    for k, v in arrays.items():
        np.save(cache_dir / f"{k}.npy", v)
    meta = {
        "arrays": list(arrays.keys()),
        "train_wells": [w for w, _ in train_records],
        "test_wells": [w for w, _ in test_records],
        "base_feature_names": builder.base_feature_names_,
        "enhanced_feature_names": builder.enhanced_feature_names_,
        "raw_cols": RAW_COLS,
        "medians": builder.medians_,
        "build_seconds": time.time() - t0,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    arrays["meta"] = meta
    return arrays


def make_por_bins(values: np.ndarray, bins: int = 80):
    transformed = np.log1p(values)
    non_default = values > DEFAULTS["POR"]
    edges = np.unique(np.quantile(transformed[non_default], np.linspace(0.0, 1.0, bins + 1)))
    k = len(edges) - 1
    labels = np.clip(np.searchsorted(edges, transformed, side="right") - 1, 0, k - 1).astype(np.int32)
    centers = np.array([np.median(transformed[non_default & (labels == c)]) for c in range(k)])
    return non_default, labels, centers, edges


def quantile_bins(values_t: np.ndarray, bins: int):
    """Generic quantile binning in an already-transformed space. Returns labels, centers(transformed), edges."""
    edges = np.unique(np.quantile(values_t, np.linspace(0.0, 1.0, bins + 1)))
    k = len(edges) - 1
    labels = np.clip(np.searchsorted(edges, values_t, side="right") - 1, 0, k - 1).astype(np.int32)
    centers = np.array([np.median(values_t[labels == c]) for c in range(k)])
    return labels, centers, edges


def kernel_matrix(actions: np.ndarray, support: np.ndarray, target: str) -> np.ndarray:
    """K[i, j] = official accuracy of action a_i when truth is y_j."""
    a, y = actions[:, None], support[None, :]
    if target == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(a, 1e-5)) - np.log10(np.maximum(y, 1e-5))))
    tol = 0.08 if target == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(a - y) / (tol * (np.abs(y) + 1e-5)))


def utility_action(prob: np.ndarray, support: np.ndarray, target: str, actions: np.ndarray | None = None, chunk: int = 8192) -> np.ndarray:
    """Bayes action for the official kernel over a discrete predictive distribution."""
    acts = support if actions is None else actions
    kern = kernel_matrix(acts, support, target)  # (n_actions, n_support)
    out = np.empty(len(prob))
    for s in range(0, len(prob), chunk):
        blk = prob[s:s + chunk]
        out[s:s + len(blk)] = acts[np.argmax(blk @ kern.T, axis=1)]
    return out
