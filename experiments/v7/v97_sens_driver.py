"""Driver run INSIDE the v97 runtime dir (self-contained; avoids module name
collisions with the main workspace's predict.py/src package).

Honest per-fold OOF drift sensitivity for v97's SW head:
each train well is predicted by ONLY the fold bundle that held it out.
"""
from __future__ import annotations

import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

RT = Path(__file__).resolve().parent
sys.path.insert(0, str(RT / "experiments/pure_dl"))
sys.path.insert(0, str(RT / "package"))
import predict as v97pred  # noqa: E402
from src.data import read_curve  # noqa: E402  (v97's own loader)

MAIN = Path("<PROJECT_ROOT>")
MISSING = -99999.0
CURVES = ["GR", "PE", "SP", "CAL", "AC", "DEN", "CNL", "RXO", "RT"]
CURVE_W = {"GR": 1.0, "PE": 1.3, "SP": 1.0, "CAL": 1.6, "AC": 0.9, "DEN": 1.0,
           "CNL": 0.9, "RXO": 1.2, "RT": 1.4}
LOG_CURVES = ("RT", "RXO")


def rng_for(well: str, tag: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(f"{well}:{tag}".encode()) & 0xFFFFFFFF)


def perturb_frame(frame, well, tag, iqr, bounds, sigma_lo, sigma_hi):
    rng = rng_for(well, tag)
    sigma_w = rng.uniform(sigma_lo, sigma_hi)
    out = frame.copy()
    for c in CURVES:
        s = out[c].to_numpy(np.float64).copy()
        valid = s > MISSING / 2
        z = rng.normal(0.0, sigma_w * CURVE_W[c])
        scale = float(rng.normal(1.0, 0.04))
        sv = s[valid]
        sv = sv * scale + z * iqr[c]
        if c in LOG_CURVES:
            dz = rng.normal(0.0, 0.18 * sigma_w * CURVE_W[c])
            sv = 10.0 ** (np.log10(np.maximum(sv, 1e-3)) + dz)
        lo, hi = bounds[c]
        s[valid] = np.clip(sv, lo, hi)
        out[c] = s
    return out


def sw_score(pred, true):
    eps = 1e-5
    return float(np.mean(np.maximum(0.0, 1.0 - np.abs(pred - true) / (0.05 * (np.abs(true) + eps)))))


def main():
    t0 = time.time()
    z = np.load(MAIN / "dl_artifacts/runtime__v97__models__oof_predictions.npz")
    well_arr, fold_arr, npz_pred = z["well"], z["fold"], z["pred"]
    meta = json.load(open(MAIN / "cache/meta.json"))
    wells = meta["train_wells"]
    labels = ["POR", "PERM", "SW"]
    records = {w: read_curve(MAIN / "data/train" / f"{w}.txt") for w in wells}

    iqr, pools = {}, {}
    for c in CURVES:
        allv = np.concatenate([records[w][c].to_numpy(np.float64) for w in wells])
        allv = allv[allv > MISSING / 2]
        pools[c] = allv
        q75, q25 = np.percentile(allv, [75, 25])
        iqr[c] = max(float(q75 - q25), 1e-6)
    FLOOR = {"GR": 1.0, "RT": 0.05, "RXO": 0.05, "DEN": 1.6}
    bounds = {}
    for c in CURVES:
        q_lo, q_hi = np.percentile(pools[c], [0.1, 99.9])
        bounds[c] = (max(FLOOR.get(c, -np.inf), float(q_lo)), float(q_hi) + 0.5 * iqr[c])

    predictors = {f: v97pred.Predictor(model_dir=RT / "models", device="cuda", folds=(f + 1,))
                  for f in range(5)}
    print(f"[init] 5 single-fold predictors ({time.time()-t0:.0f}s)", flush=True)

    ys, ps_o, ps_d = [], [], []
    max_dev = 0.0
    for wi, w in enumerate(wells):
        rows = np.flatnonzero(well_arr == wi)
        f0 = int(fold_arr[rows[0]])
        frame = records[w]
        valid = frame[labels].notna().all(axis=1).to_numpy()
        pf = perturb_frame(frame, w, "eval", iqr, bounds, 0.03, 0.20)
        p = predictors[f0]
        out_o = p.predict_well(frame)
        out_d = p.predict_well(pf)
        max_dev = max(max_dev, float(np.max(np.abs(out_o[valid, 2] - npz_pred[rows, 2]))))
        ys.extend(float(v) for v in frame.loc[valid, "SW"])
        ps_o.extend(float(v) for v in out_o[valid, 2])
        ps_d.extend(float(v) for v in out_d[valid, 2])
        if (wi + 1) % 20 == 0:
            print(f"[{wi+1}/80] ({time.time()-t0:.0f}s)", flush=True)

    y = np.array(ys)
    clean = sw_score(np.array(ps_o), y) * 100
    drift = sw_score(np.array(ps_d), y) * 100
    report = {
        "v97_sw_oof_clean": round(clean, 4),
        "v97_sw_oof_drift": round(drift, 4),
        "drift_delta": round(drift - clean, 4),
        "npz_reproduction_max_dev": max_dev,
        "reference_v97_sw_clean_oof": 75.5354,
        "note": "honest per-fold OOF (single held-out bundle per well); same eval perturbations as drift_sw_train",
        "seconds": round(time.time() - t0, 1),
    }
    (MAIN / "outputs/drift_sw/v97_oof_sensitivity.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
