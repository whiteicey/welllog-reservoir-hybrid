"""Extra features beyond the v2 FeatureBuilder, built only from input curves of one well.

- lagged raw curves (±1, ±2, ±3 samples) for the porosity-sensitive logs
- short-window medians (3, 5) and their differences from the point value
- local depth-derivative sign features
All are computed on the full well before any label filtering, then subset to labeled rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ref" / "solution_v2"))
from src.data import load_split, TARGET_COLUMNS  # noqa: E402

LAG_CURVES = ["GR", "AC", "DEN", "CNL", "RT", "RXO", "PE", "SP"]
LAGS = (-3, -2, -1, 1, 2, 3)


def well_extra_features(frame: pd.DataFrame, medians: dict) -> pd.DataFrame:
    cols = {}
    f = frame.copy()
    for c in LAG_CURVES:
        s = f[c]
        if s.isna().any():
            s = s.interpolate(limit_direction="both").fillna(medians.get(c, 0.0))
        if c in ("RT", "RXO"):
            s = np.log10(s.clip(lower=0.01))
        s = s.astype(np.float64)
        for lag in LAGS:
            cols[f"{c}_LAG{lag:+d}"] = (s.shift(lag) - s).bfill().ffill()
        for w in (3, 5):
            med = s.rolling(w, center=True, min_periods=1).median()
            cols[f"{c}_MED{w}"] = med
            cols[f"{c}_MED{w}_RESID"] = s - med
        d1 = s.diff().fillna(0.0)
        cols[f"{c}_D1"] = d1
        cols[f"{c}_D1_NEXT"] = (-s.diff(-1)).fillna(0.0)
        cols[f"{c}_CURV"] = (s.shift(-1) - 2 * s + s.shift(1)).fillna(0.0)
    # neighbourhood density-neutron separation and its lags (lithology/gas indicator)
    den = f["DEN"].interpolate(limit_direction="both").fillna(medians.get("DEN", 2.5))
    cnl = f["CNL"].interpolate(limit_direction="both").fillna(medians.get("CNL", 20.0))
    dphi = (2.65 - den) / 1.65 * 100.0
    sep = cnl - dphi
    cols["DN_SEP"] = sep
    for lag in (-2, -1, 1, 2):
        cols[f"DN_SEP_LAG{lag:+d}"] = (sep.shift(lag) - sep).bfill().ffill()
    # long-window zone descriptors (formation-scale context: 60 m and 150 m)
    for c in ("GR", "SP", "DEN", "AC", "CNL", "RT", "PE"):
        s = f[c]
        if s.isna().any():
            s = s.interpolate(limit_direction="both").fillna(medians.get(c, 0.0))
        if c == "RT":
            s = np.log10(s.clip(lower=0.01))
        s = s.astype(np.float64)
        for w in (601, 1501):
            r = s.rolling(w, center=True, min_periods=50)
            med = r.median().bfill().ffill()
            cols[f"{c}_ZMED{w}"] = med
            cols[f"{c}_ZSTD{w}"] = r.std(ddof=0).bfill().ffill()
            cols[f"{c}_ZRESID{w}"] = s - med
            cols[f"{c}_ZQ10_{w}"] = r.quantile(0.1).bfill().ffill()
            cols[f"{c}_ZQ90_{w}"] = r.quantile(0.9).bfill().ffill()
    depth = f["DEPTH"].astype(np.float64)
    cols["DEPTH_FROM_TOP"] = depth - depth.min()
    cols["DEPTH_TO_BOTTOM"] = depth.max() - depth
    out = pd.DataFrame(cols).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.astype(np.float32)


def build_extra(data_dir: Path, cache_dir: Path, medians: dict):
    p_tr, p_te = cache_dir / "tr_extra.npy", cache_dir / "te_extra.npy"
    if p_tr.exists() and p_te.exists():
        return np.load(p_tr), np.load(p_te), (cache_dir / "extra_names.txt").read_text().split("\n")
    tr, te, names = [], [], None
    for _, frame in load_split(data_dir, "train"):
        valid = frame[TARGET_COLUMNS].notna().all(axis=1).to_numpy()
        e = well_extra_features(frame, medians)
        names = e.columns.tolist()
        tr.append(e.to_numpy()[valid])
    for _, frame in load_split(data_dir, "test"):
        te.append(well_extra_features(frame, medians).to_numpy())
    tr, te = np.concatenate(tr), np.concatenate(te)
    np.save(p_tr, tr); np.save(p_te, te)
    (cache_dir / "extra_names.txt").write_text("\n".join(names))
    return tr, te, names
