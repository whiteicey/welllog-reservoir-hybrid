"""Well-to-well similarity and DTW-based stratigraphic alignment features (label-free inputs only).

Similarity matrices over the 80 training + 10 test wells:
  median-curve Euclidean distance (as in diag_wells), Perez-Cruz kNN KL divergence on 5 robust-scaled curves,
  DTW on smoothed within-well-scaled GR (0.5 m grid, psi-relaxed ends), DTW on a 4-curve composite.
Alignment features for every sample (K = 8 nearest wells by GR-DTW, warping path -> neighbour depth ->
neighbour interpreted labels): default rates, mean/std of neighbour POR, SW, log10 PERM, state fractions,
DTW distances, and the aligned depth in a reference (medoid) well.
Cross-fitting: for CV fold f the neighbour pool of every well is the training-fold wells (minus itself);
`full` uses all training wells (final training / inference).
Outputs in cache/: sim_*.npy, aln_fold{f}_{tr,te}.npy, aln_full_{tr,te}.npy, aln_names.txt, aln_meta.json
and outputs/dtw_align_report.json (label-copy baseline scores, medoid, neighbour statistics).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dtaidistance import dtw
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ref" / "solution_v2"))
from src.data import load_split, TARGET_COLUMNS  # noqa: E402
from harness import load_or_build_cache, acc_vec, N_FOLDS  # noqa: E402

CACHE = ROOT / "cache"; OUT = ROOT / "outputs"
STEP = 5          # 0.1 m rows -> 0.5 m grid
PSI = 300         # relaxed ends (150 m)
K = 8
KL_CURVES = ["GR", "DEN", "CNL", "RT", "AC"]
NAMES = ["ALN_POR_DEF", "ALN_SW_DEF", "ALN_PERM_DEF", "ALN_STATE_HC", "ALN_STATE_WATER", "ALN_POR_MEAN", "ALN_POR_STD",
         "ALN_SW_MEAN", "ALN_SW_STD", "ALN_LOGPERM_MEAN", "ALN_LOGPERM_STD", "ALN_N_LABELED", "ALN_DTW_MEAN", "ALN_DTW_MIN",
         "ALN_REF_DEPTH", "ALN_REF_REL", "ALN_KL_MEAN"]


def coarse_signal(frame, col):
    s = frame[col].interpolate(limit_direction="both")
    if s.isna().all():
        s = s.fillna(0.0)
    if col in ("RT", "RXO"):
        s = np.log10(s.clip(lower=0.01))
    v = pd.Series(s.to_numpy(np.float64)).rolling(11, center=True, min_periods=1).median().to_numpy()
    med, q1, q3 = np.nanmedian(v), np.nanpercentile(v, 25), np.nanpercentile(v, 75)
    v = (v - med) / max(q3 - q1, 1e-3)
    n = len(v) // STEP * STEP
    return np.nan_to_num(v[:n].reshape(-1, STEP).mean(1))


def coarse_depth(frame):
    d = frame["DEPTH"].to_numpy(np.float64); n = len(d) // STEP * STEP
    return d[:n].reshape(-1, STEP).mean(1)


def kl_perez_cruz(x, y, k=5):
    n, d = x.shape; m = y.shape[0]
    r = cKDTree(x).query(x, k=k + 1)[0][:, -1]
    s = cKDTree(y).query(x, k=k)[0][:, -1]
    r = np.maximum(r, 1e-9); s = np.maximum(s, 1e-9)
    return float(d * np.mean(np.log(s / r)) + np.log(m / (n - 1)))


def main():
    t0 = time.time()
    train = load_split(ROOT / "data", "train"); test = load_split(ROOT / "data", "test")
    frames = [f for _, f in train] + [f for _, f in test]
    ids = [w for w, _ in train] + [w for w, _ in test]
    n_tr = len(train); W = len(frames)
    sig_gr = [coarse_signal(f, "GR") for f in frames]
    sig_comp = [np.mean([coarse_signal(f, c) for c in ("GR", "DEN", "CNL", "RT")], 0) for f in frames]
    dep_c = [coarse_depth(f) for f in frames]
    lens = np.array([len(s) for s in sig_gr])
    print("signals ready", W, "wells, coarse lengths", lens.min(), lens.max(), f"{time.time() - t0:.0f}s", flush=True)
    # ---------- similarity matrices ----------
    def dtw_matrix(series):
        M = dtw.distance_matrix_fast(series, psi=PSI)
        M = np.where(np.isfinite(M), M, np.nan); M = np.fmin(M, M.T); np.fill_diagonal(M, 0.0)
        norm = np.sqrt(lens[:, None] + lens[None, :])
        return M / norm
    S_dtw_gr = dtw_matrix(sig_gr); print("dtw GR", f"{time.time() - t0:.0f}s", flush=True)
    S_dtw_comp = dtw_matrix(sig_comp); print("dtw composite", f"{time.time() - t0:.0f}s", flush=True)
    # KL on 5 curves, robust-scaled with pooled training statistics, 3000 points per well
    rng = np.random.default_rng(0)
    def curves(f):
        X = np.c_[[f[c].interpolate(limit_direction="both").to_numpy(np.float64) for c in KL_CURVES]].T
        X[:, KL_CURVES.index("RT")] = np.log10(np.clip(X[:, KL_CURVES.index("RT")], 0.01, None))
        return X
    Xs = [curves(f) for f in frames]
    pool = np.concatenate(Xs[:n_tr]); med = np.nanmedian(pool, 0); iqr = np.nanpercentile(pool, 75, 0) - np.nanpercentile(pool, 25, 0)
    Xs = [np.nan_to_num((X - med) / np.maximum(iqr, 1e-3)) for X in Xs]
    Xs = [X[rng.choice(len(X), min(3000, len(X)), replace=False)] for X in Xs]
    S_kl = np.zeros((W, W))
    for i in range(W):
        for j in range(i + 1, W):
            S_kl[i, j] = S_kl[j, i] = max(kl_perez_cruz(Xs[i], Xs[j]) + kl_perez_cruz(Xs[j], Xs[i]), 0.0)
    print("KL", f"{time.time() - t0:.0f}s", flush=True)
    # median-curve Euclidean (diag_wells criterion)
    raw_cols = ["GR", "PE", "SP", "CAL", "AC", "DEN", "CNL", "RXO", "RT", "DEVI", "AZIM", "BIT"]
    st = []
    for f in frames:
        X = f[raw_cols].to_numpy(np.float64).copy()
        for c in ("RXO", "RT"):
            X[:, raw_cols.index(c)] = np.log10(np.clip(X[:, raw_cols.index(c)], 0.01, None))
        st.append(np.r_[np.nanmedian(X, 0), np.nanpercentile(X, 90, 0) - np.nanpercentile(X, 10, 0)])
    st = np.nan_to_num(np.array(st)); st = (st - st[:n_tr].mean(0)) / (st[:n_tr].std(0) + 1e-9)
    S_med = np.sqrt(((st[:, None, :] - st[None, :, :]) ** 2).sum(-1))
    for name, S in (("dtw_gr", S_dtw_gr), ("dtw_comp", S_dtw_comp), ("kl", S_kl), ("median", S_med)):
        np.save(CACHE / f"sim_{name}.npy", S)
    json.dump({"wells": ids, "n_train": n_tr, "coarse_step_m": 0.5, "psi": PSI, "K": K}, open(CACHE / "well_order.json", "w"))
    # ---------- reference (medoid) well among training wells by GR-DTW ----------
    ref = int(np.argmin(S_dtw_gr[:n_tr, :n_tr].sum(1)))
    print("medoid reference well", ids[ref][:8], flush=True)
    # ---------- labels per well (full rows) ----------
    fine_depth = [f["DEPTH"].to_numpy(np.float64) for f in frames]
    lab = []
    for f in frames:
        if all(t in f for t in TARGET_COLUMNS):
            y = f[TARGET_COLUMNS].to_numpy(np.float64)
        else:
            y = np.full((len(f), 3), np.nan)
        lab.append(y)
    path_cache = {}

    def mapped_depth(w, v):
        """fine-row depths of well w mapped into well v's depth axis via the GR warping path (cached)."""
        key = (w, v)
        if key not in path_cache:
            _, paths = dtw.warping_paths_fast(sig_gr[w], sig_gr[v], psi=PSI)
            p = np.array(dtw.best_path(paths))
            i_idx, j_idx = p[:, 0], p[:, 1]
            cnt = np.bincount(i_idx, minlength=len(sig_gr[w])).astype(float)
            jm = np.bincount(i_idx, weights=j_idx, minlength=len(sig_gr[w])) / np.maximum(cnt, 1)
            jm[cnt == 0] = np.nan
            jm = pd.Series(jm).interpolate(limit_direction="both").to_numpy()
            mapped_c = np.interp(jm, np.arange(len(dep_c[v])), dep_c[v])
            path_cache[key] = np.interp(fine_depth[w], dep_c[w], mapped_c)
        return path_cache[key]

    def neighbour_labels(w, v):
        md = mapped_depth(w, v)
        idx = np.clip(np.searchsorted(fine_depth[v], md), 0, len(fine_depth[v]) - 1)
        return lab[v][idx]

    def features_for(w, pool_wells):
        d = S_dtw_gr[w, pool_wells]; order = np.argsort(d)[:K]; nb = [pool_wells[i] for i in order]; dn = d[order]
        Y = np.stack([neighbour_labels(w, v) for v in nb])  # (K, n, 3)
        por, perm, sw = Y[..., 0], Y[..., 1], Y[..., 2]
        labeled = np.isfinite(por)
        por_def = np.where(labeled, por <= 0.1, np.nan); sw_def = np.where(labeled, sw >= 99.9, np.nan); perm_def = np.where(labeled, perm <= 0.01, np.nan)
        hc = np.where(labeled, (por > 0.1) & (sw < 99.9) & (perm > 0.01), np.nan); water = np.where(labeled, (por > 0.1) & (sw >= 99.9), np.nan)
        por_nd = np.where(labeled & (por > 0.1), por, np.nan); sw_nd = np.where(labeled & (sw < 99.9), sw, np.nan)
        lp_nd = np.where(labeled & (perm > 0.01), np.log10(np.where(perm > 0.01, perm, 1.0)), np.nan)
        with np.errstate(all="ignore"):
            F = np.c_[np.nanmean(por_def, 0), np.nanmean(sw_def, 0), np.nanmean(perm_def, 0), np.nanmean(hc, 0), np.nanmean(water, 0),
                      np.nanmean(por_nd, 0), np.nanstd(por_nd, 0), np.nanmean(sw_nd, 0), np.nanstd(sw_nd, 0), np.nanmean(lp_nd, 0), np.nanstd(lp_nd, 0),
                      labeled.mean(0), np.full(Y.shape[1], dn.mean()), np.full(Y.shape[1], dn.min())]
        refd = mapped_depth(w, ref) if w != ref else fine_depth[w]
        ref_rel = (refd - fine_depth[ref].min()) / max(fine_depth[ref].max() - fine_depth[ref].min(), 1.0)
        klm = np.full(Y.shape[1], S_kl[w, nb].mean())
        return np.c_[F, refd, ref_rel, klm].astype(np.float32), nb, dn

    C = load_or_build_cache(ROOT / "data", CACHE)
    tr_well, tr_pos, te_well, folds = C["tr_well"], C["tr_pos"], C["te_well"], C["tr_fold"]
    well_fold = {int(w): int(folds[tr_well == w][0]) for w in np.unique(tr_well)}
    meta = {"reference_well": ids[ref], "reference_index": ref, "names": NAMES, "K": K, "psi": PSI, "neighbours": {}}
    def build(pool_for):
        Ftr = np.zeros((len(tr_well), len(NAMES)), np.float32); Fte = np.zeros((len(te_well), len(NAMES)), np.float32); nbs = {}
        for w in range(W):
            pool = [v for v in pool_for(w) if v != w]
            F, nb, dn = features_for(w, pool); nbs[ids[w]] = [(ids[v], float(x)) for v, x in zip(nb, dn)]
            if w < n_tr:
                rows = np.flatnonzero(tr_well == w); Ftr[rows] = F[tr_pos[rows]]
            else:
                Fte[te_well == (w - n_tr)] = F
        return Ftr, Fte, nbs
    for f in range(N_FOLDS):
        pool = [v for v in range(n_tr) if well_fold[v] != f]
        Ftr, Fte, nbs = build(lambda w, pool=pool: pool)
        np.save(CACHE / f"aln_fold{f}_tr.npy", Ftr); np.save(CACHE / f"aln_fold{f}_te.npy", Fte); meta["neighbours"][f"fold{f}"] = nbs
        print("fold", f, "features done", f"{time.time() - t0:.0f}s", flush=True)
    Ftr, Fte, nbs = build(lambda w: list(range(n_tr)))
    np.save(CACHE / "aln_full_tr.npy", Ftr); np.save(CACHE / "aln_full_te.npy", Fte); meta["neighbours"]["full"] = nbs
    (CACHE / "aln_names.txt").write_text("\n".join(NAMES)); json.dump(meta, open(CACHE / "aln_meta.json", "w"), indent=1)
    # ---------- label-copy baseline (cross-fitted OOF) ----------
    y = C["tr_y"]; A = np.zeros((len(y), len(NAMES)), np.float32)
    for f in range(N_FOLDS):
        A[folds == f] = np.load(CACHE / f"aln_fold{f}_tr.npy")[folds == f]
    A = pd.DataFrame(A, columns=NAMES)
    rep = {"reference_well": ids[ref], "seconds_to_features": time.time() - t0,
           "dtw_gr_nn_distance_train": {"q10": float(np.quantile(np.sort(S_dtw_gr[:n_tr, :n_tr], 1)[:, 1], .1)), "q50": float(np.quantile(np.sort(S_dtw_gr[:n_tr, :n_tr], 1)[:, 1], .5)), "q90": float(np.quantile(np.sort(S_dtw_gr[:n_tr, :n_tr], 1)[:, 1], .9))},
           "dtw_gr_nn_distance_test": [float(np.sort(S_dtw_gr[n_tr + i, :n_tr])[0]) for i in range(W - n_tr)],
           "criteria_agreement_top8": {}}
    for a, Sa in (("dtw_gr", S_dtw_gr), ("dtw_comp", S_dtw_comp), ("kl", S_kl), ("median", S_med)):
        for b, Sb in (("dtw_gr", S_dtw_gr), ("dtw_comp", S_dtw_comp), ("kl", S_kl), ("median", S_med)):
            if a < b:
                ov = []
                for w in range(n_tr):
                    na = set(np.argsort(np.where(np.arange(W) == w, np.inf, Sa[w]))[:8]); nbb = set(np.argsort(np.where(np.arange(W) == w, np.inf, Sb[w]))[:8])
                    ov.append(len(na & nbb) / 8)
                rep["criteria_agreement_top8"][f"{a}|{b}"] = float(np.mean(ov))
    med_por = float(np.median(y[y[:, 0] > 0.1, 0])); med_sw = float(np.median(y[y[:, 2] < 99.9, 2])); med_lp = float(np.median(np.log10(y[y[:, 1] > 0.01, 1])))
    por = np.where(A.ALN_POR_DEF.fillna(0) >= 0.5, 0.1, A.ALN_POR_MEAN.fillna(med_por))
    sw = np.where(A.ALN_SW_DEF.fillna(0) >= 0.5, 99.9, A.ALN_SW_MEAN.fillna(med_sw))
    perm = np.where(A.ALN_PERM_DEF.fillna(0) >= 0.5, 0.01, 10 ** A.ALN_LOGPERM_MEAN.fillna(med_lp))
    d = {"POR": y[:, 0] <= 0.1, "PERM": y[:, 1] <= 0.01, "SW": y[:, 2] >= 99.9}
    rep["label_copy_baseline"] = {}
    for t, p, i in (("POR", por, 0), ("PERM", perm, 1), ("SW", sw, 2)):
        a = acc_vec(p, y[:, i], t)
        rep["label_copy_baseline"][t] = {"micro": float(a.mean()), "default_seg": float(a[d[t]].mean()), "nondefault_seg": float(a[~d[t]].mean()),
                                        "gate_accuracy": float(((p == (0.1 if t == "POR" else 0.01 if t == "PERM" else 99.9)) == d[t]).mean())}
    rep["label_copy_baseline"]["TOTAL"] = 100 * (0.3 * rep["label_copy_baseline"]["POR"]["micro"] + 0.35 * rep["label_copy_baseline"]["PERM"]["micro"] + 0.35 * rep["label_copy_baseline"]["SW"]["micro"])
    rep["aln_feature_missing_rate"] = {n: float(A[n].isna().mean()) for n in NAMES}
    (OUT / "dtw_align_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "aln_feature_missing_rate"}, indent=1))


if __name__ == "__main__":
    main()
