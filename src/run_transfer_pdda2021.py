"""Transfer validation of the v3 methodology on the public SPWLA PDDA 2021 (Volve) benchmark.

Data: 9 training wells (labels PHIF, SW, VSH on ~42k of 319k rows), 4 blind wells with released truth.
Official metric there: RMSE over the three 0-1 targets. We evaluate
  E0  Random-Forest tutorial baseline on the five common logs (sanity check against the published 0.117)
  E1  CatBoost RMSE regressors on our 113+181 features (well-grouped 9-fold CV + blind wells)
  E2  CatBoost 60-bin distribution models decoded (a) by the mean (RMSE-optimal), (b) by the Bayes action of
      the tolerance kernel  -> the same model serves both metrics; shows decoding must follow the metric
  E3  SW = 1 (water) gate: CatBoost gate vs whole-well sequence-net gate (segmentation transfer)
  E4  identifiability audit on Volve: same-well half split vs cross-well vs nearest-well training
  E5  blind-well submission-style score (RMSE) for the fused system, comparable with the contest leaderboard
Volve columns are mapped onto the competition curve names with unit conversions (ft->m, us/ft->us/m,
inch->cm, neutron fraction->%). SP, DEVI, AZIM, CASE do not exist there and are left missing (constant features).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("<PROJECT_ROOT>")
sys.path.insert(0, str(ROOT / "package"))
from src.feature import FeatureBuilder, well_extra_features  # noqa: E402
from src.utils import kernel_matrix  # noqa: E402

OUT = ROOT / "outputs" / "transfer_pdda2021"; OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "external" / "pdda2021"
TARGETS = ["PHIF", "SW", "VSH"]
SEED = 42


# ----------------------------------------------------------------------------- data adapter
def adapt(df: pd.DataFrame) -> pd.DataFrame:
    d = df.replace(-9999.0, np.nan)
    out = pd.DataFrame({
        "DEPTH": d["DEPTH"] * 0.3048, "GR": d["GR"], "PE": d["PEF"], "SP": np.nan, "CAL": d["CALI"] * 2.54,
        "AC": d["DTC"] * 3.28084, "DEN": d["DEN"], "CNL": d["NEU"] * 100.0, "RXO": d["RMED"], "RT": d["RDEP"],
        "DEVI": np.nan, "AZIM": np.nan, "BIT": d["BS"] * 2.54, "CASE": np.nan,
    })
    # the competition loader's physical ranges
    rng = {"GR": (0, 2000), "PE": (0, 20), "CAL": (5, 100), "AC": (100, 1000), "DEN": (1, 4), "CNL": (-30, 100),
           "RXO": (0, 10000), "RT": (0, 10000), "BIT": (0, 50)}
    for c, (lo, hi) in rng.items():
        out.loc[(out[c] < lo) | (out[c] > hi), c] = np.nan
    for t in TARGETS:
        if t in d:
            out[t] = d[t]
    out["WELLNUM"] = d["WELLNUM"].values
    return out


def load():
    tr = pd.read_csv(DATA / "train.csv"); te = pd.read_csv(DATA / "test.csv")
    truth = pd.read_csv(DATA / "real_test_result.csv")
    tr, te = adapt(tr), adapt(te)
    te[TARGETS] = truth[TARGETS].values
    tr_wells = [(int(w), g.drop(columns="WELLNUM").reset_index(drop=True)) for w, g in tr.groupby("WELLNUM", sort=True)]
    te_wells = [(int(w), g.drop(columns="WELLNUM").reset_index(drop=True)) for w, g in te.groupby("WELLNUM", sort=True)]
    return tr_wells, te_wells


# ----------------------------------------------------------------------------- metrics
def rmse3(pred, true):
    return float(np.sqrt(np.mean((np.clip(pred, 0, 1) - true) ** 2)))


def rmse_per(pred, true):
    return {t: float(np.sqrt(np.mean((np.clip(pred[:, i], 0, 1) - true[:, i]) ** 2))) for i, t in enumerate(TARGETS)}


def tol_acc(pred, true, tol):
    return float(np.mean(np.maximum(0.0, 1.0 - np.abs(pred - true) / (tol * (np.abs(true) + 1e-7)))))


def quantile_bins(v, k):
    edges = np.unique(np.quantile(v, np.linspace(0, 1, k + 1)))
    kk = len(edges) - 1
    lab = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, kk - 1)
    centers = np.array([np.median(v[lab == c]) for c in range(kk)])
    return lab, centers


def decode_mean(prob, centers):
    return prob @ centers


def decode_bayes(prob, centers, tol):
    grid = np.linspace(centers[0], centers[-1], 600)
    K = np.maximum(0.0, 1.0 - np.abs(grid[:, None] - centers[None, :]) / (tol * (np.abs(centers[None, :]) + 1e-7)))
    return grid[np.argmax(prob @ K.T, 1)]


# ----------------------------------------------------------------------------- features
def build_features(tr_wells, te_wells):
    builder = FeatureBuilder().fit([f for _, f in tr_wells])
    def mats(frames):
        X, Xe, Y, W, D = [], [], [], [], []
        for w, f in frames:
            m = builder.transform(f); e = well_extra_features(f, builder.medians_).to_numpy()
            X.append(np.c_[m.base, e]); Xe.append(np.c_[m.enhanced, e]); Y.append(f[TARGETS].to_numpy(np.float64))
            W.append(np.full(len(f), w)); D.append(f["DEPTH"].to_numpy())
        return np.concatenate(X), np.concatenate(Xe), np.concatenate(Y), np.concatenate(W), np.concatenate(D)
    return builder, mats(tr_wells), mats(te_wells)


# ----------------------------------------------------------------------------- CatBoost helpers
def cb_reg(X, y, Xp, seed=SEED, iters=800):
    from catboost import CatBoostRegressor, Pool
    m = CatBoostRegressor(iterations=iters, learning_rate=0.04, depth=6, loss_function="RMSE", random_seed=seed, task_type="GPU",
                          devices="0", max_bin=128, verbose=0, allow_writing_files=False, gpu_ram_part=0.3)
    m.fit(Pool(X, label=y)); return m.predict(Xp)


def cb_bins(X, lab, k, Xp, seed=SEED, iters=800):
    from catboost import CatBoostClassifier, Pool
    m = CatBoostClassifier(iterations=iters, learning_rate=0.04, depth=6, loss_function="MultiClass", random_seed=seed, task_type="GPU",
                           devices="0", max_bin=128, verbose=0, allow_writing_files=False, gpu_ram_part=0.3)
    m.fit(Pool(X, label=lab)); order = np.argsort(m.classes_.astype(int))
    p = m.predict_proba(Xp)[:, order]
    full = np.zeros((len(Xp), k)); full[:, m.classes_.astype(int)[order]] = p
    return full / full.sum(1, keepdims=True)


def cb_gate(X, d, Xp, seed=SEED, iters=600):
    from catboost import CatBoostClassifier, Pool
    m = CatBoostClassifier(iterations=iters, learning_rate=0.04, depth=6, loss_function="Logloss", random_seed=seed, task_type="GPU",
                           devices="0", max_bin=128, verbose=0, allow_writing_files=False, gpu_ram_part=0.3)
    m.fit(Pool(X, label=d.astype(np.int8))); return m.predict_proba(Xp)[:, 1]


# ----------------------------------------------------------------------------- sequence net (gate + bins)
def nn_train_predict(tr_frames, tr_y, te_frames, bins_centers, epochs=30, seeds=(1, 2), crop=1024):
    import torch, torch.nn.functional as F
    from src.model import WellNet, prep_well, predict_well
    CURVES = ["GR", "PE", "SP", "CAL", "AC", "DEN", "CNL", "RXO", "RT", "DEVI"]
    gst = {}
    for c in CURVES:
        v = np.concatenate([f[c].dropna().to_numpy(np.float64) for f in tr_frames])
        if c in ("RXO", "RT"):
            v = np.log10(np.clip(v, 0.01, None))
        if len(v) == 0:
            gst[c] = {"median": 0.0, "iqr": 1.0}; continue
        q1, q3 = np.percentile(v, [25, 75]); gst[c] = {"median": float(np.median(v)), "iqr": float(max(q3 - q1, 1e-3))}
    xs = [prep_well(f, gst) for f in tr_frames]; xt = [prep_well(f, gst) for f in te_frames]
    heads = {"POR": "PHIF", "PERM": "VSH", "SW": "SW"}  # head name -> Volve target
    nb = {h: len(bins_centers[t][1]) for h, t in heads.items()}
    # per-well targets: gate (SW==1), bin labels
    tg = []
    for y in tr_y:
        d = {}
        for h, t in heads.items():
            v = y[:, TARGETS.index(t)]; lab = np.isfinite(v)
            dflt = (v >= 0.999) & lab if t == "SW" else np.zeros(len(v), bool)
            edges, centers = bins_centers[t]
            b = np.clip(np.searchsorted(edges, np.where(lab, v, 0.0), side="right") - 1, 0, len(centers) - 1)
            d[h] = (np.where(lab, dflt.astype(int), -1), np.where(lab & ~dflt, b, -1))
        tg.append(d)
    device = torch.device("cuda")
    outs = None
    for seed in seeds:
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        net = WellNet(xs[0].shape[0], ch=96, drop=0.1, k=5, nbins=nb).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        steps = epochs * 60
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
        lengths = np.array([x.shape[1] for x in xs], float); pw = lengths / lengths.sum()
        for st in range(steps):
            xb, gb, cb = [], {h: [] for h in heads}, {h: [] for h in heads}
            for _ in range(12):
                i = rng.choice(len(xs), p=pw); n = xs[i].shape[1]; s = rng.integers(0, max(n - crop, 0) + 1)
                x = xs[i][:, s:s + crop]
                if x.shape[1] < crop:
                    x = np.pad(x, ((0, 0), (0, crop - x.shape[1])))
                xb.append(x)
                for h in heads:
                    g, c = tg[i][h]; g, c = g[s:s + crop], c[s:s + crop]
                    if len(g) < crop:
                        g = np.pad(g, (0, crop - len(g)), constant_values=-1); c = np.pad(c, (0, crop - len(c)), constant_values=-1)
                    gb[h].append(g); cb[h].append(c)
            xb = torch.from_numpy(np.stack(xb)).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(xb)
            loss = 0.0
            for h in heads:
                o = out[h].float(); g = torch.from_numpy(np.stack(gb[h])).to(device); c = torch.from_numpy(np.stack(cb[h])).to(device)
                lg = F.binary_cross_entropy_with_logits(o[:, 0], (g == 1).float(), reduction="none")
                loss = loss + (lg * (g >= 0)).sum() / (g >= 0).sum().clamp(min=1)
                lc = F.cross_entropy(o[:, 1:], c.clamp(min=0), reduction="none")
                loss = loss + 0.5 * (lc * (c >= 0)).sum() / (c >= 0).sum().clamp(min=1)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step(); sched.step()
        preds = [predict_well(net, x, device) for x in xt]
        if outs is None:
            outs = preds
        else:
            outs = [{h: (a[h][0] + b[h][0], a[h][1] + b[h][1]) for h in heads} for a, b in zip(outs, preds)]
    return [{h: (o[h][0] / len(seeds), o[h][1] / len(seeds)) for h in heads} for o in outs], heads


# ----------------------------------------------------------------------------- main experiment
def main():
    t0 = time.time()
    tr_wells, te_wells = load()
    builder, (Xtr, Xtr_e, Ytr, Wtr, Dtr), (Xte, Xte_e, Yte, Wte, Dte) = build_features(tr_wells, te_wells)
    lab_tr = np.isfinite(Ytr).all(1); lab_te = np.isfinite(Yte).all(1)
    print("features", Xtr.shape, "labeled train rows", lab_tr.sum(), "test rows", lab_te.sum(), f"{time.time() - t0:.0f}s", flush=True)
    R = {"rows": {"train": int(len(Ytr)), "train_labeled": int(lab_tr.sum()), "test": int(len(Yte))}}
    wells = np.unique(Wtr)
    TOL = {"PHIF": 0.08, "SW": 0.05}

    # ---------- E0: tutorial RF on five logs (DEN, GR, NEU, PEF, RDEP), drop missing ----------
    from sklearn.ensemble import RandomForestRegressor
    five_tr = np.c_[[tr_wells[i][1][c].to_numpy() for i in range(len(tr_wells)) for c in []]] if False else None
    def five(frames):
        cols = ["DEN", "GR", "CNL", "PE", "RT"]
        return np.concatenate([f[cols].to_numpy(np.float64) for _, f in frames])
    F5tr, F5te = five(tr_wells), five(te_wells)
    m5 = lab_tr & np.isfinite(F5tr).all(1); m5t = np.isfinite(F5te).all(1)
    rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=8).fit(F5tr[m5], Ytr[m5])
    p5 = np.full(Yte.shape, np.nan); p5[m5t] = rf.predict(F5te[m5t])
    # rows with missing logs: fill with training mean (the tutorial dropped them; we must score every row)
    p5[~m5t] = Ytr[m5].mean(0)
    R["E0_rf_tutorial_blind_rmse"] = {"all3": rmse3(p5, Yte), **rmse_per(p5, Yte), "rows_with_all_five_logs": float(m5t.mean())}
    print("E0", R["E0_rf_tutorial_blind_rmse"], flush=True)

    # ---------- leave-well-out CV over the 9 training wells ----------
    K = 60
    cv = {"E1_reg": np.full(Ytr.shape, np.nan), "E2_mean": np.full(Ytr.shape, np.nan), "E2_bayes": np.full(Ytr.shape, np.nan),
          "gate_cb": np.full(len(Ytr), np.nan), "gate_nn": np.full(len(Ytr), np.nan), "E2_nn_mean": np.full(Ytr.shape, np.nan), "E2_nn_bayes": np.full(Ytr.shape, np.nan)}
    for w in wells:
        tw = time.time()
        tr = lab_tr & (Wtr != w); va = Wtr == w; va_lab = va & lab_tr
        if va_lab.sum() == 0:
            continue
        bins = {}
        for i, t in enumerate(TARGETS):
            yv = Ytr[tr, i]
            if t == "SW":
                nd = yv < 0.999
                lab, centers = quantile_bins(yv[nd], K); edges = np.unique(np.quantile(yv[nd], np.linspace(0, 1, K + 1)))
                rows = np.flatnonzero(tr)[nd]
            else:
                lab, centers = quantile_bins(yv, K); edges = np.unique(np.quantile(yv, np.linspace(0, 1, K + 1)))
                rows = np.flatnonzero(tr)
            bins[t] = (edges, centers)
            # E1 regression
            cv["E1_reg"][va, i] = cb_reg(Xtr[rows], Ytr[rows, i], Xtr[va])
            # E2 distribution model
            prob = cb_bins(Xtr[rows], lab, len(centers), Xtr[va])
            cv["E2_mean"][va, i] = decode_mean(prob, centers)
            cv["E2_bayes"][va, i] = decode_bayes(prob, centers, TOL.get(t, 0.08))
        d_sw = (Ytr[:, 1] >= 0.999)
        cv["gate_cb"][va] = cb_gate(Xtr_e[tr], d_sw[tr], Xtr_e[va])
        # sequence net on whole wells (inputs everywhere, labels sparse)
        tr_frames = [f for ww, f in tr_wells if ww != w]; tr_y = [f[TARGETS].to_numpy(np.float64) for ww, f in tr_wells if ww != w]
        va_frames = [f for ww, f in tr_wells if ww == w]
        outs, heads = nn_train_predict(tr_frames, tr_y, va_frames, bins)
        o = outs[0]
        cv["gate_nn"][va] = o["SW"][0]
        for h, t in heads.items():
            i = TARGETS.index(t); centers = bins[t][1]; p = o[h][1] / o[h][1].sum(1, keepdims=True)
            cv["E2_nn_mean"][va, i] = decode_mean(p, centers); cv["E2_nn_bayes"][va, i] = decode_bayes(p, centers, TOL.get(t, 0.08))
        print({"well": int(w), "seconds": round(time.time() - tw), "labeled": int(va_lab.sum())}, flush=True)
    np.savez(OUT / "cv_oof.npz", **cv, Y=Ytr, W=Wtr, lab=lab_tr)
    m = lab_tr
    Y = Ytr[m]; d = Y[:, 1] >= 0.999
    def with_gate(cont, g, th):
        out = cont.copy(); out[:, 1] = np.where(g >= th, 1.0, cont[:, 1]); return out
    res = {}
    for name in ("E1_reg", "E2_mean", "E2_bayes", "E2_nn_mean", "E2_nn_bayes"):
        P = cv[name][m]
        res[name] = {"rmse_no_gate": rmse3(P, Y), **{f"rmse_{t}": v for t, v in rmse_per(P, Y).items()},
                     "tol_PHIF": tol_acc(P[:, 0], Y[:, 0], 0.08), "tol_SW_nondefault": tol_acc(P[~d, 1], Y[~d, 1], 0.05)}
        for gname in ("gate_cb", "gate_nn"):
            g = cv[gname][m]
            best = max(((th, rmse3(with_gate(P, g, th), Y)) for th in np.linspace(0.3, 0.9, 13)), key=lambda z: -z[1])
            res[name][f"rmse_with_{gname}"] = {"threshold": float(best[0]), "rmse": best[1]}
            res[name][f"tolSW_with_{gname}@0.5"] = tol_acc(with_gate(P, g, 0.5)[:, 1], Y[:, 1], 0.05)
    for gname in ("gate_cb", "gate_nn"):
        g = cv[gname][m]
        res[f"{gname}_accuracy@0.5"] = float(((g >= 0.5) == d).mean())
    res["SW_default_share_labeled"] = float(d.mean())
    R["CV_leave_well_out"] = res
    print(json.dumps(res, indent=1), flush=True)

    # ---------- E4: identifiability audit on Volve (LightGBM on 12 curves + depth) ----------
    import lightgbm as lgb
    raw_cols = ["GR", "PE", "CAL", "AC", "DEN", "CNL", "RXO", "RT", "BIT", "DEPTH"]
    Xr = np.concatenate([f[raw_cols].to_numpy(np.float64) for _, f in tr_wells])
    for c in ("RXO", "RT"):
        j = raw_cols.index(c); Xr[:, j] = np.log10(np.clip(Xr[:, j], 0.01, None))
    Xr = np.nan_to_num(Xr)
    aud = {}
    for i, t in enumerate(TARGETS):
        half, cross, near = [], [], []
        for w in wells:
            mw = lab_tr & (Wtr == w)
            if t == "SW":
                mw = mw & (Ytr[:, 1] < 0.999)
            if mw.sum() < 300:
                continue
            idx = np.flatnonzero(mw); h = np.arange(len(idx)) < len(idx) // 2
            mdl = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, n_jobs=8)
            p1 = mdl.fit(Xr[idx[h]], Ytr[idx[h], i]).predict(Xr[idx[~h]])
            p2 = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, n_jobs=8).fit(Xr[idx[~h]], Ytr[idx[~h], i]).predict(Xr[idx[h]])
            half.append(0.5 * (np.sqrt(np.mean((p1 - Ytr[idx[~h], i]) ** 2)) + np.sqrt(np.mean((p2 - Ytr[idx[h], i]) ** 2))))
            mo = lab_tr & (Wtr != w)
            if t == "SW":
                mo = mo & (Ytr[:, 1] < 0.999)
            pc = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, n_jobs=8).fit(Xr[mo], Ytr[mo, i]).predict(Xr[idx])
            cross.append(np.sqrt(np.mean((pc - Ytr[idx, i]) ** 2)))
            # nearest well by median curve statistics
            st = {ww: np.nanmedian(Xr[Wtr == ww], 0) for ww in wells}
            dist = sorted(((np.nansum((st[ww] - st[w]) ** 2), ww) for ww in wells if ww != w))
            nb = [ww for _, ww in dist[:2]]
            mn = mo & np.isin(Wtr, nb)
            if mn.sum() > 300:
                pn = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, n_jobs=8).fit(Xr[mn], Ytr[mn, i]).predict(Xr[idx])
                near.append(np.sqrt(np.mean((pn - Ytr[idx, i]) ** 2)))
        aud[t] = {"same_well_half_split_rmse": float(np.mean(half)), "cross_well_rmse": float(np.mean(cross)), "two_nearest_wells_rmse": float(np.mean(near)) if near else None}
    R["E4_audit_volve"] = aud
    print("E4", aud, flush=True)

    # ---------- E5: blind wells: train on all 9 wells, predict the 4 test wells ----------
    bl = {}
    tr = lab_tr
    bins = {}
    P_reg = np.zeros(Yte.shape); P_mean = np.zeros(Yte.shape); P_bayes = np.zeros(Yte.shape)
    for i, t in enumerate(TARGETS):
        yv = Ytr[tr, i]
        if t == "SW":
            nd = yv < 0.999; lab, centers = quantile_bins(yv[nd], K); edges = np.unique(np.quantile(yv[nd], np.linspace(0, 1, K + 1))); rows = np.flatnonzero(tr)[nd]
        else:
            lab, centers = quantile_bins(yv, K); edges = np.unique(np.quantile(yv, np.linspace(0, 1, K + 1))); rows = np.flatnonzero(tr)
        bins[t] = (edges, centers)
        P_reg[:, i] = np.mean([cb_reg(Xtr[rows], Ytr[rows, i], Xte, seed=s) for s in (42, 2026)], 0)
        prob = np.mean([cb_bins(Xtr[rows], lab, len(centers), Xte, seed=s) for s in (42, 2026)], 0)
        P_mean[:, i] = decode_mean(prob, centers); P_bayes[:, i] = decode_bayes(prob, centers, TOL.get(t, 0.08))
    g_cb = cb_gate(Xtr_e[tr], Ytr[:, 1][tr] >= 0.999, Xte_e)
    outs, heads = nn_train_predict([f for _, f in tr_wells], [f[TARGETS].to_numpy(np.float64) for _, f in tr_wells], [f for _, f in te_wells], bins, seeds=(1, 2, 3))
    g_nn = np.concatenate([o["SW"][0] for o in outs])
    P_nn = np.zeros(Yte.shape)
    for h, t in heads.items():
        i = TARGETS.index(t); centers = bins[t][1]
        P_nn[:, i] = np.concatenate([decode_mean(o[h][1] / o[h][1].sum(1, keepdims=True), centers) for o in outs])
    d_te = Yte[:, 1] >= 0.999
    th_cb = R["CV_leave_well_out"]["E2_mean"]["rmse_with_gate_cb"]["threshold"]; th_nn = R["CV_leave_well_out"]["E2_mean"]["rmse_with_gate_nn"]["threshold"]
    cands = {
        "E1_reg_no_gate": P_reg, "E2_mean_no_gate": P_mean, "E2_bayes_no_gate": P_bayes, "nn_mean_no_gate": P_nn,
        "E2_mean+gate_cb": with_gate(P_mean, g_cb, th_cb), "E2_mean+gate_nn": with_gate(P_mean, g_nn, th_nn),
        "E2_mean+gate_avg": with_gate(P_mean, 0.5 * (g_cb + g_nn), 0.5 * (th_cb + th_nn)),
        "fusion_0.7tree_0.3nn+gate_avg": with_gate(0.7 * P_mean + 0.3 * P_nn, 0.5 * (g_cb + g_nn), 0.5 * (th_cb + th_nn)),
    }
    for name, P in cands.items():
        bl[name] = {"rmse_all3": rmse3(P, Yte), **{f"rmse_{t}": v for t, v in rmse_per(P, Yte).items()},
                    "tol_PHIF": tol_acc(P[:, 0], Yte[:, 0], 0.08), "tol_SW_all": tol_acc(np.clip(P[:, 1], 0, 1), Yte[:, 1], 0.05)}
    bl["gate_accuracy_blind"] = {"cb@th": float(((g_cb >= th_cb) == d_te).mean()), "nn@th": float(((g_nn >= th_nn) == d_te).mean()), "SW_default_share_blind": float(d_te.mean())}
    bl["leaderboard_reference"] = {"winner_UTFE": 0.0525, "second_MoLPhy": 0.0602, "third_Tomsk": 0.0634, "benchmark_RF": 0.11718}
    R["E5_blind_wells"] = bl
    print(json.dumps(bl, indent=1), flush=True)
    R["seconds"] = time.time() - t0
    (OUT / "report.json").write_text(json.dumps(R, indent=1))
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
