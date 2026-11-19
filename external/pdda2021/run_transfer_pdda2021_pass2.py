"""Second pass on the Volve blind wells (cheap): regression + gate combinations, per-well-normalised features,
and the physics/range post-processing rules used by the 2021 contest's third-place team, reported separately.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "<PROJECT_ROOT>")
from run_transfer_pdda2021 import (load, build_features, rmse3, rmse_per, tol_acc, quantile_bins, decode_mean, cb_reg, cb_gate, TARGETS, OUT)

t0 = time.time()
tr_wells, te_wells = load()
builder, (Xtr, Xtr_e, Ytr, Wtr, Dtr), (Xte, Xte_e, Yte, Wte, Dte) = build_features(tr_wells, te_wells)
lab = np.isfinite(Ytr).all(1)
d_tr = Ytr[:, 1] >= 0.999; d_te = Yte[:, 1] >= 0.999
res = {}


def with_gate(P, g, th):
    out = P.copy(); out[:, 1] = np.where(g >= th, 1.0, P[:, 1]); return out


for fname, (A, B) in {"base+extra": (Xtr, Xte), "enh+extra": (Xtr_e, Xte_e)}.items():
    P = np.zeros(Yte.shape)
    for i, t in enumerate(TARGETS):
        rows = np.flatnonzero(lab & ~d_tr) if t == "SW" else np.flatnonzero(lab)
        P[:, i] = np.mean([cb_reg(A[rows], Ytr[rows, i], B, seed=s) for s in (42, 2026)], 0)
    g = np.mean([cb_gate(A[lab], d_tr[lab], B, seed=s) for s in (42, 2026)], 0)
    for th in (0.3, 0.5):
        Q = with_gate(P, g, th)
        res[f"reg[{fname}]+gate@{th}"] = {"rmse_all3": rmse3(Q, Yte), **{f"rmse_{t}": v for t, v in rmse_per(Q, Yte).items()},
                                          "gate_acc": float(((g >= th) == d_te).mean()), "tol_SW_all": tol_acc(np.clip(Q[:, 1], 0, 1), Yte[:, 1], 0.05)}
    res[f"reg[{fname}]_no_gate"] = {"rmse_all3": rmse3(P, Yte), **{f"rmse_{t}": v for t, v in rmse_per(P, Yte).items()}}
    # Tomsk-style range rules derived from the training data only (no test labels)
    raw_te = pd.concat([f for _, f in te_wells], ignore_index=True); raw_tr = pd.concat([f for _, f in tr_wells], ignore_index=True)
    gr_max = np.nanmax(raw_tr.loc[lab, "GR"]); den_hi = 3.0
    Q = with_gate(P, g, 0.3)
    Q[raw_te["GR"].to_numpy() > gr_max, 2] = 1.0
    Q[raw_te["DEN"].to_numpy() > den_hi, 0] = 0.02
    Q[np.log10(np.clip(raw_te["RT"].to_numpy(), 1e-3, None)) < -2, 1] = 1.0
    res[f"reg[{fname}]+gate@0.3+range_rules"] = {"rmse_all3": rmse3(Q, Yte), **{f"rmse_{t}": v for t, v in rmse_per(Q, Yte).items()},
                                                  "rows_hit": {"GR>train_max": int((raw_te["GR"] > gr_max).sum()), "DEN>3": int((raw_te["DEN"] > den_hi).sum()), "logRT<-2": int((np.log10(np.clip(raw_te["RT"], 1e-3, None)) < -2).sum())}}
    # per blind well breakdown for the best plain configuration
    Q = with_gate(P, g, 0.3)
    res[f"reg[{fname}]+gate@0.3_per_well"] = {int(w): {"rmse_all3": rmse3(Q[Wte == w], Yte[Wte == w]), "SW_default_share": float(d_te[Wte == w].mean()), "gate_acc": float(((g[Wte == w] >= 0.3) == d_te[Wte == w]).mean())} for w in np.unique(Wte)}
    print(fname, json.dumps({k: v for k, v in res.items() if k.startswith(f"reg[{fname}]") and "per_well" not in k}, indent=1), flush=True)
res["seconds"] = time.time() - t0
(OUT / "report_blind_pass2.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if "per_well" in k}, indent=1))
