"""Run-length / state-structure diagnostics: training labels vs v2 test predictions vs v2 OOF predictions."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from harness import load_or_build_cache, default_mask, V2_CAL, BOUNDS, DEFAULTS, N_FOLDS


def states(por, perm, sw):
    return np.select([por <= 0.1, sw >= 99.9, perm <= 0.01], [0, 2, 3], 1)


def run_stats(seq_list):
    out = {}
    for s in range(4):
        L = []
        for st in seq_list:
            ch = np.r_[True, st[1:] != st[:-1]]
            idx = np.flatnonzero(ch); lens = np.diff(np.r_[idx, len(st)]); vals = st[idx]
            L += lens[vals == s].tolist()
        L = np.array(L) if L else np.zeros(1)
        out[str(s)] = {"runs": int(len(L)), "q05": float(np.percentile(L, 5)), "q50": float(np.percentile(L, 50)),
                       "runs_lt5": int((L < 5).sum()), "runs_lt10": int((L < 10).sum()), "rows_in_runs_lt10": int(L[L < 10].sum()), "rows": int(L.sum())}
    n = sum(len(st) for st in seq_list)
    out["runs_per_10k_rows"] = float(sum(v["runs"] for k, v in out.items() if k.isdigit()) / n * 1e4)
    return out


C = load_or_build_cache(Path("data"), Path("cache"))
y, wells, folds, pos = C["tr_y"], C["tr_well"], C["tr_fold"], C["tr_pos"]
# split by contiguous runs
brk = np.flatnonzero((np.diff(wells) != 0) | (np.diff(pos) != 1)) + 1
segs = np.split(np.arange(len(y)), brk)
R = {"train_labels": run_stats([states(y[i, 0], y[i, 1], y[i, 2]) for i in segs])}
O = np.load("outputs/cv_v2/oof.npz")
pred = np.zeros_like(y)
for f in range(N_FOLDS):
    va = folds == f
    c = np.expm1(O["por_centers"][f])[np.argmax(O["por_prob"][va], 1)]
    pred[va, 0] = np.where(O["por_gate"][va] >= V2_CAL["POR"]["threshold"], 0.1, c)
pred[:, 1] = np.where(O["perm_gate"] >= V2_CAL["PERM"]["threshold"], 0.01, np.clip(10 ** (O["perm_log"] * 1.03), *BOUNDS["PERM"]))
pred[:, 2] = np.where(O["sw_gate"] >= V2_CAL["SW"]["threshold"], 99.9, np.clip(O["sw_reg"] * 1.05 - 3, *BOUNDS["SW"]))
R["v2_oof_pred"] = run_stats([states(pred[i, 0], pred[i, 1], pred[i, 2]) for i in segs])
# transition matrix of true states
tr = np.zeros((4, 4))
for i in segs:
    s = states(y[i, 0], y[i, 1], y[i, 2]); np.add.at(tr, (s[:-1], s[1:]), 1)
R["train_transition_probs"] = (tr / tr.sum(1, keepdims=True)).round(5).tolist()
R["train_state_fractions"] = np.bincount(states(y[:, 0], y[:, 1], y[:, 2]), minlength=4).tolist()
# label consistency across targets (cross-target default agreement)
R["cross_target_default_agreement"] = {
    "POR_def_and_PERM_def": float((default_mask(y[:, 0], "POR") & default_mask(y[:, 1], "PERM")).mean()),
    "POR_def": float(default_mask(y[:, 0], "POR").mean()), "PERM_def": float(default_mask(y[:, 1], "PERM").mean()), "SW_def": float(default_mask(y[:, 2], "SW").mean()),
    "POR_nondef_and_SW_def": float((~default_mask(y[:, 0], "POR") & default_mask(y[:, 2], "SW")).mean()),
    "POR_def_and_SW_nondef": float((default_mask(y[:, 0], "POR") & ~default_mask(y[:, 2], "SW")).mean()),
}
# how often OOF gates disagree with each other where labels agree
R["v2_oof_gate_inconsistency"] = {
    "pred_POR_def_but_PERM_nondef": float(((pred[:, 0] <= 0.1) & (pred[:, 1] > 0.01)).mean()),
    "pred_POR_nondef_but_PERM_def": float(((pred[:, 0] > 0.1) & (pred[:, 1] <= 0.01)).mean()),
    "pred_POR_def_but_SW_nondef": float(((pred[:, 0] <= 0.1) & (pred[:, 2] < 99.9)).mean()),
}
Path("outputs/diag_runs.json").write_text(json.dumps(R, indent=1))
print(json.dumps(R, indent=1))
