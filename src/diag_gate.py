"""Where do gate errors live? Distance to the nearest true boundary, per-well error rates, thin beds."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from harness import load_or_build_cache, default_mask, V2_CAL, N_FOLDS, acc_vec, BOUNDS, DEFAULTS

C = load_or_build_cache(Path("data"), Path("cache"))
y, wells, folds, pos = C["tr_y"], C["tr_well"], C["tr_fold"], C["tr_pos"]
O = np.load("outputs/cv_v2/oof.npz")
brk = np.flatnonzero((np.diff(wells) != 0) | (np.diff(pos) != 1)) + 1
segs = np.split(np.arange(len(y)), brk)
R = {}
for ti, (t, gk) in enumerate([("POR", "por_gate"), ("PERM", "perm_gate"), ("SW", "sw_gate")]):
    d = default_mask(y[:, ti], t)
    g = O[gk] >= V2_CAL[t]["threshold"]
    err = g != d
    # distance of every row to nearest true transition (within its contiguous segment)
    dist = np.full(len(y), 10 ** 6)
    for idx in segs:
        dd = d[idx]
        tr = np.flatnonzero(dd[1:] != dd[:-1]) + 1  # boundary before position tr
        if len(tr) == 0:
            continue
        p = np.arange(len(idx))
        # distance to nearest boundary index (boundary sits between tr-1 and tr)
        left = np.searchsorted(tr, p, side="right") - 1
        dl = np.where(left >= 0, p - tr[np.clip(left, 0, None)] + 1, 10 ** 6)
        right = np.clip(left + 1, None, len(tr) - 1)
        dr = np.where(left + 1 < len(tr), tr[right] - p, 10 ** 6)
        dist[idx] = np.minimum(dl, dr)
    bins = [1, 2, 3, 5, 10, 20, 50, 10 ** 7]
    hist = {}
    prev = 0
    for b in bins:
        m = (dist > prev) & (dist <= b)
        hist[f"({prev},{b}]"] = {"rows": int(m.sum()), "err_rows": int(err[m].sum()), "err_rate": float(err[m].mean()) if m.any() else None,
                                 "share_of_all_errors": float(err[m].sum() / err.sum())}
        prev = b
    # thin true nondefault beds (run length < L) recall
    thin = {}
    for idx in segs:
        dd = ~d[idx]; ch = np.r_[True, dd[1:] != dd[:-1]]; st = np.flatnonzero(ch); ln = np.diff(np.r_[st, len(idx)]); vv = dd[st]
        for s0, l0, v0 in zip(st, ln, vv):
            if not v0:
                continue
            key = "<5" if l0 < 5 else "<10" if l0 < 10 else "<20" if l0 < 20 else ">=20"
            rows = idx[s0:s0 + l0]
            thin.setdefault(key, [0, 0])
            thin[key][0] += l0; thin[key][1] += int((~g[rows]).sum())
    per_well = np.array([err[wells == w].mean() for w in np.unique(wells)])
    R[t] = {"gate_error_rate": float(err.mean()), "by_distance_to_true_boundary": hist,
            "thin_nondefault_bed_recall": {k: {"rows": v[0], "recalled_rows": v[1], "recall": v[1] / v[0]} for k, v in thin.items()},
            "per_well_error_rate": {"q10": float(np.quantile(per_well, .1)), "q50": float(np.quantile(per_well, .5)), "q90": float(np.quantile(per_well, .9)), "max": float(per_well.max())},
            "false_default_share_within_2_of_boundary": float(((g & ~d) & (dist <= 2)).sum() / max((g & ~d).sum(), 1)),
            "missed_default_share_within_2_of_boundary": float(((~g & d) & (dist <= 2)).sum() / max((~g & d).sum(), 1))}
    # what if gate errors farther than 2 samples from a boundary were fixed (upper bound of non-boundary gain)
    cont = None
    if t == "POR":
        cont = np.zeros(len(y))
        for f in range(N_FOLDS):
            va = folds == f; cont[va] = np.expm1(O["por_centers"][f])[np.argmax(O["por_prob"][va], 1)]
    elif t == "PERM":
        cont = np.clip(10 ** (O["perm_log"] * 1.03), *BOUNDS["PERM"])
    else:
        cont = np.clip(O["sw_reg"] * 1.05 - 3, *BOUNDS["SW"])
    base = acc_vec(np.where(g, DEFAULTS[t], cont), y[:, ti], t).mean()
    g_fix = np.where(dist > 2, d, g)
    R[t]["score_v2"] = float(base)
    R[t]["score_if_nonboundary_errors_fixed"] = float(acc_vec(np.where(g_fix, DEFAULTS[t], cont), y[:, ti], t).mean())
    g_fix5 = np.where(dist > 5, d, g)
    R[t]["score_if_errors_beyond5_fixed"] = float(acc_vec(np.where(g_fix5, DEFAULTS[t], cont), y[:, ti], t).mean())
Path("outputs/diag_gate.json").write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
