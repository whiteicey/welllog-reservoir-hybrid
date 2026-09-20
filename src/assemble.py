"""Assemble final per-target decisions from cached OOF/test outputs of several models.

Low-dimensional, nested blending per target:
  continuous: mixture of predictive distributions (CatBoost bins, sequence-net bins) on a simplex grid
              (step 0.25), decoded with the official-kernel Bayes action on a dense action grid;
  gate:       mixture of default probabilities (v2 CatBoost gate, state-model marginal, sequence-net gate)
              on a simplex grid, optional 2-state HMM smoothing, single threshold.
Configuration (weights, HMM stay, threshold) is selected on the 4 inner folds and applied to the outer
fold (nested score), and separately on all 5 folds for the test-time configuration.
Threshold search is exact and vectorised (sort by gate probability, cumulative sums).
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from exp_decisions import hmm_posterior, by_well_sequences
from harness import (BOUNDS, DEFAULTS, N_FOLDS, V2_CAL, acc_vec, breakdown, default_mask, kernel_matrix,
                     load_or_build_cache, paired_well_bootstrap, total)

TI = {"POR": 0, "PERM": 1, "SW": 2}
TINV = {"POR": np.expm1, "PERM": lambda v: 10 ** v, "SW": lambda v: v}


def dense_actions(sup, t):
    return np.linspace(sup[0], sup[-1], 800) if t == "SW" else np.exp(np.linspace(np.log(sup[0] * 0.9), np.log(sup[-1] * 1.1), 800))


def simplex_grid(k, step=0.25):
    if k == 1:
        return [(1.0,)]
    n = int(round(1 / step))
    return [tuple(c / n for c in combo) for combo in itertools.product(range(n + 1), repeat=k) if sum(combo) == n]


class DistSource:
    def __init__(self, spec, t):
        self.t = t
        self.O, self.T = np.load(spec["oof"]), np.load(spec["test"])
        self.pk, self.ck = spec.get("prob_key", "prob"), spec.get("centers_key", "centers")

    def support(self, f):
        return TINV[self.t](self.O[self.ck][f])

    def prob(self, rows=None, f=None):
        p = (self.O[self.pk][rows] if rows is not None else self.T[self.pk][f]).astype(np.float64)
        return p / p.sum(1, keepdims=True)


def expected_utility(srcs, weights, t, grid, rows=None, f=None):
    util = 0
    for s, w in zip(srcs, weights):
        if w == 0:
            continue
        p = s.prob(rows=rows, f=f)
        util = util + w * (p @ kernel_matrix(grid, s.support(f), t).T)
    return util


def best_threshold_curve(g, acc_d, acc_c, grid):
    """Exact score for every threshold in grid. score(th) = mean(acc_c) + sum_{g>=th}(acc_d-acc_c)/n."""
    order = np.argsort(-g)
    gs = g[order]
    gain = np.cumsum((acc_d - acc_c)[order])
    n = len(g)
    cnt = np.searchsorted(-gs, -grid, side="right")  # number of rows with g >= th
    add = np.where(cnt > 0, gain[np.clip(cnt - 1, 0, n - 1)], 0.0)
    return (acc_c.sum() + add) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    C = load_or_build_cache(Path("data"), Path("cache"))
    y, folds, wells, pos = C["tr_y"], C["tr_fold"], C["tr_well"], C["tr_pos"]
    n, nt = len(y), len(C["te_well"])
    rows_by_fold = [np.flatnonzero(folds == f) for f in range(N_FOLDS)]
    args.out.mkdir(parents=True, exist_ok=True)
    final_oof = np.zeros_like(y); final_test = np.zeros((nt, 3))
    report = {}
    decision = {}
    V2 = np.load("outputs/cv_v2/oof.npz")
    th_grid = np.round(np.linspace(0.02, 0.9, 89), 4)
    for t in ("POR", "PERM", "SW"):
        ti = TI[t]; spec = cfg[t]
        d = default_mask(y[:, ti], t)
        acc_default = acc_vec(np.full(n, DEFAULTS[t]), y[:, ti], t)
        # ---------- continuous candidates ----------
        srcs = [DistSource(s, t) for s in spec["dist_sources"]]
        cw_grid = [tuple(w) for w in spec.get("weight_grid", simplex_grid(len(srcs)))]
        cont = {}
        for cw in cw_grid:
            c = np.zeros(n)
            for f in range(N_FOLDS):
                rows = rows_by_fold[f]
                grid = dense_actions(srcs[0].support(f), t)
                c[rows] = grid[np.argmax(expected_utility(srcs, cw, t, grid, rows=rows, f=f), 1)]
            cont[cw] = np.clip(c, *BOUNDS[t])
        acc_cont = {cw: acc_vec(c, y[:, ti], t) for cw, c in cont.items()}
        # ---------- gate candidates ----------
        def load_gate(path, g):
            a = np.load(path)[g["key"]]
            if g["key"] == "post":  # joint state posterior -> per-target default marginal
                return {"POR": a[..., 0], "PERM": a[..., 0] + a[..., 3], "SW": a[..., 0] + a[..., 2]}[t]
            return a
        G = [load_gate(g["oof"], g) for g in spec["gate_sources"]]
        GT = [load_gate(g["test"], g) for g in spec["gate_sources"]]
        gw_grid = [tuple(w) for w in spec.get("gate_weight_grid", simplex_grid(len(G)))]
        hmm_grid = spec.get("hmm_stay", [None, 0.95, 0.98, 0.99])

        def smooth(g, w, p, stay):
            if stay is None:
                return g
            out = g.copy()
            for idx in by_well_sequences(w, p):
                if len(idx) >= 3:
                    out[idx] = hmm_posterior(g[idx], stay=stay)
            return out

        gates = {}
        for gw in gw_grid:
            g = sum(w * G[i] for i, w in enumerate(gw))
            for stay in hmm_grid:
                gates[(gw, stay)] = smooth(g, wells, pos, stay)
        combos = [(cw, gk) for cw in cw_grid for gk in gates]
        print(t, "combos", len(combos), flush=True)
        # ---------- nested selection ----------
        nested_pred = np.zeros(n); chosen = []
        for f in range(N_FOLDS):
            inner = folds != f; outer = rows_by_fold[f]
            best = (-1, None)
            for cw, gk in combos:
                curve = best_threshold_curve(gates[gk][inner], acc_default[inner], acc_cont[cw][inner], th_grid)
                j = int(np.argmax(curve))
                if curve[j] > best[0]:
                    best = (float(curve[j]), (cw, gk, float(th_grid[j])))
            cw, gk, th = best[1]
            chosen.append({"fold": f, "cont_w": cw, "gate_w": gk[0], "hmm_stay": gk[1], "threshold": th, "inner_score": best[0]})
            nested_pred[outer] = np.where(gates[gk][outer] >= th, DEFAULTS[t], cont[cw][outer])
        # ---------- global selection (test-time configuration) ----------
        best = (-1, None)
        for cw, gk in combos:
            curve = best_threshold_curve(gates[gk], acc_default, acc_cont[cw], th_grid)
            j = int(np.argmax(curve))
            if curve[j] > best[0]:
                best = (float(curve[j]), (cw, gk, float(th_grid[j])))
        cw, gk, th = best[1]
        gw, stay = gk
        final_oof[:, ti] = nested_pred
        decision[t] = {"cont_w": list(cw), "gate_w": list(gw), "hmm_stay": stay, "threshold": th, "oof_score_selected": best[0],
                       "dist_sources": [s["name"] for s in spec["dist_sources"]], "gate_sources": [g["name"] for g in spec["gate_sources"]]}
        # v2 reference
        if t == "POR":
            v2c = np.zeros(n)
            for f in range(N_FOLDS):
                va = folds == f
                v2c[va] = np.expm1(V2["por_centers"][f])[np.argmax(V2["por_prob"][va], 1)]
            v2g = V2["por_gate"]
        elif t == "PERM":
            v2c, v2g = np.clip(10 ** (V2["perm_log"] * 1.03), *BOUNDS["PERM"]), V2["perm_gate"]
        else:
            v2c, v2g = np.clip(V2["sw_reg"] * 1.05 - 3.0, *BOUNDS["SW"]), V2["sw_gate"]
        v2pred = np.where(v2g >= V2_CAL[t]["threshold"], DEFAULTS[t], v2c)
        rep = {
            "nested": breakdown(nested_pred, y[:, ti], t, wells, folds),
            "nested_choices": chosen,
            "global_choice": decision[t],
            "v2": breakdown(v2pred, y[:, ti], t, wells, folds),
            "paired_nested_vs_v2": paired_well_bootstrap(acc_vec(nested_pred, y[:, ti], t), acc_vec(v2pred, y[:, ti], t), wells),
            "continuous_only_by_weight": {str(k): float(a[~d].mean()) for k, a in acc_cont.items()},
            "continuous_only_v2": float(acc_vec(v2c[~d], y[~d, ti], t).mean()),
            "gate_only_best_by_source": {},
        }
        # ablation: each gate source alone with nested threshold and the globally chosen continuous
        for gi, gsrc in enumerate(spec["gate_sources"]):
            gw1 = tuple(1.0 if j == gi else 0.0 for j in range(len(G)))
            if (gw1, None) in gates:
                curve = best_threshold_curve(gates[(gw1, None)], acc_default, acc_cont[cw], th_grid)
                rep["gate_only_best_by_source"][gsrc["name"]] = float(curve.max())
        for k in ("nested", "v2"):
            rep[k].pop("per_well", None)
        report[t] = rep
        # ---------- test prediction ----------
        cont_te = np.zeros(nt)
        util = 0
        for f in range(N_FOLDS):
            grid = dense_actions(srcs[0].support(f), t)
            util = util + expected_utility(srcs, cw, t, grid, f=f)
        cont_te = np.clip(grid[np.argmax(util, 1)], *BOUNDS[t])  # grids differ slightly per fold; use last grid (same range)
        gate_te = sum(w * GT[i].mean(0) for i, w in enumerate(gw))
        gate_te = smooth(gate_te, C["te_well"], np.arange(nt), stay)
        final_test[:, ti] = np.where(gate_te >= th, DEFAULTS[t], cont_te)
        print(t, "nested", round(rep["nested"]["micro"], 5), "v2", round(rep["v2"]["micro"], 5), "choice", decision[t], flush=True)
    report["TOTAL_nested"] = total({t: report[t]["nested"]["micro"] for t in TI})
    report["TOTAL_v2"] = total({t: report[t]["v2"]["micro"] for t in TI})
    report["TOTAL_global_selected_optimistic"] = total({t: decision[t]["oof_score_selected"] for t in TI})
    np.save(args.out / "final_oof.npy", final_oof); np.save(args.out / "final_test.npy", final_test)
    (args.out / "report.json").write_text(json.dumps(report, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    (args.out / "decision.json").write_text(json.dumps(decision, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("TOTAL nested", report["TOTAL_nested"], "v2", report["TOTAL_v2"], "optimistic", report["TOTAL_global_selected_optimistic"])


if __name__ == "__main__":
    main()
