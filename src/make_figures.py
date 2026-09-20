"""Paper figures from the experiment JSONs + cached OOF predictions. Run on the server with a matplotlib venv:
  /home/ubuntu/kunlun_welllog_20260918/.venv/bin/python make_figures.py
Writes PNG (300 dpi) + PDF + the underlying numbers (CSV) into outputs/figures/.
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("outputs/figures"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120})
C_TREE, C_NN, C_V1, C_V2, C_V3 = "#4C72B0", "#DD8452", "#C9A227", "#8C8C8C", "#55A868"
T = ("POR", "PERM", "SW")
NL = chr(10)


def save(fig, name, rows=None, header=None):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    if rows is not None:
        with open(OUT / f"{name}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print("wrote", name)


# ---------------- Fig 1: identifiability audit ----------------
D = json.load(open("outputs/diag_wells.json"))
labels = ["same well" + NL + "upper→lower half", "same well" + NL + "interleaved blocks", "cross-well" + NL + "64→16", "leave-well-out" + NL + "12 nearest", "leave-well-out" + NL + "79 wells"]
vals = {"POR": [D["within_well_halfsplit_POR"], 0.553, 0.309, D["lwo_POR_K12"], D["lwo_POR_K79"]],
        "PERM": [D["within_well_halfsplit_PERM"], 0.885, 0.776, np.nan, np.nan],
        "SW": [D["within_well_halfsplit_SW"], 0.286, 0.143, D["lwo_SW_K12"], D["lwo_SW_K79"]]}
fig, ax = plt.subplots(figsize=(7.2, 3.2))
x = np.arange(len(labels)); w = 0.26
for i, t in enumerate(T):
    ax.bar(x + (i - 1) * w, np.array(vals[t], dtype=float), w, label=t, color=[C_TREE, C_V3, C_NN][i])
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("official accuracy (non-default samples)")
ax.set_ylim(0, 1); ax.legend(frameon=False, ncol=3)
ax.set_title("Where can the labels be learned from? (LightGBM on 13 curves + depth)")
save(fig, "fig1_identifiability", [[l.replace(NL, " ")] + [vals[t][i] for t in T] for i, l in enumerate(labels)], ["experiment"] + list(T))

# ---------------- Fig 2: gate error anatomy ----------------
G = json.load(open("outputs/diag_gate.json"))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax = axes[0]
keys = list(G["POR"]["by_distance_to_true_boundary"].keys())
xl = ["1", "2", "3", "4–5", "6–10", "11–20", "21–50", ">50"]
for i, t in enumerate(T):
    err = [G[t]["by_distance_to_true_boundary"][k]["err_rate"] for k in keys]
    ax.plot(range(len(keys)), err, marker="o", ms=3, label=t, color=[C_TREE, C_V3, C_NN][i])
ax.set_xticks(range(len(keys))); ax.set_xticklabels(xl); ax.set_xlabel("distance to nearest true boundary (samples)")
ax.set_ylabel("gate error rate"); ax.legend(frameon=False); ax.set_title("Baseline gate errors concentrate at boundaries")
ax = axes[1]
cats = ["<5", "<10", "<20", ">=20"]
for i, t in enumerate(T):
    rec = [G[t]["thin_nondefault_bed_recall"].get(c, {}).get("recall", np.nan) for c in cats]
    ax.bar(np.arange(len(cats)) + (i - 1) * 0.26, rec, 0.26, label=t, color=[C_TREE, C_V3, C_NN][i])
ax.set_xticks(range(len(cats))); ax.set_xticklabels(["<0.5 m", "0.5–1 m", "1–2 m", "≥2 m"]); ax.set_xlabel("true non-default bed thickness")
ax.set_ylabel("recall of non-default samples"); ax.set_ylim(0, 1); ax.set_title("Thin beds are missed")
save(fig, "fig2_gate_anatomy")

# ---------------- Fig 3: model family comparison ----------------
NN = json.load(open("outputs/nn_gate_b/report.json")); NNc = json.load(open("outputs/nn_base_c/report.json"))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax = axes[0]
x = np.arange(3)
ax.bar(x - 0.27, [NN[t]["nondefault_only"]["v2"] for t in T], 0.27, label="trees (v2)", color=C_TREE)
ax.bar(x, [NNc[t]["nondefault_only"]["nn"] for t in T], 0.27, label="sequence net + feature channels", color=C_NN)
ax.bar(x + 0.27, [NN[t]["nondefault_only"]["nn"] for t in T], 0.27, label="sequence net (curves only)", color="#F2B98F")
ax.set_xticks(x); ax.set_xticklabels(T); ax.set_ylabel("accuracy on true non-default samples"); ax.set_title("Regression: trees win")
ax.set_ylim(0, 1.12); ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=1)
ax = axes[1]
ax.bar(x - 0.27, [NN[t]["gate_acc"]["v2"] for t in T], 0.27, label="trees (v2 gate)", color=C_TREE)
ax.bar(x, [NNc[t]["gate_acc"]["nn@0.5"] for t in T], 0.27, label="sequence net + feature channels", color=C_NN)
ax.bar(x + 0.27, [NN[t]["gate_acc"]["nn@0.5"] for t in T], 0.27, label="sequence net (curves only)", color="#F2B98F")
ax.set_xticks(x); ax.set_xticklabels(T); ax.set_ylabel("default/non-default gate accuracy"); ax.set_ylim(0.8, 0.96); ax.set_title("Segmentation: networks win")
save(fig, "fig3_division_of_labour",
     [[t, NN[t]["nondefault_only"]["v2"], NNc[t]["nondefault_only"]["nn"], NN[t]["nondefault_only"]["nn"], NN[t]["gate_acc"]["v2"], NNc[t]["gate_acc"]["nn@0.5"], NN[t]["gate_acc"]["nn@0.5"]] for t in T],
     ["target", "nondef_trees", "nondef_nn_feat", "nondef_nn_curves", "gate_trees", "gate_nn_feat", "gate_nn_curves"])

# ---------------- Fig 4: three generations + fusion progression, CV vs leaderboard ----------------
V1 = json.load(open("outputs/cv_v1/report.json"))
steps = ["v1" + NL + "(two-stage CatBoost)", "v2" + NL + "(target-specific)", "+ distribution models" + NL + "+ state gate", "+ sequence net", "+ POR lag/zone" + NL + "features", "+ 2nd network" + NL + "(v3 final)"]
cv = [V1["TOTAL"], 80.884, 81.163, 81.412, 81.520, 81.632]
lb = {steps[0]: 81.0052, steps[1]: 82.2718, steps[-1]: 83.2579}
fig, ax = plt.subplots(figsize=(7.6, 3.2))
ax.plot(range(len(steps)), cv, marker="o", color=C_V3, label="well-grouped 5-fold CV (nested)")
first = True
for i, s in enumerate(steps):
    if s in lb:
        ax.scatter([i], [lb[s]], color=C_NN, zorder=3, label="blind leaderboard (5 wells)" if first else None); first = False
        ax.annotate(f"{lb[s]:.2f}", (i, lb[s]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, color=C_NN)
    ax.annotate(f"{cv[i]:.2f}", (i, cv[i]), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8, color=C_V3)
ax.set_xticks(range(len(steps))); ax.set_xticklabels(steps, fontsize=7.5); ax.set_ylabel("official total score"); ax.set_ylim(79.0, 83.8)
ax.legend(frameon=False, loc="upper left"); ax.set_title("Three generations and the cumulative gains of the hybrid")
save(fig, "fig4_progression", [[s.replace(NL, " "), cv[i], lb.get(s, "")] for i, s in enumerate(steps)], ["step", "cv_total", "leaderboard"])

# ---------------- Fig 5: leaderboard: generations + ablation ----------------
names = ["v1" + NL + "(first entry)", "v2" + NL + "baseline", "v3 gates" + NL + "+ v2 values", "v2 gates" + NL + "+ v3 values", "v3 without" + NL + "networks", "v3 full"]
scores = [81.0052, 82.2718, 82.9916, 82.5381, 82.8034, 83.2579]
fig, ax = plt.subplots(figsize=(5.6, 3.0))
cols = [C_V1, C_V2, C_TREE, C_TREE, C_V3, C_V3]
ax.bar(range(6), [s - 80.5 for s in scores], color=cols, bottom=80.5)
for i, s in enumerate(scores):
    ax.annotate(f"{s:.3f}" + NL + f"({s - scores[1]:+.3f})" if i > 1 else f"{s:.3f}", (i, s), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.5)
ax.set_xticks(range(6)); ax.set_xticklabels(names, fontsize=8); ax.set_ylim(80.5, 83.9); ax.set_ylabel("blind leaderboard score")
ax.set_title("Blind wells: three generations and the v3 ablation (Δ vs v2)")
save(fig, "fig5_leaderboard_ablation", [[n.replace(NL, " "), s] for n, s in zip(names, scores)], ["submission", "score"])

# ---------------- Fig 6: example well track (OOF), v1 / v2 / v3 ----------------
sys.path.insert(0, ".")
from harness import load_or_build_cache, acc_vec
C = load_or_build_cache(Path("data"), Path("cache"))
y, wells, depth, folds = C["tr_y"], C["tr_well"], C["tr_depth"], C["tr_fold"]
final = np.load("outputs/assembly_D/final_oof.npy")
v1p = np.load("outputs/cv_v1/final_oof.npy")
V2 = np.load("outputs/cv_v2/oof.npz")
v2p = np.zeros_like(y)
for f in range(5):
    va = folds == f
    v2p[va, 0] = np.where(V2["por_gate"][va] >= 0.2613781234057015, 0.1, np.expm1(V2["por_centers"][f])[np.argmax(V2["por_prob"][va], 1)])
v2p[:, 1] = np.where(V2["perm_gate"] >= 0.37384488065262944, 0.01, np.clip(10 ** (V2["perm_log"] * 1.03), 0.01, 1e4))
v2p[:, 2] = np.where(V2["sw_gate"] >= 0.08702066266867681, 99.9, np.clip(V2["sw_reg"] * 1.05 - 3.0, 8.305, 99.9))
scores_w = []
for w in np.unique(wells):
    m = wells == w
    scores_w.append((w, acc_vec(final[m, 0], y[m, 0], "POR").mean(), (y[m, 0] > 0.1).mean(), m.sum()))
scores_w.sort(key=lambda z: abs(z[1] - 0.79))
w = [z for z in scores_w if 0.25 < z[2] < 0.45 and z[3] > 7000][0][0]
m = wells == w
d = depth[m]; lo = d.min() + (d.max() - d.min()) * 0.35; hi = lo + 120
sel = (d >= lo) & (d <= hi)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 6.0), sharey=True)
for i, (t, lab) in enumerate(zip(T, ("POR (%)", "PERM (mD)", "SW (%)"))):
    ax = axes[i]
    yy, p3, p2, p1 = y[m, i][sel], final[m, i][sel], v2p[m, i][sel], v1p[m, i][sel]
    if t == "PERM":
        yy, p3, p2, p1 = np.log10(yy), np.log10(p3), np.log10(p2), np.log10(p1); lab = "log10 PERM (mD)"
    ax.plot(yy, d[sel], color="black", lw=1.0, label="interpreted label")
    ax.plot(p1, d[sel], color=C_V1, lw=0.7, alpha=0.8, label="v1 first entry")
    ax.plot(p2, d[sel], color=C_V2, lw=0.8, alpha=0.9, label="v2 baseline")
    ax.plot(p3, d[sel], color=C_V3, lw=0.8, alpha=0.9, label="v3 hybrid")
    ax.set_xlabel(lab)
    if i == 0:
        ax.set_ylabel("depth (m)"); ax.legend(frameon=False, fontsize=7, loc="lower right")
axes[0].invert_yaxis()
fig.suptitle(f"Held-out well {C['meta']['train_wells'][w][:8]} (out-of-fold predictions), {lo:.0f}–{hi:.0f} m", fontsize=9)
save(fig, "fig6_example_well")
print("well used for fig6:", C["meta"]["train_wells"][w])

# ---------------- Fig 7: per-target progression across generations (CV) ----------------
V2R = json.load(open("outputs/cv_v2/report_v2_rule.json")); V3R = json.load(open("outputs/assembly_D/report.json"))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax = axes[0]
for i, t in enumerate(T):
    ax.bar(np.arange(3) + (i - 1) * 0.26, [V1[t]["micro"], V2R[t]["micro"], V3R[t]["nested"]["micro"]], 0.26, label=t, color=[C_TREE, C_V3, C_NN][i])
ax.set_xticks(range(3)); ax.set_xticklabels(["v1", "v2", "v3"]); ax.set_ylim(0.7, 0.92); ax.set_ylabel("official accuracy (all samples)"); ax.legend(frameon=False, ncol=3, fontsize=7)
ax.set_title("Per-property score by generation (CV)")
ax = axes[1]
for i, t in enumerate(T):
    ax.bar(np.arange(3) + (i - 1) * 0.26, [V1[t]["nondefault_seg"], V2R[t]["nondefault_seg"], V3R[t]["nested"]["nondefault_seg"]], 0.26, label=t, color=[C_TREE, C_V3, C_NN][i])
ax.set_xticks(range(3)); ax.set_xticklabels(["v1", "v2", "v3"]); ax.set_ylabel("accuracy on true non-default samples"); ax.set_ylim(0, 0.9)
ax.set_title("Non-default (reservoir) samples only")
save(fig, "fig7_generations_per_target",
     [[t, V1[t]["micro"], V2R[t]["micro"], V3R[t]["nested"]["micro"], V1[t]["nondefault_seg"], V2R[t]["nondefault_seg"], V3R[t]["nested"]["nondefault_seg"]] for t in T],
     ["target", "v1_micro", "v2_micro", "v3_micro", "v1_nondef", "v2_nondef", "v3_nondef"])

# ---------------- Fig 8: type-well selection criteria (leave-well-out) ----------------
if Path("outputs/diag_typewell.json").exists():
    TW = json.load(open("outputs/diag_typewell.json"))
    crits = [("median", "median-curve distance"), ("kl", "KL divergence"), ("dtw_gr", "GR DTW"), ("dtw_comp", "composite DTW")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, t in zip(axes, ("POR", "SW")):
        x = np.arange(len(crits)); w = 0.26
        ax.bar(x - w, [TW[f"{t}_{c}_K12"] for c, _ in crits], w, label="12 nearest wells", color="#F2B98F")
        ax.bar(x, [TW[f"{t}_{c}_K30"] for c, _ in crits], w, label="30 nearest wells", color=C_NN)
        ax.bar(x + w, [TW[f"{t}_{c}_weighted_all"] for c, _ in crits], w, label="all wells, similarity-weighted", color=C_TREE)
        ax.axhline(TW[f"{t}_all79"], color="black", ls="--", lw=1, label="all 79 wells, unweighted")
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in crits], fontsize=7, rotation=15); ax.set_title(f"{t}: type-well selection (leave-well-out)")
        ax.set_ylabel("official accuracy (non-default samples)")
        if t == "POR":
            ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    save(fig, "fig8_typewell", [[t, c, TW[f"{t}_{c}_K12"], TW[f"{t}_{c}_K30"], TW[f"{t}_{c}_weighted_all"], TW[f"{t}_all79"]] for t in ("POR", "SW") for c, _ in crits],
         ["target", "criterion", "K12", "K30", "weighted_all", "all79"])
