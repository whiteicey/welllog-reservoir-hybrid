"""Reproduce a small public Volve validation without confidential training data.

Run on the SSH server with the project virtualenv. This trains only on the
public PDDA train wells and evaluates the released public test wells. The public
blind wells were previously used during development, so this is a reproducibility
check, not a claim of a fresh untouched external test.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import os

import numpy as np
import pandas as pd

MAIN = Path(os.environ.get("WELLLOG_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(MAIN / "external" / "pdda2021"))
from run_transfer_pdda2021 import (
    TARGETS, adapt, build_features, cb_bins, cb_gate, cb_reg,
    decode_bayes, decode_mean, load, quantile_bins, rmse3, rmse_per, tol_acc,
)

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (42, 2026)
TOL = {"PHIF": 0.08, "SW": 0.05, "VSH": 0.08}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metrics(pred, truth):
    return {
        "rmse_all3": rmse3(pred, truth),
        **rmse_per(pred, truth),
        "tol_PHIF": tol_acc(pred[:, 0], truth[:, 0], 0.08),
        "tol_SW": tol_acc(np.clip(pred[:, 1], 0, 1), truth[:, 1], 0.05),
        "tol_VSH": tol_acc(pred[:, 2], truth[:, 2], 0.08),
    }


def main() -> None:
    started = datetime.now(timezone.utc)
    t0 = time.time()
    tr_wells, te_wells = load()
    _, (Xtr, _, Ytr, Wtr, _), (Xte, _, Yte, Wte, Dte) = build_features(tr_wells, te_wells)
    lab = np.isfinite(Ytr).all(1)
    dtr = Ytr[:, 1] >= 0.999
    predictions = {"truth": Yte, "well": Wte, "depth": Dte}
    results = {"variants": {}, "dataset": {
        "train_rows": len(Ytr), "train_labeled_rows": int(lab.sum()), "test_rows": len(Yte),
        "train_wells": [int(w) for w in np.unique(Wtr)],
        "public_test_wells": [int(w) for w in np.unique(Wte)],
        "features": "base + well_extra_features, fitted on public training wells only",
        "targets": TARGETS,
        "test_protocol": "Released public test wells previously used in development; this is a reproduction, not a new blind test.",
    }}

    # Public RF-style baseline, matching the previous five-log protocol.
    cols = ["DEN", "GR", "CNL", "PE", "RT"]
    ftr = np.concatenate([f[cols].to_numpy(np.float64) for _, f in tr_wells])
    fte = np.concatenate([f[cols].to_numpy(np.float64) for _, f in te_wells])
    good = lab & np.isfinite(ftr).all(1)
    good_te = np.isfinite(fte).all(1)
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=8)
    rf.fit(ftr[good], Ytr[good])
    p_rf = np.tile(Ytr[good].mean(0), (len(Yte), 1))
    p_rf[good_te] = rf.predict(fte[good_te])
    predictions["rf_five_log"] = p_rf
    results["variants"]["rf_five_log"] = metrics(p_rf, Yte)

    # Same tree features for regression and 60-bin distributions.
    p_reg = np.zeros(Yte.shape)
    p_mean = np.zeros(Yte.shape)
    p_bayes = np.zeros(Yte.shape)
    bins = {}
    for i, target in enumerate(TARGETS):
        values = Ytr[lab, i]
        if target == "SW":
            nd = values < 0.999
            b_labels, centers = quantile_bins(values[nd], 60)
            edges = np.unique(np.quantile(values[nd], np.linspace(0, 1, 61)))
            rows = np.flatnonzero(lab)[nd]
        else:
            b_labels, centers = quantile_bins(values, 60)
            edges = np.unique(np.quantile(values, np.linspace(0, 1, 61)))
            rows = np.flatnonzero(lab)
        bins[target] = {"edges": edges.tolist(), "centers": centers.tolist()}
        p_reg[:, i] = np.mean([cb_reg(Xtr[rows], Ytr[rows, i], Xte, seed=s) for s in SEEDS], axis=0)
        prob = np.mean([cb_bins(Xtr[rows], b_labels, len(centers), Xte, seed=s) for s in SEEDS], axis=0)
        p_mean[:, i] = decode_mean(prob, centers)
        p_bayes[:, i] = decode_bayes(prob, centers, TOL[target])
        print("trained", target, flush=True)
    predictions["tree_regression"] = p_reg
    predictions["distribution_mean"] = p_mean
    predictions["distribution_tolerance_bayes"] = p_bayes
    for name, pred in [("tree_regression", p_reg), ("distribution_mean", p_mean), ("distribution_tolerance_bayes", p_bayes)]:
        results["variants"][name] = metrics(pred, Yte)

    # The joint SW==1 gate is fixed at .5, previously selected in development.
    gate = cb_gate(Xtr[lab], dtr[lab], Xte, seed=42)
    gated = p_reg.copy()
    gated[:, 1] = np.where(gate >= 0.5, 1.0, gated[:, 1])
    predictions["tree_regression_joint_sw_gate"] = gated
    results["variants"]["tree_regression_joint_sw_gate"] = metrics(gated, Yte)
    results["variants"]["tree_regression_joint_sw_gate"]["gate_accuracy"] = float(((gate >= .5) == (Yte[:, 1] >= .999)).mean())

    # Per-well outcomes are reported, not just pooled RMSE.
    per_well = {}
    for w in np.unique(Wte):
        m = Wte == w
        per_well[str(int(w))] = {
            name: metrics(pred[m], Yte[m]) for name, pred in predictions.items() if name not in {"truth", "well", "depth"}
        }
    results["per_well"] = per_well
    results["bins"] = bins
    results["seconds"] = time.time() - t0
    results["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (OUT / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "predictions.npz", **predictions)

    rows = []
    for name, score in results["variants"].items():
        row = {"variant": name, **score}
        for w, vals in per_well.items():
            row[f"well_{w}_rmse"] = vals[name]["rmse_all3"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)

    # One paper-ready figure, generated only from this run's results.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    names = ["rf_five_log", "tree_regression", "distribution_mean", "distribution_tolerance_bayes", "tree_regression_joint_sw_gate"]
    labels = ["RF five logs", "Tree regression", "Dist. mean", "Dist. Bayes", "Tree + SW gate"]
    colors = ["#9ca3af", "#3973ac", "#56a374", "#be7c36", "#7856a8"]
    vals = [results["variants"][n]["rmse_all3"] for n in names]
    axes[0].bar(labels, vals, color=colors)
    axes[0].set_ylabel("RMSE (VSH + PHIF + SW)")
    axes[0].set_title("Public Volve: pooled blind-well RMSE")
    axes[0].tick_params(axis="x", rotation=25)
    for i, v in enumerate(vals):
        axes[0].text(i, v + .001, f"{v:.4f}", ha="center", fontsize=8)
    axes[0].set_ylim(0, max(vals) * 1.18)
    wells = list(per_well)
    x = np.arange(len(wells))
    for name, label, color in [("rf_five_log", "RF", "#9ca3af"), ("tree_regression", "Tree", "#3973ac"), ("tree_regression_joint_sw_gate", "Tree + gate", "#7856a8")]:
        axes[1].plot(x, [per_well[w][name]["rmse_all3"] for w in wells], marker="o", label=label, color=color)
    axes[1].set_xticks(x, wells)
    axes[1].set_xlabel("Public test well ID")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Per-well variability, not a universal transfer guarantee")
    axes[1].legend(frameon=False)
    fig.savefig(OUT / "fig_public_volve.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_public_volve.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "script": str(Path(__file__).relative_to(MAIN)),
        "script_sha256": sha(Path(__file__)),
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "seconds": results["seconds"],
        "seeds": SEEDS,
        "python": platform.python_version(),
        "data_files": {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted((MAIN / "external/pdda2021").glob("*.csv"))},
        "outputs": {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "experiment_manifest.json"},
        "limitations": [
            "Public test wells have been used in earlier development; not a fresh blind test.",
            "Volve has VSH/PHIF/SW, not the confidential task's POR/PERM/SW targets.",
            "RMSE and the confidential tolerance score are different metrics.",
            "The .5 joint SW gate was selected in earlier development, not re-tuned here.",
            "This experiment does not establish universal cross-domain transfer.",
        ],
    }
    (OUT / "experiment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "seconds": results["seconds"], "variants": results["variants"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
