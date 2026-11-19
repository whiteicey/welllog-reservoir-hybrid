"""Train the actual v4 local--global dual-backbone architecture on public Volve.

No internal weights, labels, checkpoints or training rows are used. The network,
conditional-density heads, two-member routing and losses are imported from the
original v4 runtime. Only the public target mapping and evaluation are adapted.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNTIME = HERE / "runtime"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RUNTIME))

from run_transfer_pdda2021 import load, TARGETS, rmse3, rmse_per, tol_acc
from train_invariant import feature_contract
from paired_local_context_router_sequence import Network, retained_indices, calibration_indices
from paired_backbone_sequence import objective
from por_t3_patch_loss import head_loss, probability
from multiscale_sequence import window_indices

TOL = (0.08, 0.05, 0.08)  # PHIF, SW, VSH; target order is TARGETS.
TRANSFORM = [np.log1p, lambda x: x, np.log1p]
INVERSE = [np.expm1, np.expm1, lambda x: x]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def public_audit(tr_wells, te_wells) -> dict:
    ytr = np.concatenate([f[TARGETS].to_numpy(float) for _, f in tr_wells])
    lab = np.isfinite(ytr).all(1)
    result = {
        "train_rows": len(ytr), "complete_label_rows": int(lab.sum()),
        "train_wells": [int(w) for w, _ in tr_wells], "test_wells": [int(w) for w, _ in te_wells],
        "test_rows": int(sum(len(f) for _, f in te_wells)),
        "target_order": TARGETS,
        "state_rules": "PHIF and VSH have no default atom; SW==1 is the public SW state",
        "label_range_note": "Labels are not altered. Some VSH labels fall outside the nominal [0,1] range; predictions are clipped only during RMSE evaluation.",
        "targets": {},
    }
    for j, target in enumerate(TARGETS):
        y = ytr[lab, j]
        result["targets"][target] = {
            "min": float(y.min()), "max": float(y.max()),
            "outside_0_1_rows": int(((y < 0) | (y > 1)).sum()),
            "zero_rows": int((y == 0).sum()),
            "one_rows": int((y == 1).sum()),
        }
    return result


class PublicData:
    """Feature/target preparation fitted only on the permitted public train wells."""
    def __init__(self, tr_wells, te_wells, fit_indices):
        self.tr_wells = tr_wells
        self.te_wells = te_wells
        self.fit_indices = list(fit_indices)
        frames = tr_wells + te_wells
        from package.src.feature import FeatureBuilder
        self.builder = FeatureBuilder().fit([frames[i][1] for i in self.fit_indices])
        contract = feature_contract()
        indices = np.asarray(contract["selected_indices"], dtype=np.int64)
        names = self.builder.enhanced_feature_names_
        assert [names[i] for i in indices] == contract["whitelist"]
        self.contract = contract
        self.view_indices = retained_indices(contract["whitelist"])
        self.numeric_indices, self.conditioning_indices = calibration_indices(contract["whitelist"])
        matrices = [self.builder.transform(frame).enhanced[:, indices] for _, frame in frames]
        nfeat = len(indices)
        self.fm = np.zeros(nfeat, np.float32)
        self.fs = np.ones(nfeat, np.float32)
        for c in range(nfeat):
            values = np.concatenate([matrices[i][:, c] for i in self.fit_indices])
            finite = values[np.isfinite(values)]
            if len(finite):
                q = np.quantile(finite, np.linspace(0, 1, 17))
                self.fm[c] = q[8]
                self.fs[c] = max(float(q[12] - q[4]), 1e-3)
        frames_x = []
        for matrix in matrices:
            x = np.clip((matrix - self.fm) / self.fs, -8, 8).astype(np.float32)
            # Neutral encoding of unavailable values, not a fabricated measurement.
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            frames_x.append(x)
        self.x = torch.from_numpy(np.concatenate(frames_x)).to(device="cuda", dtype=torch.float16)
        self.y = np.concatenate([f[TARGETS].to_numpy(float) for _, f in frames])
        self.well = np.concatenate([np.full(len(f), w, np.int32) for w, f in frames])
        self.depth_np = np.concatenate([f["DEPTH"].to_numpy(np.float32) for _, f in frames])
        self.depth = torch.from_numpy(self.depth_np).cuda()
        self.starts = np.r_[0, np.cumsum([len(f) for _, f in frames])]
        self.lower = torch.from_numpy(np.repeat(self.starts[:-1], np.diff(self.starts))).cuda()
        self.upper = torch.from_numpy(np.repeat(self.starts[1:], np.diff(self.starts))).cuda()
        self.labeled_np = np.isfinite(self.y).all(1)
        self.labeled = torch.from_numpy(self.labeled_np).cuda()
        default = np.zeros(self.y.shape, dtype=bool)
        default[:, 1] = np.nan_to_num(self.y[:, 1], nan=0.0) >= .999
        self.default = torch.from_numpy(default).cuda()
        train_mask = np.isin(self.well, [frames[i][0] for i in self.fit_indices])
        self.train_mask_np = train_mask
        self.train_mask = torch.from_numpy(train_mask).cuda()
        self.test_mask_np = ~train_mask
        fit_depth = self.depth_np[train_mask & self.labeled_np]
        q = np.percentile(fit_depth, [25, 50, 75])
        self.depth_median = float(q[1])
        self.depth_scale = max(float(q[2] - q[0]), .1)
        self.absolute = ((self.depth - self.depth_median) / self.depth_scale).clamp(-8, 8)
        self.medians, self.scales, self.initial = [], [], []
        self.centers, self.edges = [], []
        z = np.zeros(self.y.shape, dtype=np.float32)
        fit_rows = train_mask & self.labeled_np
        for j in range(3):
            transformed = TRANSFORM[j](self.y[:, j])
            use = fit_rows & ~default[:, j]
            fit = transformed[use]
            if not len(fit):
                raise ValueError(f"No non-default public training labels for {TARGETS[j]}")
            q25, q50, q75 = np.percentile(fit, [25, 50, 75])
            scale = max(float(q75 - q25), .1)
            self.medians.append(float(q50)); self.scales.append(scale)
            z[:, j] = (transformed - q50) / scale
            self.initial.append(np.quantile((fit - q50) / scale, np.linspace(.05, .95, 8)))
            support = np.linspace(fit.min(), fit.max(), 256)
            self.centers.append(INVERSE[j](support))
            self.edges.append((np.r_[-np.inf, (support[:-1] + support[1:]) / 2, np.inf] - q50) / scale)
        z[~self.labeled_np] = 0
        self.labels = torch.from_numpy(z).cuda()
        self.train_labeled = torch.from_numpy(np.flatnonzero(fit_rows)).cuda()
        if not len(self.train_labeled):
            raise ValueError("No public training labels")
        print("DATA", json.dumps({"fit_wells": [frames[i][0] for i in self.fit_indices], "features": nfeat,
                                    "fit_input_rows": int(train_mask.sum()), "fit_label_rows": int(fit_rows.sum()),
                                    "neutral_missing_encoding": True}), flush=True)


def metrics(pred, true):
    return {"rmse_all3": rmse3(pred, true), **rmse_per(pred, true),
            "tol_PHIF": tol_acc(pred[:, 0], true[:, 0], .08),
            "tol_SW": tol_acc(np.clip(pred[:, 1], 0, 1), true[:, 1], .05),
            "tol_VSH": tol_acc(pred[:, 2], true[:, 2], .08)}


def predict(model, data: PublicData, eval_wells):
    """Posterior mean for RMSE and kernel actions only for the auxiliary diagnostic."""
    model.eval()
    centers, anchors = [], []
    for wi, (well_id, frame) in enumerate(data.tr_wells + data.te_wells):
        if int(well_id) not in eval_wells:
            continue
        cc = np.arange(data.starts[wi], data.starts[wi + 1], 128) + 64
        centers.extend(cc)
        anchors.extend(np.minimum(cc, data.starts[wi + 1] - 1))
    mean = np.full((len(data.y), 3), np.nan)
    bayes = np.full((len(data.y), 3), np.nan)
    covered = np.zeros(len(data.y), dtype=np.int8)
    edges = [torch.as_tensor(e, device="cuda", dtype=torch.float32) for e in data.edges]
    with torch.no_grad():
        for begin in range(0, len(centers), 32):
            cc = torch.tensor(centers[begin:begin + 32], device="cuda")
            anchor = torch.tensor(anchors[begin:begin + 32], device="cuda")
            ix, exists = window_indices(cc, data.lower, data.upper, anchors=anchor)
            relative = torch.tanh((data.depth[ix] - data.depth[anchor, None]) / 10)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, _, (go, raw) = model.prediction_parts(data.x[ix], relative, exists, absolute_depth=data.absolute[ix])
            original = ix[:, 64:192].cpu().numpy()
            use = exists[:, 64:192].cpu().numpy()
            rows = original[use]
            if not len(rows):
                continue
            assert (covered[rows] == 0).all()
            covered[rows] = 1
            select = torch.from_numpy(use).cuda()
            gate = go[:, 64:192][select].float().sigmoid()
            rr = raw[:, 64:192][select].float()
            weight = rr[:, :, 0].softmax(-1)
            mu, sd = rr[:, :, 1], F.softplus(rr[:, :, 2]) + .025
            for j in range(3):
                cp = probability(j, weight[:, j], mu[:, j], sd[:, j], edges[j])
                ct = data.centers[j]
                if j == 1:
                    joint = torch.cat([gate[:, j:j + 1], (1 - gate[:, j:j + 1]) * cp], -1)
                    support = np.r_[1.0, ct]
                else:
                    joint = cp
                    support = ct
                mean[rows, j] = (joint @ torch.as_tensor(support, device="cuda", dtype=torch.float32)).cpu().numpy()
                action = np.unique(np.r_[support, np.linspace(ct[0], ct[-1], 600)])
                utility = np.maximum(0, 1 - np.abs(action[:, None] - support[None, :]) /
                                     (TOL[j] * (np.abs(support[None, :]) + 1e-7)))
                utility = torch.as_tensor(utility, device="cuda", dtype=torch.float32)
                bayes[rows, j] = action[(joint @ utility.T).argmax(-1).cpu().numpy()]
    expected = np.isin(data.well, list(eval_wells))
    assert (covered[expected] == 1).all() and np.isfinite(mean[expected]).all() and np.isfinite(bayes[expected]).all()
    return mean, bayes


def train_one(data: PublicData, cfg, name: str, out: Path, fixed_epochs=None):
    seed = cfg["seed"] + (0 if name == "final" else int(name.split("_")[-1]))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = Network(initial_means=np.stack(data.initial), retain_indices=data.view_indices,
                    numeric_indices=data.numeric_indices, conditioning_indices=data.conditioning_indices).cuda()
    expert = list(model.members.parameters())
    router = list(model.router.parameters())
    opt = torch.optim.AdamW(expert, lr=cfg["learning_rate"], weight_decay=.01)
    ropt = torch.optim.AdamW(router, lr=cfg["learning_rate"], weight_decay=.01)
    epochs = fixed_epochs or cfg["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * cfg["steps"], eta_min=1e-5)
    rsched = torch.optim.lr_scheduler.CosineAnnealingLR(ropt, epochs * cfg["steps"], eta_min=1e-5)
    core = torch.zeros(1, 256, device="cuda", dtype=torch.bool); core[:, 64:192] = True
    best_rmse = float("inf"); best_epoch = 0; best_state = None; best_pred = None; best_bayes = None; bad = 0
    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True); ropt.zero_grad(set_to_none=True)
        total = 0.0
        for step in range(cfg["steps"]):
            anchors = data.train_labeled[torch.randint(len(data.train_labeled), (cfg["windows_per_step"],), device="cuda")]
            ix, exists = window_indices(anchors, data.lower, data.upper)
            relative = torch.tanh((data.depth[ix] - data.depth[anchors, None]) / 10)
            supervised = exists & core & data.labeled[ix] & data.train_mask[ix]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(data.x[ix], relative, exists, auxiliary=True, absolute_depth=data.absolute[ix])
            weights = torch.ones_like(supervised, dtype=torch.float32)
            expert_loss = objective(outputs[0], data.labels[ix], data.default[ix], supervised,
                                    True, weights=weights, epoch=epoch)
            router_loss = head_loss(outputs[1], data.labels[ix], data.default[ix], supervised, weights)
            loss = expert_loss + router_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at {name} epoch {epoch + 1} step {step + 1}")
            opt.zero_grad(set_to_none=True); ropt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(expert, 2)
            torch.nn.utils.clip_grad_norm_(router, 2)
            opt.step(); ropt.step(); sched.step(); rsched.step()
            total += float(loss.detach())
        print("EPOCH", name, epoch + 1, "loss", round(total / cfg["steps"], 6), flush=True)
        if fixed_epochs is None:
            eval_wells = sorted(set(data.well[~data.train_mask_np].tolist()))
            mean, bayes = predict(model, data, eval_wells)
            use = (~data.train_mask_np) & data.labeled_np
            score = metrics(mean[use], data.y[use])
            history.append({"epoch": epoch + 1, "loss": total / cfg["steps"], "metrics": score})
            print("VALIDATION", name, epoch + 1, json.dumps(score), flush=True)
            if score["rmse_all3"] < best_rmse:
                best_rmse = score["rmse_all3"]; best_epoch = epoch + 1; bad = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_pred, best_bayes = mean.copy(), bayes.copy()
            else:
                bad += 1
            if epoch + 1 >= cfg["min_epochs"] and bad >= cfg["patience"]:
                break
    if fixed_epochs is None:
        assert best_state is not None and best_epoch > 0
        model.load_state_dict(best_state)
        mean, bayes = best_pred, best_bayes
        score = best_rmse
    else:
        best_epoch = fixed_epochs
        eval_wells = sorted(set(data.well[data.test_mask_np].tolist()))
        mean, bayes = predict(model, data, eval_wells)
        use = data.test_mask_np & data.labeled_np
        score = metrics(mean[use], data.y[use])
    checkpoint = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "epoch": best_epoch, "best_epoch": best_epoch, "best_validation_score": score, "config": cfg,
        "feature_median": data.fm, "feature_scale": data.fs,
        "absolute_depth_median": data.depth_median, "absolute_depth_scale": data.depth_scale,
        "selected_feature_names": data.contract["whitelist"], "selected_feature_indices": data.contract["selected_indices"],
        "builder_medians": data.builder.medians_, "contract": data.contract,
        "target_medians": data.medians, "target_scales": data.scales,
        "regression_median": data.medians, "regression_scale": data.scales,
        "centers": data.centers, "density_edges": data.edges,
        "depth_median": data.depth_median, "depth_scale": data.depth_scale,
        "fit_wells": [data.tr_wells[i][0] for i in data.fit_indices],
        "score": score,
    }
    torch.save(checkpoint, out / f"{name}.pt")
    result = {"name": name, "best_epoch": best_epoch, "seconds": time.perf_counter() - started,
              "score": score, "history": history, "mean": mean, "bayes": bayes}
    print("TRAIN_RESULT", name, json.dumps({"epoch": best_epoch, "score": score, "seconds": result["seconds"]}), flush=True)
    return result


def make_figure(report, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scores = report["public_test"]["variants"]
    names = ["rf_five_log", "target_tree", "target_tree_gate", "v4_root_dual_backbone_mean", "v4_root_dual_backbone_bayes"]
    labels = ["RF", "Tree", "Tree + gate", "v4 mean", "v4 Bayes"]
    vals = [scores[n]["rmse_all3"] for n in names]
    colors = ["#9ca3af", "#3973ac", "#56a374", "#7856a8", "#be7c36"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    ax[0].bar(labels, vals, color=colors)
    ax[0].set_ylabel("RMSE")
    ax[0].set_title("Public Volve: v4 root dual-backbone evaluation")
    ax[0].tick_params(axis="x", rotation=20)
    for i, v in enumerate(vals): ax[0].text(i, v + .002, f"{v:.4f}", ha="center", fontsize=8)
    ax[0].set_ylim(0, max(vals) * 1.18)
    wells = report["public_test"]["per_well"]
    x = np.arange(len(wells))
    for name, label, color in [("target_tree_gate", "Tree + gate", "#56a374"), ("v4_root_dual_backbone_mean", "v4 mean", "#7856a8")]:
        ax[1].plot(x, [wells[w][name]["rmse_all3"] for w in wells], marker="o", label=label, color=color)
    ax[1].set_xticks(x, list(wells))
    ax[1].set_xlabel("Public test well ID")
    ax[1].set_ylabel("RMSE")
    ax[1].set_title("Per-well comparison")
    ax[1].legend(frameon=False)
    fig.savefig(out / "fig_v4_public.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_v4_public.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "results" / "v4_public_20261113")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--windows-per-step", type=int, default=32)
    args = ap.parse_args()
    if args.out.exists() and not args.smoke:
        raise FileExistsError(f"Refuse to overwrite existing experiment directory: {args.out}")
    if args.smoke:
        args.out = HERE / "results" / "smoke_v4_public"
        args.epochs, args.steps, args.windows_per_step = 1, 2, 4
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = {
        "seed": 20261113, "epochs": args.epochs, "steps": args.steps,
        "windows_per_step": args.windows_per_step, "learning_rate": 0.0006,
        "min_epochs": 8, "patience": 4, "window_length": 256,
        "architecture": "paired_local_context_router_sequence.Network; two independent local-global backbones, calibrated68 view, detached local router, conditional-density heads",
        "target_mapping": {"POR": "PHIF", "PERM": "VSH", "SW": "SW"},
        "default_mapping": "PHIF/VSH no default atom; SW=1 state only",
        "target_transforms": ["log1p PHIF", "identity SW", "log1p VSH"],
        "missing_values": "public -9999 -> NaN; neutral zero after training-well robust scaling with missingness indicators",
        "training_selection": "3-fold well-grouped CV, select best public RMSE checkpoint; final epoch is median of CV best epochs",
        "test_rule": "Fit all nine public train wells after CV selection, then evaluate released test wells once; no test labels used for model, feature, epoch or decoder selection",
        "internal_weights_or_data": False,
        "smoke": args.smoke,
        "source_sha256": sha(Path(__file__)),
    }
    json_write(args.out / "config_frozen.json", cfg)
    tr_wells, te_wells = load()
    json_write(args.out / "data_audit.json", public_audit(tr_wells, te_wells))
    folds = list(GroupKFold(3).split(np.arange(len(tr_wells)), groups=np.arange(len(tr_wells))))
    results = []
    for k, (fit, val) in enumerate(folds):
        if args.smoke and k > 0:
            break
        fit_wells = [tr_wells[i] for i in fit]
        val_wells = [tr_wells[i] for i in val]
        data = PublicData(fit_wells, val_wells, list(range(len(fit_wells))))
        result = train_one(data, cfg, f"fold_{k + 1}", args.out)
        results.append(result)
        del data, result
        gc.collect(); torch.cuda.empty_cache()
    best_epochs = [r["best_epoch"] for r in results]
    if args.smoke:
        json_write(args.out / "smoke_report.json", {"passed": True, "epochs": best_epochs})
        print("SMOKE_PASS", best_epochs, flush=True)
        return
    final_epochs = int(round(float(np.median(best_epochs))))
    cfg["final_epochs"] = final_epochs
    cfg["cv_best_epochs"] = best_epochs
    json_write(args.out / "config_frozen.json", cfg)
    final_data = PublicData(tr_wells, te_wells, list(range(len(tr_wells))))
    final = train_one(final_data, cfg, "final", args.out, fixed_epochs=final_epochs)
    test_mask = final_data.test_mask_np & final_data.labeled_np
    public_metrics = metrics(final["mean"][test_mask], final_data.y[test_mask])
    bayes_metrics = metrics(final["bayes"][test_mask], final_data.y[test_mask])
    per_well = {}
    for w in sorted(set(final_data.well[final_data.test_mask_np].tolist())):
        use = (final_data.well == w) & final_data.labeled_np
        per_well[str(w)] = {
            "v4_root_dual_backbone_mean": metrics(final["mean"][use], final_data.y[use]),
            "v4_root_dual_backbone_bayes": metrics(final["bayes"][use], final_data.y[use]),
        }
    variants = {
        "rf_five_log": {"rmse_all3": 0.1108188578187798},
        "target_tree": {"rmse_all3": 0.1078851368016146},
        "target_tree_gate": {"rmse_all3": 0.10618727913336519},
        "v4_root_dual_backbone_mean": public_metrics,
        "v4_root_dual_backbone_bayes": bayes_metrics,
    }
    for w, values in per_well.items():
        use = final_data.well == int(w)
        values["target_tree_gate"] = {"rmse_all3": None}
    report = {
        "config": cfg,
        "cv": [{"fold": r["name"], "best_epoch": r["best_epoch"], "score": r["score"], "history": r["history"]} for r in results],
        "public_test": {"metrics": public_metrics, "bayes_metrics": bayes_metrics, "variants": variants, "per_well": per_well},
        "test_wells": [int(w) for w, _ in te_wells],
        "no_internal_data_or_weights": True,
        "source_script_sha256": sha(Path(__file__)),
        "runtime_sources": {p.name: sha(p) for p in RUNTIME.glob("*.py")},
    }
    # Preserve row-level predictions and the comparison values from the earlier public run.
    np.savez_compressed(args.out / "public_test_predictions.npz", mean=final["mean"], bayes=final["bayes"],
                        truth=final_data.y, well=final_data.well, depth=final_data.depth_np)
    for w, values in per_well.items():
        mask = final_data.well == int(w)
        values["target_tree_gate"] = {"rmse_all3": None, "note": "Earlier public tree result is retained in paper_workspace/public_volve/metrics.json"}
    # Fill exact earlier tree+gate per-well values without retraining.
    prior_path = ROOT / "paper_workspace" / "public_volve" / "metrics.json"
    if prior_path.exists():
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        for w in per_well:
            if w in prior["per_well"]:
                per_well[w]["target_tree_gate"] = {"rmse_all3": prior["per_well"][w]["tree_regression_joint_sw_gate"]["rmse_all3"]}
    json_write(args.out / "metrics.json", report)
    make_figure(report, args.out)
    json_write(args.out / "training_report.json", {"status": "complete", "config": cfg, "metrics": report["public_test"]})
    print("FINAL_PUBLIC_METRICS", json.dumps(report["public_test"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
