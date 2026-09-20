"""Train the final v3 models on all 80 wells, driven by a config JSON that mirrors the CV experiments.

Config keys (see final_cfg.json):
  cb_seeds            seeds for CatBoost models (as in CV: 42, 2026)
  gates    {target: {features, iterations}}        per-target default gate (CatBoost logloss)
  bins     {target: {features, bins, iterations, depth, lr, l2}}  per-target distribution model
  states   {features, iterations}                  4-state joint reservoir-state model (optional)
  nn       {seeds, train_args{...}, use_base}       whole-well sequence net (optional)
  decision {target: {cont_w, gate_w, hmm_stay, threshold, dist_sources, gate_sources}}  from assemble.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, Pool

from extra_features import build_extra
from harness import default_mask, load_or_build_cache, quantile_bins

TI = {"POR": 0, "PERM": 1, "SW": 2}
TFWD = {"POR": np.log1p, "PERM": np.log10, "SW": lambda v: v}


def states(y):
    return np.select([y[:, 0] <= 0.1, y[:, 2] >= 99.9, y[:, 1] <= 0.01], [0, 2, 3], 1)


def gpu(seed, depth=8, lr=0.04, l2=3.0):
    return dict(task_type="GPU", devices="0", max_bin=128, random_seed=seed, learning_rate=lr, depth=depth, l2_leaf_reg=l2,
                bootstrap_type="Bayesian", allow_writing_files=False, verbose=0, gpu_ram_part=0.45)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/final"))
    ap.add_argument("--skip-nn", action="store_true")
    ap.add_argument("--skip-cb", action="store_true")
    ap.add_argument("--only-stack", action="store_true", help="train only the level-2 stack model (gates/bins/states must already be in model_meta.json)")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    mdir = args.out_dir / "models"; mdir.mkdir(parents=True, exist_ok=True)
    meta_path = args.out_dir / "model_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    C = load_or_build_cache(Path("data"), Path("cache"))
    y = C["tr_y"]
    etr, _, extra_names = build_extra(Path("data"), Path("cache"), C["meta"]["medians"])
    feats = {"base": C["tr_base"], "enh": C["tr_enh"], "base+extra": np.c_[C["tr_base"], etr], "enh+extra": np.c_[C["tr_enh"], etr]}
    meta.update({"model_name": "welllog_v3_distribution_tree_sequence_hybrid", "medians": C["meta"]["medians"], "extra_names": extra_names,
                 "base_feature_names": C["meta"]["base_feature_names"], "enhanced_feature_names": C["meta"]["enhanced_feature_names"],
                 "decision": cfg["decision"], "cb_seeds": cfg["cb_seeds"], "training_rows": int(len(y)), "training_wells": 80})
    t0 = time.time()
    if not args.skip_cb and not args.only_stack:
        meta["gates"], meta["bins"] = {}, {}
        for t, spec in cfg["gates"].items():
            ti = TI[t]; X = feats[spec["features"]]; d = default_mask(y[:, ti], t)
            files = []
            for seed in cfg["cb_seeds"]:
                m = CatBoostClassifier(iterations=spec["iterations"], loss_function="Logloss", **gpu(seed + spec.get("seed_offset", 0)))
                m.fit(Pool(X, label=d.astype(np.int8)))
                p = mdir / f"{t}_gate_seed{seed}.cbm"; m.save_model(p); files.append(p.name)
            meta["gates"][t] = {"features": spec["features"], "files": files}
            print(t, "gate done", f"{time.time() - t0:.0f}s", flush=True)
        for t, spec in cfg["bins"].items():
            ti = TI[t]; X = feats[spec["features"]]; d = default_mask(y[:, ti], t)
            labels, centers, edges = quantile_bins(TFWD[t](y[~d, ti]), spec["bins"])
            assert len(centers) == spec["bins"], (t, len(centers))
            files = []
            for seed in cfg["cb_seeds"]:
                m = CatBoostClassifier(iterations=spec["iterations"], loss_function="MultiClass",
                                       **gpu(seed + 100, depth=spec.get("depth", 8), lr=spec.get("lr", 0.04), l2=spec.get("l2", 3.0)))
                m.fit(Pool(X[~d], label=labels))
                order = np.argsort(m.classes_.astype(int)); assert (m.classes_.astype(int)[order] == np.arange(spec["bins"])).all()
                p = mdir / f"{t}_bins_seed{seed}.cbm"; m.save_model(p); files.append(p.name)
            meta["bins"][t] = {"features": spec["features"], "bins": spec["bins"], "centers_t": centers.tolist(), "files": files}
            print(t, "bins done", f"{time.time() - t0:.0f}s", flush=True)
        if cfg.get("states"):
            spec = cfg["states"]; X = feats[spec["features"]]; S = states(y); files = []
            for seed in cfg["cb_seeds"]:
                m = CatBoostClassifier(iterations=spec["iterations"], loss_function="MultiClass", **gpu(seed, lr=0.05))
                m.fit(Pool(X, label=S))
                order = np.argsort(m.classes_.astype(int)); assert (m.classes_.astype(int)[order] == np.arange(4)).all()
                p = mdir / f"STATES_seed{seed}.cbm"; m.save_model(p); files.append(p.name)
            meta["states"] = {"features": spec["features"], "files": files}
            print("states done", f"{time.time() - t0:.0f}s", flush=True)
        meta_path.write_text(json.dumps(meta, indent=1))
    if not args.skip_cb and cfg.get("stack"):
        # level-2 state model trained on the CV out-of-fold outputs of the first-level sources
        import stack_states as SS
        spec = cfg["stack"]; nn_order = tuple(spec["nn_order"])
        V2 = np.load("outputs/cv_v2/oof.npz"); S1 = np.load("outputs/states_base_extra/oof.npz")
        gates_tr = np.c_[V2["por_gate"], V2["perm_gate"], V2["sw_gate"]]
        folds = C["tr_fold"]
        nn_tr = {}
        for name in nn_order:
            O = np.load(f"outputs/{name}/oof.npz")
            g = np.c_[[O[f"{t}_gate"] for t in SS.TARGETS]].T
            summ = {}
            for t in SS.TARGETS:
                sm = np.zeros((len(y), 4), np.float32)
                for f in range(5):
                    va = folds == f
                    sm[va] = SS.dist_summary(O[f"{t}_prob"][va], O[f"{t}_centers"][f])
                summ[t] = sm
            nn_tr[name] = (g, summ)
        tree_tr = {}
        for t, name in spec["tree_dist"].items():
            O = np.load(f"outputs/{name}/oof.npz")
            sm = np.zeros((len(y), 4), np.float32)
            for f in range(5):
                va = folds == f
                sm[va] = SS.dist_summary(O["prob"][va], O["centers"][f])
            tree_tr[t] = sm
        Xs = SS.level2_features(np.c_[C["tr_base"], etr], gates_tr, S1["post"], nn_tr, tree_tr, nn_order)
        Sy = states(y); files = []
        for seed in cfg["cb_seeds"]:
            m = CatBoostClassifier(iterations=spec["iterations"], loss_function="MultiClass", **gpu(seed, lr=0.05))
            m.fit(Pool(Xs, label=Sy))
            order = np.argsort(m.classes_.astype(int)); assert (m.classes_.astype(int)[order] == np.arange(4)).all()
            pth = mdir / f"STACK_seed{seed}.cbm"; m.save_model(pth); files.append(pth.name)
        meta["stack"] = {"files": files, "nn_order": list(nn_order), "tree_dist": spec["tree_dist"], "n_features": int(Xs.shape[1])}
        meta_path.write_text(json.dumps(meta, indent=1))
        print("stack done", Xs.shape, f"{time.time() - t0:.0f}s", flush=True)
    if not args.skip_nn and cfg.get("nns"):
        import torch
        import nn_seq
        device = torch.device("cuda")
        meta.setdefault("nns", {})
        for ncfg in cfg["nns"]:
            if ncfg["name"] in meta["nns"] and not ncfg.get("retrain", False):
                print("nn", ncfg["name"], "already trained, skipping", flush=True); continue
            ns = argparse.Namespace(**ncfg["train_args"])
            D = nn_seq.WellData(Path("data"), use_base=bool(ncfg.get("use_base", 0)))
            bins = nn_seq.make_bins(D.ytr, list(range(len(D.xtr))))
            pseudo = None
            if ncfg.get("pseudo_test"):
                res = json.loads(Path(ncfg["pseudo_test"]).read_text(encoding="utf-8"))
                preds = {e["logId"]: e["predictions"] for e in res["resultData"]}
                pseudo = {}
                for wi, (wid, frame) in enumerate(D.test):
                    yp = np.array([[r["POR"], r["PERM"], r["SW"]] for r in preds[wid]], dtype=np.float64)
                    assert len(yp) == D.xte[wi].shape[1]
                    pseudo[f"test{wi}"] = (D.xte[wi], yp)
                print("nn", ncfg["name"], "pseudo-labelled test wells:", len(pseudo), flush=True)
            nn_files = []
            for seed in ncfg["seeds"]:
                net = nn_seq.train_fold(D, bins, list(range(len(D.xtr))), ns, device, seed, pseudo=pseudo)
                p = mdir / f"{ncfg['name']}_seed{seed}.pt"; torch.save(net.state_dict(), p); nn_files.append(p.name)
                print("nn", ncfg["name"], "seed", seed, "done", f"{time.time() - t0:.0f}s", flush=True)
            meta["nns"][ncfg["name"]] = {"files": nn_files, "gstats": D.gstats, "n_in": D.n_in, "channels": ns.channels, "dropout": ns.dropout,
                                         "kernel": getattr(ns, "kernel", 5), "use_base": bool(ncfg.get("use_base", 0)), "fstats": D.fstats,
                                         "bins": {t: {"centers_t": bins[t]["centers"].tolist()} for t in bins}, "nbins": nn_seq.NBINS}
            meta_path.write_text(json.dumps(meta, indent=1))
    meta_path.write_text(json.dumps(meta, indent=1))
    print("saved", args.out_dir, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
