#!/usr/bin/env python3
"""Main prediction script (v3 tree-sequence hybrid with metric-aligned decoding).

    python predict.py --data_dir ./data --output ./result.json

Optional: --config config.yaml (default), --model_dir models, --metadata models/model_meta.json, --device auto|cuda|cpu.
Per test well: FeatureBuilder (113/237 features; medians restored from model_meta.json, no training data needed)
-> 181 extra lag/zone features -> CatBoost gates, 4-state model, bin-distribution models
-> whole-well sequence networks -> per property: mixture of predictive distributions decoded with the
official-kernel Bayes action; gate = weighted mixture of default probabilities vs threshold -> result.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data import load_split
from src.feature import FeatureBuilder, well_extra_features, feature_sets
from src.model import (TARGETS, load_cb, mean_proba, prep_well, append_base_channels, load_wellnets, predict_well,
                       dist_summary, level2_features)
from src.utils import (BOUNDS, DEFAULTS, TI, TINV, kernel_matrix, dense_actions, hmm_posterior, load_config,
                       select_device, make_rows, write_result)

HERE = Path(__file__).resolve().parent


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", "--data-dir", dest="data_dir", type=Path, required=True, help="directory containing test/ (same layout as the released data)")
    ap.add_argument("--output", type=Path, required=True, help="path of the result.json to write")
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--model_dir", "--model-dir", dest="model_dir", type=Path, default=None)
    ap.add_argument("--metadata", type=Path, default=None)
    ap.add_argument("--device", default=None, help="auto | cuda | cpu (default from config.yaml)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    args.model_dir = args.model_dir or HERE / cfg.get("model_dir", "models")
    args.metadata = args.metadata or HERE / cfg.get("metadata", "models/model_meta.json")
    args.device = args.device or cfg.get("device", "auto")
    return args


def main():
    args = parse_args()
    meta = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    dec = meta["decision"]
    builder = FeatureBuilder.from_metadata(meta)
    gates = {t: load_cb(args.model_dir, s["files"]) for t, s in meta["gates"].items()}
    bins = {t: load_cb(args.model_dir, s["files"]) for t, s in meta["bins"].items()}
    states_models = load_cb(args.model_dir, meta["states"]["files"]) if meta.get("states") else None
    stack_models = load_cb(args.model_dir, meta["stack"]["files"]) if meta.get("stack") else None
    nn_groups, device = {}, None
    if meta.get("nns"):
        device = select_device(args.device)
        for name, spec in meta["nns"].items():
            nn_groups[name] = (spec, load_wellnets(args.model_dir, spec, device))
    result, total_rows = [], 0
    for log_id, frame in load_split(args.data_dir, "test"):
        mats = builder.transform(frame)
        extra = well_extra_features(frame, builder.medians_).to_numpy()
        feats = feature_sets(mats, extra)
        depth = frame["DEPTH"].to_numpy(np.float64); n = len(frame)
        # sequence nets (one output dict per group)
        nn_out = {}
        for name, (spec, nets) in nn_groups.items():
            x = prep_well(frame, spec["gstats"])
            if spec.get("use_base"):
                x = append_base_channels(x, mats.base, spec["fstats"])
            acc = None
            for net in nets:
                r = predict_well(net, x, device)
                acc = r if acc is None else {t: (acc[t][0] + r[t][0], acc[t][1] + r[t][1]) for t in r}
            nn_out[name] = ({t: (acc[t][0] / len(nets), acc[t][1] / len(nets)) for t in acc}, spec)
        # first-level outputs, computed once per well
        post = mean_proba(states_models, feats[meta["states"]["features"]], classes=4) if states_models else None
        state_marg = None if post is None else {"POR": post[:, 0], "PERM": post[:, 0] + post[:, 3], "SW": post[:, 0] + post[:, 2]}
        gate_cb = {t: mean_proba(gates[t], feats[meta["gates"][t]["features"]])[:, 1] for t in TARGETS}
        tree_prob = {}
        for t in TARGETS:
            pc = mean_proba(bins[t], feats[meta["bins"][t]["features"]], classes=meta["bins"][t]["bins"])
            tree_prob[t] = pc / pc.sum(1, keepdims=True)
        stack_marg = None
        if stack_models:
            sp = meta["stack"]
            nn_l2 = {}
            for name in sp["nn_order"]:
                out_nn, spec_nn = nn_out[name]
                g3 = np.c_[[out_nn[t][0] for t in TARGETS]].T
                summ = {t: dist_summary(out_nn[t][1], np.asarray(spec_nn["bins"][t]["centers_t"])) for t in TARGETS}
                nn_l2[name] = (g3, summ)
            tree_l2 = {t: dist_summary(tree_prob[t], np.asarray(meta["bins"][t]["centers_t"])) for t in TARGETS}
            X2 = level2_features(feats["base+extra"], np.c_[gate_cb["POR"], gate_cb["PERM"], gate_cb["SW"]], post, nn_l2, tree_l2, tuple(sp["nn_order"]))
            assert X2.shape[1] == sp["n_features"], (X2.shape, sp["n_features"])
            post2 = mean_proba(stack_models, X2, classes=4)
            stack_marg = {"POR": post2[:, 0], "PERM": post2[:, 0] + post2[:, 3], "SW": post2[:, 0] + post2[:, 2]}
        pred = np.zeros((n, 3))
        for t in TARGETS:
            ti = TI[t]; d = dec[t]
            # continuous value: expected official utility over a dense action grid, mixture of sources
            sup_cb = TINV[t](np.asarray(meta["bins"][t]["centers_t"]))
            grid = dense_actions(sup_cb, t)
            util = 0
            for name, w in zip(d["dist_sources"], d["cont_w"]):
                if w == 0:
                    continue
                if name.startswith("nn"):
                    out_nn, spec_nn = nn_out[name]
                    g_nn, p_nn = out_nn[t]; p_nn = p_nn / p_nn.sum(1, keepdims=True)
                    sup = TINV[t](np.asarray(spec_nn["bins"][t]["centers_t"]))
                    util = util + w * (p_nn @ kernel_matrix(grid, sup, t).T)
                else:
                    util = util + w * (tree_prob[t] @ kernel_matrix(grid, sup_cb, t).T)
            cont = np.clip(grid[np.argmax(util, 1)], *BOUNDS[t])
            # gate: mixture of default probabilities, optional HMM smoothing, single threshold
            gate = 0
            for name, w in zip(d["gate_sources"], d["gate_w"]):
                if w == 0:
                    continue
                if name.startswith("nn"):
                    gate = gate + w * nn_out[name][0][t][0]
                elif name == "states":
                    gate = gate + w * state_marg[t]
                elif name == "stack_states":
                    gate = gate + w * stack_marg[t]
                else:
                    gate = gate + w * gate_cb[t]
            if d.get("hmm_stay") is not None and n >= 3:
                gate = hmm_posterior(gate, stay=d["hmm_stay"])
            pred[:, ti] = np.where(gate >= d["threshold"], DEFAULTS[t], cont)
        rows = make_rows(depth, pred, log_id)
        result.append({"logId": log_id, "predictions": rows}); total_rows += n
        print(log_id, n, flush=True)
    write_result(args.output, result, meta.get("model_name", "welllog_v3"))
    print("wrote", args.output, "wells", len(result), "rows", total_rows)


if __name__ == "__main__":
    main()
