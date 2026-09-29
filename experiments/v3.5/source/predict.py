"""v3 推理：树分布/门控/状态 + 两组 WellNet 对测试井推理，
按 decision.json 融合，输出官方 result.json。

用法:
    python predict.py --data_dir ./data/test --out_dir ./outputs --output ./result.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (TARGETS, N_FOLDS, SENTINEL, N_BINS, NN_CONFIGS,
                        DIST_SEEDS, GATE_SEEDS, STATES_SEEDS)
from src.data import read_log_file, interp_missing, label_arrays
from src.features import base_enhanced_features, extra_features
from src.nn_input import build_nn_input
from src.wellnet import WellNet
from src.scoring import action_grid, utility_matrix
from train_trees import DIST_FEATS, GATE_FEATS
from train_nn import infer_well


def load_cbm(path):
    m = CatBoostClassifier()
    m.load_model(path)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="./outputs")
    ap.add_argument("--output", default="./result.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(args.out_dir, "trees_meta.json"),
              encoding="utf-8") as f:
        tmeta = json.load(f)
    with open(os.path.join(args.out_dir, "bins.json"), encoding="utf-8") as f:
        bins_by_fold = json.load(f)
    with open(os.path.join(args.out_dir, "decision.json"),
              encoding="utf-8") as f:
        decision = json.load(f)["decision"]

    # ---- 加载树模型 ----
    print("加载 CatBoost 模型 ...")
    dist_models = {t: {k: [load_cbm(os.path.join(
        args.out_dir, f"dist_{t}_fold{k}_seed{s}.cbm"))
        for s in DIST_SEEDS] for k in range(N_FOLDS)} for t in TARGETS}
    gate_models = {t: {k: [load_cbm(os.path.join(
        args.out_dir, f"gate_{t}_fold{k}_seed{s}.cbm"))
        for s in GATE_SEEDS] for k in range(N_FOLDS)} for t in TARGETS}
    state_models = {k: [load_cbm(os.path.join(
        args.out_dir, f"states_fold{k}_seed{s}.cbm"))
        for s in STATES_SEEDS] for k in range(N_FOLDS)}

    # ---- 加载 NN ----
    nn_models, nn_meta = {}, {}
    for cfg_name, cfg in NN_CONFIGS.items():
        meta_path = os.path.join(args.out_dir, f"nn_{cfg_name}_meta.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path, encoding="utf-8") as f:
            nn_meta[cfg_name] = json.load(f)
        ms = []
        for k in range(N_FOLDS):
            for s in cfg["seeds"]:
                path = os.path.join(args.out_dir,
                                    f"nn_{cfg_name}_fold{k}_seed{s}.pt")
                m = WellNet(nn_meta[cfg_name]["n_in"]).to(device)
                m.load_state_dict(torch.load(path, map_location=device))
                m.eval()
                ms.append(m)
        nn_models[cfg_name] = ms
        print(f"加载 nn_{cfg_name}: {len(ms)} 个模型")

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.txt")))
    if not files:
        raise FileNotFoundError(f"目录中没有 txt 文件: {args.data_dir}")

    result_data = []
    for path in files:
        print(f"预测 {os.path.basename(path)} ...")
        df = read_log_file(path)
        df = interp_missing(df, tmeta["curve_medians"])
        Xb_full, Xe_full = base_enhanced_features(df)
        Xb = Xb_full[tmeta["feat_base"]]
        Xen = Xe_full[tmeta["feat_enh"]]
        Xx = extra_features(df)[tmeta["feat_extra"]]
        feat_map = {"b": Xb, "e": Xen, "be": pd.concat([Xb, Xx], axis=1)}
        n = len(df)

        # ---- 树推理（效用对 5 折求和 → 融合层直接用） ----
        tree_eu = {}      # t -> (n, A)
        tree_gate = {t: np.zeros(n) for t in TARGETS}
        actions = {}
        for t in TARGETS:
            Xd = feat_map[DIST_FEATS[t]]
            Xg = feat_map[GATE_FEATS[t]]
            centers0 = np.array(bins_by_fold[t]["0"]["centers"])
            actions[t] = action_grid(t, centers0)
            eu = np.zeros((n, len(actions[t])))
            for k in range(N_FOLDS):
                p = np.mean([m.predict_proba(Xd)
                             for m in dist_models[t][k]], axis=0)
                centers = np.array(bins_by_fold[t][str(k)]["centers"])
                p = p[:, :len(centers)].astype(np.float64)
                p /= p.sum(axis=1, keepdims=True).clip(min=1e-9)
                K = utility_matrix(t, actions[t], centers)
                eu += p @ K.T
                tree_gate[t] += np.mean(
                    [m.predict_proba(Xg)[:, 1]
                     for m in gate_models[t][k]], axis=0)
            tree_eu[t] = eu / N_FOLDS
            tree_gate[t] /= N_FOLDS

        sp = np.mean([np.mean([m.predict_proba(feat_map["be"]) for m in state_models[k]],
                              axis=0) for k in range(N_FOLDS)], axis=0)
        states_gate = {"POR": sp[:, 0], "PERM": sp[:, 0] + sp[:, 3],
                       "SW": sp[:, 0] + sp[:, 2]}

        # ---- NN 推理 ----
        nn_eu = {c: {} for c in nn_models}
        nn_gate = {c: {} for c in nn_models}
        for cfg_name, ms in nn_models.items():
            meta = nn_meta[cfg_name]
            if NN_CONFIGS[cfg_name]["use_base"]:
                bstats = meta["bstats"]
                x = build_nn_input(df, {c: tuple(v) for c, v
                                        in meta["gstats"].items()},
                                   Xb, bstats, meta["base_cols"])
            else:
                x = build_nn_input(df, {c: tuple(v) for c, v
                                        in meta["gstats"].items()})
            gate_acc = {t: np.zeros(n) for t in TARGETS}
            prob_acc = {t: np.zeros((n, N_BINS[t])) for t in TARGETS}
            for m in ms:
                res = infer_well(m, x, device)
                for t in TARGETS:
                    gate_acc[t] += res[t][0]
                    prob_acc[t] += res[t][1].T
            for t in TARGETS:
                gate_acc[t] /= len(ms)
                prob_acc[t] /= len(ms)
                # 各折 support 不同 → 用 fold0 centers 近似（分布头
                # 的箱序一致性由同一 bins.json 保证，逐折差异很小）
                eu = np.zeros((n, len(actions[t])))
                for k in range(N_FOLDS):
                    centers = np.array(bins_by_fold[t][str(k)]["centers"])
                    p = prob_acc[t][:, :len(centers)].astype(np.float64)
                    p /= p.sum(axis=1, keepdims=True).clip(min=1e-9)
                    K = utility_matrix(t, actions[t], centers)
                    eu += p @ K.T
                nn_eu[cfg_name][t] = eu / N_FOLDS
                nn_gate[cfg_name][t] = gate_acc[t]

        # ---- 按 decision 融合 ----
        preds = {}
        for t in TARGETS:
            d = decision[t]
            src_eu = {"tree_dist": tree_eu[t]}
            src_gate = {"tree_gate": tree_gate[t], "states": states_gate[t]}
            for c in nn_models:
                src_eu[f"nn_{c}"] = nn_eu[c][t]
                src_gate[f"nn_{c}"] = nn_gate[c][t]
            eu = sum(src_eu[nm] * w for nm, w
                     in zip(d["cont_sources"], d["cont_weights"]) if w > 0)
            cont = actions[t][np.argmax(eu, axis=1)]
            g = sum(src_gate[nm] * w for nm, w
                    in zip(d["gate_sources"], d["gate_weights"]) if w > 0)
            preds[t] = np.where(g >= d["threshold"], SENTINEL[t], cont)

        depths = df["DEPTH"].to_numpy(np.float64)
        records = [{"depth": round(float(depths[i]), 4),
                    "POR": round(float(preds["POR"][i]), 4),
                    "PERM": round(float(preds["PERM"][i]), 4),
                    "SW": round(float(preds["SW"][i]), 4)}
                   for i in range(n)]
        result_data.append({
            "logId": os.path.splitext(os.path.basename(path))[0],
            "predictions": records})

    out = {"modelId": "", "modelName": "tree-seq-hybrid-v3",
           "version": "3.0", "resultData": result_data}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"已写出 {args.output}（{len(result_data)} 口井）")


if __name__ == "__main__":
    main()
