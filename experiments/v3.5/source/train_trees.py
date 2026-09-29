"""树模型训练：分布模型（POR/PERM/SW 分箱 MultiClass）+ 三个门控二分类
+ 4 状态联合模型。CatBoost 双种子 × 按井 5 折。

输出 out_dir/:
    dist_{TGT}_fold{k}_seed{s}.cbm    分布模型
    gate_{TGT}_fold{k}_seed{s}.cbm    门控
    states_fold{k}_seed{s}.cbm        4 状态
    trees_oof.npz                     OOF: 每目标 prob/gate/state 后验 + 元数据
    bins.json                         每折分箱 edges/centers
    trees_meta.json

用法:
    python train_trees.py --data_dir ./data/train --out_dir ./outputs [--gpu]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (TARGETS, N_FOLDS, N_BINS, DIST_PARAMS, DIST_SEEDS,
                        GATE_ITERS, GATE_PARAMS, GATE_SEEDS,
                        STATES_PARAMS, STATES_SEEDS)
from src.data import (load_dir, curve_medians, interp_missing, label_arrays,
                      make_folds)
from src.features import build_all_features
from src.scoring import (quantile_bins, bin_assign, bayes_decode,
                         acc_vector, total_score, sweep_threshold)

# 每目标特征集（对齐参考方案最终配方）:
#   分布: POR=base+extra  PERM=base  SW=enhanced
#   门控: POR=base  PERM=base  SW=enhanced；4状态=base+extra
DIST_FEATS = {"POR": "be", "PERM": "b", "SW": "e"}
GATE_FEATS = {"POR": "b", "PERM": "b", "SW": "e"}


def feat_matrix(kind, Xb, Xe, Xx):
    if kind == "b":
        return Xb
    if kind == "e":
        return Xe
    return pd.concat([Xb, Xx], axis=1)   # "be" = base + extra


def fit_cls(params, seeds, X_tr, y_tr, gpu, save_paths, devices="0"):
    """多种子训练，返回按种子平均的 predict_proba 函数。"""
    models = []
    for s, path in zip(seeds, save_paths):
        p = dict(params, random_seed=s)
        if gpu:
            p["task_type"] = "GPU"
            p["devices"] = devices  # 默认只用一张卡，避免独占整机
        else:
            p.pop("bootstrap_type", None)  # Bayesian bootstrap CPU 慢，保留默认
        m = CatBoostClassifier(**p)
        m.fit(X_tr, y_tr)
        m.save_model(path)
        models.append(m)
    return models


def predict_proba_avg(models, X):
    return np.mean([m.predict_proba(X) for m in models], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="./outputs")
    ap.add_argument("--gpu", action="store_true", help="CatBoost 用 GPU 训练")
    ap.add_argument("--devices", default="0",
                    help="CatBoost GPU 卡号，如 '0' 或 '1'（默认只用 0 号卡）")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("[1/4] 读取数据与特征 ...")
    dfs = load_dir(args.data_dir)
    meds = curve_medians(dfs)
    dfs = [interp_missing(d, meds) for d in dfs]
    df = pd.concat(dfs, ignore_index=True)
    labels = label_arrays(df)
    Xb, Xe, Xx = build_all_features(dfs)
    well = df["well_id"].to_numpy()
    fold = make_folds(well)
    print(f"    {len(dfs)} 井 / {len(df)} 行 / base {Xb.shape[1]} 维 "
          f"/ enhanced {Xe.shape[1]} 维 / extra {Xx.shape[1]} 维")

    n = len(df)
    oof = {}
    bins_all = {t: {} for t in TARGETS}

    print("[2/4] 训练分布模型与门控 ...")
    for t in TARGETS:
        y = labels[t]["y"]
        has = labels[t]["has"]
        sen = labels[t]["sen"]
        real = has & ~sen
        Xd = feat_matrix(DIST_FEATS[t], Xb, Xe, Xx)
        Xg = feat_matrix(GATE_FEATS[t], Xb, Xe, Xx)

        oof[f"{t}_prob"] = np.zeros((n, N_BINS[t]), dtype=np.float32)
        oof[f"{t}_gate"] = np.full(n, np.nan, dtype=np.float32)

        for k in range(N_FOLDS):
            tr = (fold != k)
            va = (fold == k)
            # ---- 分布模型（仅真实点） ----
            tr_real = tr & real
            y_tr = y[tr_real]
            edges, centers = quantile_bins(t, y_tr, N_BINS[t])
            bins_all[t][str(k)] = {"edges": edges.tolist(),
                                   "centers": centers.tolist()}
            cls_y = bin_assign(t, y_tr, edges)
            paths = [os.path.join(args.out_dir,
                                  f"dist_{t}_fold{k}_seed{s}.cbm")
                     for s in DIST_SEEDS]
            models = fit_cls(DIST_PARAMS, DIST_SEEDS,
                             Xd[tr_real], cls_y, args.gpu, paths,
                             args.devices)
            p = predict_proba_avg(models, Xd[va])
            kbins = len(centers)
            oof[f"{t}_prob"][np.where(va)[0], :kbins] = p

            # ---- 门控（有标签点） ----
            tr_has = tr & has
            gp = dict(GATE_PARAMS, iterations=GATE_ITERS[t])
            paths = [os.path.join(args.out_dir,
                                  f"gate_{t}_fold{k}_seed{s}.cbm")
                     for s in GATE_SEEDS]
            models = fit_cls(gp, GATE_SEEDS,
                             Xg[tr_has], sen[tr_has].astype(np.int64),
                             args.gpu, paths, args.devices)
            oof[f"{t}_gate"][va] = predict_proba_avg(models, Xg[va])[:, 1]
            print(f"    {t} fold{k} 完成")

    print("[3/4] 训练 4 状态联合模型 ...")
    state = labels["state"]
    Xs = feat_matrix("be", Xb, Xe, Xx)
    oof["state_prob"] = np.full((n, 4), np.nan, dtype=np.float32)
    for k in range(N_FOLDS):
        tr = (fold != k) & (state >= 0)
        va = (fold == k)
        paths = [os.path.join(args.out_dir, f"states_fold{k}_seed{s}.cbm")
                 for s in STATES_SEEDS]
        models = fit_cls(STATES_PARAMS, STATES_SEEDS,
                         Xs[tr], state[tr], args.gpu, paths, args.devices)
        oof["state_prob"][va] = predict_proba_avg(models, Xs[va])
        print(f"    states fold{k} 完成")

    print("[4/4] OOF 快评（单源：树分布 + 树门控） ...")
    th_grid = np.linspace(0.02, 0.90, 89)
    accs = {}
    for t in TARGETS:
        has = labels[t]["has"]
        y_true = labels[t]["y"][has]
        idx = np.where(has)[0]
        # 逐折 Bayes 解码（各折 centers 不同）
        cont = np.zeros(len(idx))
        for k in range(N_FOLDS):
            m = fold[idx] == k
            if not m.any():
                continue
            centers = np.array(bins_all[t][str(k)]["centers"])
            probs = oof[f"{t}_prob"][idx[m], :len(centers)].astype(np.float64)
            probs /= probs.sum(axis=1, keepdims=True).clip(min=1e-9)
            cont[m] = bayes_decode(t, probs, centers)
        gate = oof[f"{t}_gate"][idx]
        scores = sweep_threshold(t, y_true, gate, cont, th_grid)
        best = int(np.argmax(scores))
        accs[t] = float(scores[best])
        print(f"    {t}: acc={accs[t]:.4f} th={th_grid[best]:.2f}")
    print(f"树单源 OOF Total_Score = {total_score(accs):.2f}")

    np.savez_compressed(
        os.path.join(args.out_dir, "trees_oof.npz"),
        well=well, fold=fold, depth=df["DEPTH"].to_numpy(np.float64),
        **oof,
        **{f"{t}_y": labels[t]["y"] for t in TARGETS},
        **{f"{t}_has": labels[t]["has"] for t in TARGETS},
        state=state)
    with open(os.path.join(args.out_dir, "bins.json"), "w",
              encoding="utf-8") as f:
        json.dump(bins_all, f)
    meta = {"feat_base": list(Xb.columns), "feat_enh": list(Xe.columns),
            "feat_extra": list(Xx.columns),
            "curve_medians": meds, "dist_feats": DIST_FEATS,
            "gate_feats": GATE_FEATS, "oof_acc_single": accs,
            "oof_score_single": total_score(accs)}
    with open(os.path.join(args.out_dir, "trees_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"已保存至 {args.out_dir}")


if __name__ == "__main__":
    main()
