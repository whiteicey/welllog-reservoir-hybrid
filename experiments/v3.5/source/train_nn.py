"""WellNet 训练：gate_b（33 通道 3 种子）与 base_c（+base 特征 2 种子）。

与 train_trees.py 共用同一折切分（trees_oof.npz 中的 fold），分箱边界
读 bins.json（与树分布模型同一套 support，融合时可直接混合效用）。

输出 out_dir/:
    nn_{cfg}_fold{k}_seed{s}.pt
    nn_{cfg}_oof.npz     每目标 gate/prob OOF（种子平均）
    nn_{cfg}_meta.json

用法:
    python train_nn.py --data_dir ./data/train --out_dir ./outputs --cfg gate_b
    python train_nn.py --data_dir ./data/train --out_dir ./outputs --cfg base_c
"""
import argparse
import json
import os
import sys
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (TARGETS, N_FOLDS, N_BINS, SEED, NN_CONFIGS,
                        NN_CROP, NN_BATCH, NN_EPOCHS, NN_STEPS, NN_LR,
                        NN_WD, NN_GRAD_CLIP, NN_GAIN_STD, NN_OFFSET_STD,
                        NN_BASE_TOPK, SCORE_W)
from src.data import load_dir, curve_medians, interp_missing, label_arrays
from src.nn_input import global_stats, base_stats, build_nn_input
from src.wellnet import WellNet
from src.scoring import bin_assign, bayes_decode, sweep_threshold, total_score


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_well_tensors(dfs, gstats, use_base, Xb_list, bstats, base_cols):
    xs = []
    for i, d in enumerate(dfs):
        xb = Xb_list[i] if use_base else None
        xs.append(build_nn_input(d, gstats, xb, bstats, base_cols))
    return xs


def gate_point_weights(gate: np.ndarray) -> np.ndarray:
    """边界感知权重：层界 ±EDGE_RADIUS 点 ×EDGE_WEIGHT，
    薄层（段长<THIN_LEN）内点 ×THIN_WEIGHT，可叠乘。gate=-1 处为 0。"""
    from src.config import EDGE_RADIUS, EDGE_WEIGHT, THIN_LEN, THIN_WEIGHT
    n = len(gate)
    w = np.ones(n, dtype=np.float32)
    valid = gate >= 0
    w[~valid] = 0.0
    g = gate.copy()
    g[~valid] = -9  # 隔断无标签区，避免跨缺口误判边界
    # 边界: 相邻两点门控值不同（且都有标签）
    chg = np.where((g[1:] != g[:-1]) & (g[1:] >= 0) & (g[:-1] >= 0))[0]
    edge_mask = np.zeros(n, dtype=bool)
    for c in chg:
        lo = max(0, c - EDGE_RADIUS + 1)
        hi = min(n, c + 1 + EDGE_RADIUS)
        edge_mask[lo:hi] = True
    w[edge_mask & valid] *= EDGE_WEIGHT
    # 薄层: 连续同值段长 < THIN_LEN
    seg_start = 0
    for i in range(1, n + 1):
        if i == n or g[i] != g[seg_start]:
            if g[seg_start] >= 0 and (i - seg_start) < THIN_LEN:
                w[seg_start:i] *= THIN_WEIGHT
            seg_start = i
    return w


def build_well_labels(dfs, bins_by_fold, fold_of_well, boundary_aware=False):
    """每井: 每目标 (gate 0/1/-1, bin id 或 -1, gate 点权重)。分箱用该井
    所属折的 edges（该井是该折的验证井，edges 来自其训练折 → 无泄漏）。"""
    labels = []
    for i, d in enumerate(dfs):
        la = label_arrays(d)
        k = str(fold_of_well[i])
        per = {}
        for t in TARGETS:
            y, has, sen = la[t]["y"], la[t]["has"], la[t]["sen"]
            gate = np.where(has, sen.astype(np.int64), -1)
            edges = np.array(bins_by_fold[t][k]["edges"])
            b = bin_assign(t, y, edges)
            b = np.where(has & ~sen, b, -1)
            if boundary_aware:
                gw = gate_point_weights(gate)
            else:
                gw = (gate >= 0).astype(np.float32)
            per[t] = (gate.astype(np.int64), b.astype(np.int64), gw)
        labels.append(per)
    return labels


def sample_crop(x, lab, crop):
    L = x.shape[1]
    if L <= crop:
        pad = crop - L
        xx = np.pad(x, ((0, 0), (0, pad)))
        out = {t: (np.pad(lab[t][0], (0, pad), constant_values=-1),
                   np.pad(lab[t][1], (0, pad), constant_values=-1),
                   np.pad(lab[t][2], (0, pad)))
               for t in TARGETS}
        return xx, out
    s = np.random.randint(0, L - crop + 1)
    sl = slice(s, s + crop)
    return x[:, sl], {t: (lab[t][0][sl], lab[t][1][sl], lab[t][2][sl])
                      for t in TARGETS}


def augment(x, n_curve_ch=20):
    g = 1.0 + np.random.randn(n_curve_ch, 1).astype(np.float32) * NN_GAIN_STD
    o = np.random.randn(n_curve_ch, 1).astype(np.float32) * NN_OFFSET_STD
    x = x.copy()
    x[:n_curve_ch] = x[:n_curve_ch] * g + o
    return x


@torch.no_grad()
def infer_well(model, x, device):
    model.eval()
    xt = torch.from_numpy(x).unsqueeze(0).to(device)
    with torch.autocast(device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=device.type == "cuda"):
        out = model(xt)
    res = {}
    for t in TARGETS:
        gate = torch.sigmoid(out[t][0].float())[0].cpu().numpy()
        prob = torch.softmax(out[t][1].float(), dim=1)[0].cpu().numpy()
        res[t] = (gate, prob)   # (L,), (K, L)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="./outputs")
    ap.add_argument("--cfg", choices=list(NN_CONFIGS), required=True)
    ap.add_argument("--epochs", type=int, default=NN_EPOCHS)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    args = ap.parse_args()
    cfg = NN_CONFIGS[args.cfg]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"cfg={args.cfg} device={device}")

    print("[1/3] 数据准备 ...")
    trees = np.load(os.path.join(args.out_dir, "trees_oof.npz"),
                    allow_pickle=True)
    with open(os.path.join(args.out_dir, "bins.json"), encoding="utf-8") as f:
        bins_by_fold = json.load(f)

    dfs = load_dir(args.data_dir)
    meds = curve_medians(dfs)
    dfs = [interp_missing(d, meds) for d in dfs]
    well_ids = [d["well_id"].iloc[0] for d in dfs]

    # 从树输出恢复每井折号（同口径保证）
    tw, tf = trees["well"], trees["fold"]
    fold_of_well = {}
    for i, wid in enumerate(well_ids):
        fold_of_well[i] = int(tf[tw == wid][0])

    gstats = global_stats(dfs)
    Xb_list, bstats, base_cols = None, None, None
    if cfg["use_base"]:
        from src.features import base_enhanced_features
        Xb_list = [base_enhanced_features(d)[0] for d in dfs]
        Xb_cat = pd.concat(Xb_list, ignore_index=True)
        var = Xb_cat.var().sort_values(ascending=False)
        base_cols = list(var.index[:NN_BASE_TOPK])
        bstats = base_stats(Xb_cat[base_cols])

    xs = build_well_tensors(dfs, gstats, cfg["use_base"],
                            Xb_list, bstats, base_cols)
    labels = build_well_labels(dfs, bins_by_fold, fold_of_well,
                               boundary_aware=cfg.get("boundary_aware", False))
    n_in = xs[0].shape[0]
    lens = np.array([x.shape[1] for x in xs], dtype=np.float64)
    print(f"    {len(xs)} 井, 输入 {n_in} 通道")

    run_folds = args.folds if args.folds else list(range(N_FOLDS))
    # OOF 容器（按井拼接）
    oof_gate = {t: {} for t in TARGETS}
    oof_prob = {t: {} for t in TARGETS}

    for k in run_folds:
        tr_ids = [i for i in range(len(xs)) if fold_of_well[i] != k]
        va_ids = [i for i in range(len(xs)) if fold_of_well[i] == k]
        p_sample = lens[tr_ids] / lens[tr_ids].sum()
        print(f"[2/3] fold {k}: {len(tr_ids)} train / {len(va_ids)} val 井")

        seed_outputs = []
        for s in cfg["seeds"]:
            set_seed(SEED * 100 + s)
            model = WellNet(n_in).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=NN_LR,
                                    weight_decay=NN_WD)
            total_steps = args.epochs * NN_STEPS
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=NN_LR, total_steps=total_steps, pct_start=0.1)
            ce = nn.CrossEntropyLoss(ignore_index=-1)
            step = 0
            for ep in range(args.epochs):
                model.train()
                ep_loss = 0.0
                for _ in range(NN_STEPS):
                    bx, blab = [], []
                    for _ in range(NN_BATCH):
                        wi = tr_ids[np.random.choice(len(tr_ids), p=p_sample)]
                        xx, ll = sample_crop(xs[wi], labels[wi], NN_CROP)
                        bx.append(augment(xx))
                        blab.append(ll)
                    xb = torch.from_numpy(np.stack(bx)).to(device)
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type,
                                        dtype=torch.bfloat16,
                                        enabled=device.type == "cuda"):
                        out = model(xb)
                        loss = 0.0
                        for ti, t in enumerate(TARGETS):
                            g_log, b_log = out[t]
                            g_lab = torch.from_numpy(np.stack(
                                [b[t][0] for b in blab])).to(device)
                            b_lab = torch.from_numpy(np.stack(
                                [b[t][1] for b in blab])).to(device)
                            g_w = torch.from_numpy(np.stack(
                                [b[t][2] for b in blab])).to(device)
                            if (g_w > 0).any():
                                bce = nn.functional.\
                                    binary_cross_entropy_with_logits(
                                        g_log, (g_lab >= 1).float(),
                                        reduction="none")
                                loss = loss + SCORE_W[t] * (
                                    (bce * g_w).sum() / g_w.sum())
                            loss = loss + SCORE_W[t] * cfg["bin_weight"] * \
                                ce(b_log, b_lab)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), NN_GRAD_CLIP)
                    opt.step()
                    sched.step()
                    step += 1
                    ep_loss += float(loss)
                if (ep + 1) % 10 == 0:
                    print(f"    seed{s} ep{ep+1} loss={ep_loss/NN_STEPS:.4f}")
            torch.save(model.state_dict(), os.path.join(
                args.out_dir, f"nn_{args.cfg}_fold{k}_seed{s}.pt"))
            # 验证井推理
            outs = {}
            for wi in va_ids:
                outs[wi] = infer_well(model, xs[wi], device)
            seed_outputs.append(outs)

        # 种子平均，写入 OOF
        for wi in va_ids:
            for t in TARGETS:
                gate = np.mean([so[wi][t][0] for so in seed_outputs], axis=0)
                prob = np.mean([so[wi][t][1] for so in seed_outputs], axis=0)
                oof_gate[t][wi] = gate
                oof_prob[t][wi] = prob.T          # (L, K)

    print("[3/3] 保存 OOF 并快评 ...")
    # 与 trees_oof 的行序对齐（按 dfs 顺序拼接 = trees 的行序）
    dump = {}
    accs = {}
    th_grid = np.linspace(0.02, 0.90, 89)
    for t in TARGETS:
        gs, ps = [], []
        for i in range(len(xs)):
            L = xs[i].shape[1]
            if i in oof_gate[t]:
                gs.append(oof_gate[t][i])
                ps.append(oof_prob[t][i])
            else:  # 未训练的折填 NaN
                gs.append(np.full(L, np.nan, dtype=np.float32))
                ps.append(np.full((L, N_BINS[t]), np.nan, dtype=np.float32))
        dump[f"{t}_gate"] = np.concatenate(gs)
        dump[f"{t}_prob"] = np.concatenate(ps)

        # 快评（仅已训练折）
        la_y = trees[f"{t}_y"]
        la_has = trees[f"{t}_has"]
        valid = la_has & ~np.isnan(dump[f"{t}_gate"])
        if valid.any():
            idx = np.where(valid)[0]
            y_true = la_y[idx]
            cont = np.zeros(len(idx))
            for k in range(N_FOLDS):
                m = tf[idx] == k
                if not m.any():
                    continue
                centers = np.array(bins_by_fold[t][str(k)]["centers"])
                pr = dump[f"{t}_prob"][idx[m], :len(centers)].astype(np.float64)
                pr /= pr.sum(axis=1, keepdims=True).clip(min=1e-9)
                cont[m] = bayes_decode(t, pr, centers)
            scores = sweep_threshold(t, y_true, dump[f"{t}_gate"][idx],
                                     cont, th_grid)
            accs[t] = float(scores.max())
            print(f"    {t}: NN 单源 acc={accs[t]:.4f}")
    if len(accs) == 3:
        print(f"NN({args.cfg}) 单源 OOF Total_Score = {total_score(accs):.2f}")

    np.savez_compressed(
        os.path.join(args.out_dir, f"nn_{args.cfg}_oof.npz"), **dump)
    meta = {"cfg": args.cfg, "n_in": n_in, "base_cols": base_cols,
            "gstats": {c: list(v) for c, v in gstats.items()},
            "bstats": bstats, "oof_acc_single": accs}
    with open(os.path.join(args.out_dir, f"nn_{args.cfg}_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"已保存至 {args.out_dir}")


if __name__ == "__main__":
    main()
