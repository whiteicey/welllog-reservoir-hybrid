"""低自由度嵌套融合：分布效用混合 + 门控概率混合 + 单阈值。

来源:
    连续: 树分布模型 + nn_base_c 分箱头 + nn_gate_b 分箱头
    门控: 树 gate + 4 状态边缘化 + nn_gate_b + nn_base_c
嵌套 CV: 外层 5 折——内层 4 折选 (cw, gw, th)，应用到外折；
        全局配置在全部 OOF 上选，写入 decision.json 供推理用。

实现要点: 期望效用矩阵 (n×800×源数) 太大，按行分块流式计算——
对每个连续权重组合只保存解码后的连续预测 (n,)，
之后的门控×阈值搜索全部基于这些向量，内存 O(组合数×n)。

用法:
    python assemble.py --out_dir ./outputs
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import TARGETS, N_FOLDS, SENTINEL, BLEND_STEP
from src.scoring import (acc_vector, total_score, action_grid,
                         utility_matrix)

CHUNK = 40000


def simplex_grid(n_src, step=BLEND_STEP):
    ticks = int(round(1.0 / step))
    out = []
    for combo in itertools.product(range(ticks + 1), repeat=n_src):
        if sum(combo) == ticks:
            out.append(tuple(c / ticks for c in combo))
    return out


def state_gate(state_prob):
    return {
        "POR": state_prob[:, 0],
        "PERM": state_prob[:, 0] + state_prob[:, 3],
        "SW": state_prob[:, 0] + state_prob[:, 2],
    }


def decode_all_mixes(name, prob_sources, bins_by_fold, fold_idx, idx,
                     actions, cw_grid):
    """分块计算每个连续权重组合的解码预测。

    返回 dict: cw -> (n_idx,) 连续预测。
    """
    n = len(idx)
    conts = {cw: np.empty(n) for cw in cw_grid}
    # 每折的核矩阵 K[k][src] = (A, k_bins)
    Ks = {}
    for k in range(N_FOLDS):
        centers = np.array(bins_by_fold[name][str(k)]["centers"])
        Ks[k] = utility_matrix(name, actions, centers)

    for start in range(0, n, CHUNK):
        sl = slice(start, min(start + CHUNK, n))
        rows = idx[sl]
        f_rows = fold_idx[sl]
        # 每个源在该块的 EU
        eus = []
        for probs_full in prob_sources:
            eu = np.zeros((len(rows), len(actions)), dtype=np.float32)
            for k in range(N_FOLDS):
                m = f_rows == k
                if not m.any():
                    continue
                K = Ks[k]
                p = probs_full[rows[m], :K.shape[1]].astype(np.float64)
                p /= p.sum(axis=1, keepdims=True).clip(min=1e-9)
                eu[m] = (p @ K.T).astype(np.float32)
            eus.append(eu)
        for cw in cw_grid:
            mix = None
            for e, w in zip(eus, cw):
                if w <= 0:
                    continue
                mix = e * w if mix is None else mix + e * w
            if mix is None:
                conts[cw][sl] = np.nan
            else:
                conts[cw][sl] = actions[np.argmax(mix, axis=1)]
    return conts


def best_gate_th(y, acc_cont, acc_default, gates_rows, gw_grid, th_grid):
    """给定连续预测的 acc 向量，搜门控权重×阈值。"""
    base = acc_cont.sum()
    n = len(y)
    best = None
    for gw in gw_grid:
        g = None
        for gv, w in zip(gates_rows, gw):
            if w <= 0:
                continue
            g = gv * w if g is None else g + gv * w
        if g is None:
            continue
        order = np.argsort(-g)
        delta = (acc_default - acc_cont)[order]
        cum = np.concatenate([[0.0], np.cumsum(delta)])
        ms = np.searchsorted(-g[order], -th_grid, side="right")
        scores = (base + cum[ms]) / n
        bi = int(np.argmax(scores))
        if best is None or scores[bi] > best[0]:
            best = (float(scores[bi]), gw, float(th_grid[bi]))
    return best


def choose_config(name, y_true, conts, gates, rows, gw_grid, th_grid):
    y = y_true[rows]
    acc_default = acc_vector(name, y, np.full_like(y, SENTINEL[name]))
    gates_rows = [g[rows] for g in gates]
    best = None
    for cw, cont in conts.items():
        if np.isnan(cont[rows][0]):
            continue
        acc_cont = acc_vector(name, y, cont[rows])
        r = best_gate_th(y, acc_cont, acc_default, gates_rows,
                         gw_grid, th_grid)
        if r is None:
            continue
        score, gw, th = r
        if best is None or score > best[0]:
            best = (score, cw, gw, th)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./outputs")
    args = ap.parse_args()

    trees = np.load(os.path.join(args.out_dir, "trees_oof.npz"),
                    allow_pickle=True)
    with open(os.path.join(args.out_dir, "bins.json"), encoding="utf-8") as f:
        bins_by_fold = json.load(f)
    nn = {}
    for cfg in ("gate_b", "base_c", "edge_b"):
        path = os.path.join(args.out_dir, f"nn_{cfg}_oof.npz")
        if os.path.exists(path):
            nn[cfg] = np.load(path, allow_pickle=True)
        else:
            print(f"警告: 缺少 {path}，跳过该 NN 源")

    fold = trees["fold"]
    th_grid = np.linspace(0.02, 0.90, 89)
    sgate = state_gate(trees["state_prob"].astype(np.float64))

    decision, nested_accs = {}, {}
    for t in TARGETS:
        y_full = trees[f"{t}_y"].astype(np.float64)
        has = trees[f"{t}_has"].astype(bool)
        gate_tree = trees[f"{t}_gate"].astype(np.float64)
        valid = has & ~np.isnan(gate_tree)
        idx = np.where(valid)[0]
        y_true = y_full[idx]
        fold_idx = fold[idx]

        centers0 = np.array(bins_by_fold[t]["0"]["centers"])
        actions = action_grid(t, centers0)
        cont_names = ["tree_dist"]
        prob_sources = [trees[f"{t}_prob"]]
        for cfg in ("base_c", "gate_b", "edge_b"):
            if cfg in nn:
                cont_names.append(f"nn_{cfg}")
                prob_sources.append(nn[cfg][f"{t}_prob"])
        cw_grid = simplex_grid(len(prob_sources))
        print(f"[{t}] 解码 {len(cw_grid)} 个连续混合 ...")
        conts = decode_all_mixes(t, prob_sources, bins_by_fold, fold_idx,
                                 idx, actions, cw_grid)

        gate_names = ["tree_gate", "states"]
        gates = [gate_tree[idx], sgate[t][idx]]
        for cfg in ("gate_b", "base_c", "edge_b"):
            if cfg in nn:
                gate_names.append(f"nn_{cfg}")
                gates.append(nn[cfg][f"{t}_gate"].astype(np.float64)[idx])
        gw_grid = simplex_grid(len(gates))

        nested_pred = np.zeros(len(idx))
        for ko in range(N_FOLDS):
            inner = np.where(fold_idx != ko)[0]
            outer = np.where(fold_idx == ko)[0]
            _, cw, gw, th = choose_config(t, y_true, conts, gates,
                                          inner, gw_grid, th_grid)
            g = sum(gv[outer] * w for gv, w in zip(gates, gw) if w > 0)
            nested_pred[outer] = np.where(
                g >= th, SENTINEL[t], conts[cw][outer])
        nested_accs[t] = float(np.mean(acc_vector(t, y_true, nested_pred)))

        all_rows = np.arange(len(idx))
        gscore, cw, gw, th = choose_config(t, y_true, conts, gates,
                                           all_rows, gw_grid, th_grid)
        decision[t] = {
            "cont_sources": cont_names, "cont_weights": list(cw),
            "gate_sources": gate_names, "gate_weights": list(gw),
            "threshold": th, "global_acc": gscore,
        }
        print(f"[{t}] nested acc={nested_accs[t]:.4f}  global acc={gscore:.4f}")
        print(f"      cont={dict(zip(cont_names, cw))}")
        print(f"      gate={dict(zip(gate_names, gw))}  th={th:.2f}")

    score = total_score(nested_accs)
    print(f"嵌套 OOF Total_Score = {score:.2f}")
    with open(os.path.join(args.out_dir, "decision.json"), "w",
              encoding="utf-8") as f:
        json.dump({"decision": decision, "nested_acc": nested_accs,
                   "nested_score": score}, f, ensure_ascii=False, indent=2)
    print("已写出 decision.json")


if __name__ == "__main__":
    main()
