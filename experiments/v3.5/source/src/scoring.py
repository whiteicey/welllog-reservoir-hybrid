"""官方评分核、分位分箱、Bayes 效用解码、阈值精确扫描。

评分核:
    POR/SW: acc = max(0, 1 - |a-y| / (tol*(|y|+eps)))   tol=0.08/0.05
    PERM:   acc = max(0, 1 - |log10(a/y)|)
"""
import numpy as np

from .config import EPS, POR_TOL, SW_TOL, SENTINEL, BOUNDS, N_ACTIONS


def acc_vector(name, y, a):
    """逐点得分。y 真值数组，a 预测数组（同形状或标量）。"""
    y = np.asarray(y, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    if name == "PERM":
        ratio = np.clip(a, EPS, None) / np.clip(y, EPS, None)
        return np.clip(1.0 - np.abs(np.log10(np.clip(ratio, EPS, None))),
                       0.0, None)
    tol = POR_TOL if name == "POR" else SW_TOL
    return np.clip(1.0 - np.abs(a - y) / (tol * (np.abs(y) + EPS)), 0.0, None)


def total_score(accs: dict) -> float:
    return (0.30 * accs["POR"] + 0.35 * accs["PERM"] + 0.35 * accs["SW"]) * 100.0


# ---------- 目标变换 ----------

def transform(name, y):
    y = np.asarray(y, dtype=np.float64)
    if name == "POR":
        return np.log1p(np.clip(y, 0, None))
    if name == "PERM":
        return np.log10(np.clip(y, 1e-4, None))
    return y  # SW 恒等


def inverse(name, z):
    z = np.asarray(z, dtype=np.float64)
    if name == "POR":
        return np.expm1(z)
    if name == "PERM":
        return 10.0 ** z
    return z


# ---------- 分位分箱 ----------

def quantile_bins(name, y_real, n_bins):
    """在变换空间做等分位分箱。返回 (edges, centers_原值空间)。

    centers 取每箱内标签中位数（原值空间）。
    """
    z = transform(name, y_real)
    edges = np.quantile(z, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    k = len(edges) - 1
    idx = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, k - 1)
    centers = np.empty(k, dtype=np.float64)
    for j in range(k):
        m = idx == j
        centers[j] = np.median(y_real[m]) if m.any() else \
            inverse(name, 0.5 * (edges[j] + edges[j + 1]))
    return edges, centers


def bin_assign(name, y, edges):
    z = transform(name, y)
    k = len(edges) - 1
    return np.clip(np.searchsorted(edges, z, side="right") - 1, 0, k - 1)


# ---------- Bayes 效用解码 ----------

def action_grid(name, support):
    """800 点候选动作网格。SW 线性等距；POR/PERM 对数等距外扩 10%。"""
    s_min, s_max = float(np.min(support)), float(np.max(support))
    if name == "SW":
        return np.linspace(s_min, s_max, N_ACTIONS)
    lo = np.log(max(0.9 * s_min, 1e-4))
    hi = np.log(1.1 * s_max)
    return np.exp(np.linspace(lo, hi, N_ACTIONS))


def utility_matrix(name, actions, support):
    """K[i,j] = acc(action_i, support_j)。"""
    return acc_vector(name, support[None, :], actions[:, None])


def bayes_decode(name, probs, support, actions=None):
    """对每行概率分布解码期望得分最大的动作。

    probs: (n, k) 行归一化概率; support: (k,) 箱中心。
    返回 (n,) 连续预测。
    """
    if actions is None:
        actions = action_grid(name, support)
    K = utility_matrix(name, actions, support)      # (A, k)
    eu = probs @ K.T                                 # (n, A)
    lo, hi = BOUNDS[name]
    return np.clip(actions[np.argmax(eu, axis=1)], lo, hi)


def expected_utility_mix(name, prob_list, support_list, weights,
                         actions):
    """多源分布在效用层面加权混合后解码（各源 support 可不同）。

    返回 (n,) 连续预测。
    """
    eu = None
    for p, s, w in zip(prob_list, support_list, weights):
        if w <= 0:
            continue
        K = utility_matrix(name, actions, s)
        part = p @ K.T
        eu = part * w if eu is None else eu + part * w
    lo, hi = BOUNDS[name]
    return np.clip(actions[np.argmax(eu, axis=1)], lo, hi)


# ---------- 阈值精确扫描 ----------

def sweep_threshold(name, y_true, gate_p, cont_pred, th_grid):
    """对所有阈值一次性算精确分数。

    返回 (scores 数组与 th_grid 对齐, acc_cont, acc_default)。
    """
    acc_c = acc_vector(name, y_true, cont_pred)
    acc_d = acc_vector(name, y_true, np.full_like(y_true, SENTINEL[name]))
    order = np.argsort(-gate_p)             # gate 降序
    g_sorted = gate_p[order]
    delta = (acc_d - acc_c)[order]
    cum = np.concatenate([[0.0], np.cumsum(delta)])
    base = acc_c.sum()
    n = len(y_true)
    # 对每个阈值，默认组 = 前 m 个（g >= th）
    ms = np.searchsorted(-g_sorted, -th_grid, side="right")
    scores = (base + cum[ms]) / n
    return scores
