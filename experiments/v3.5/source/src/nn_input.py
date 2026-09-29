"""WellNet 的输入张量构造：井 DataFrame → (C, L) float32。

通道: 10 曲线 ×（全局 robust 归一 + 井内 robust 归一）= 20
      + 2 深度（井内相对、绝对缩放）+ 10 缺失掩码 = 32
      use_base 时再拼树 base 特征（全局 median/IQR 归一，clip ±8）。
缺失已在 data.interp_missing 中插值填充，掩码列为 {c}_MISS。
"""
import numpy as np
import pandas as pd

from .config import NN_CURVES


def _log_rt(df):
    """RT/RXO 用 log10 后的副本。"""
    d = df.copy()
    for c in ("RT", "RXO"):
        d[c] = np.log10(np.clip(d[c].to_numpy(np.float64), 0.01, None))
    return d


def global_stats(dfs: list) -> dict:
    """全训练集每曲线 median/IQR（log 变换后口径）。"""
    cat = pd.concat([_log_rt(d)[NN_CURVES] for d in dfs], ignore_index=True)
    stats = {}
    for c in NN_CURVES:
        v = cat[c]
        med = float(v.median())
        iqr = float(max(v.quantile(0.75) - v.quantile(0.25), 1e-3))
        stats[c] = (med, iqr)
    return stats


def base_stats(Xb: pd.DataFrame) -> dict:
    med = Xb.median()
    iqr = (Xb.quantile(0.75) - Xb.quantile(0.25)).clip(lower=1e-3)
    return {"median": med.to_dict(), "iqr": iqr.to_dict()}


def build_nn_input(df: pd.DataFrame, gstats: dict,
                   Xb: pd.DataFrame = None, bstats: dict = None,
                   base_cols: list = None) -> np.ndarray:
    d = _log_rt(df)
    L = len(d)
    chans = []
    for c in NN_CURVES:
        v = d[c].to_numpy(np.float64)
        med_g, iqr_g = gstats[c]
        chans.append((v - med_g) / iqr_g)
        med_w = np.median(v)
        iqr_w = max(np.percentile(v, 75) - np.percentile(v, 25), 1e-3)
        chans.append((v - med_w) / iqr_w)
    depth = d["DEPTH"].to_numpy(np.float64)
    chans.append(np.arange(L, dtype=np.float64) / max(L - 1, 1))
    chans.append((depth - 1800.0) / 600.0)
    for c in NN_CURVES:
        chans.append(d[f"{c}_MISS"].to_numpy(np.float64))
    x = np.stack(chans).astype(np.float32)
    x[:20] = np.clip(x[:20], -8.0, 8.0)

    if Xb is not None:
        med = np.array([bstats["median"][c] for c in base_cols])[:, None]
        iqr = np.array([bstats["iqr"][c] for c in base_cols])[:, None]
        xb = (Xb[base_cols].to_numpy(np.float64).T - med) / iqr
        xb = np.clip(np.nan_to_num(xb, nan=0.0), -8.0, 8.0)
        x = np.concatenate([x, xb.astype(np.float32)], axis=0)
    return x
