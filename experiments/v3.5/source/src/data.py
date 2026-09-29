"""数据读取、缺失处理、标签掩码、4 状态标签、固定折切分。"""
import os
import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import CURVES, MISSING, TARGETS, SENTINEL, N_FOLDS


# 需要插值/掩码处理的列（含 BIT，特征工程用到）
PROC_COLS = CURVES + ["BIT"]


def read_log_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", skiprows=[1])
    df.columns = [c.strip() for c in df.columns]
    df["DEPTH"] = df["DEPTH"].astype(np.float64)
    df = df.sort_values("DEPTH").reset_index(drop=True)
    df["well_id"] = os.path.splitext(os.path.basename(path))[0]
    for c in PROC_COLS:
        if c not in df.columns:
            df[c] = np.nan
        v = df[c].astype(np.float64)
        df[c] = v.where(v > MISSING / 2)
    df["DEPTH_MISS"] = 0.0
    return df


def interp_missing(df: pd.DataFrame, train_medians: dict) -> pd.DataFrame:
    """曲线按井线性插值（双向），仍缺填训练集中位数；保留缺失掩码列。"""
    for c in PROC_COLS:
        df[f"{c}_MISS"] = df[c].isna().astype(np.float32)
        df[c] = df[c].interpolate(limit_direction="both")
        df[c] = df[c].fillna(train_medians.get(c, 0.0))
    return df


def load_dir(dir_path: str) -> list:
    files = sorted(glob.glob(os.path.join(dir_path, "*.txt")))
    if not files:
        raise FileNotFoundError(f"目录中没有 txt 文件: {dir_path}")
    return [read_log_file(f) for f in files]


def curve_medians(dfs: list) -> dict:
    cat = pd.concat([d[PROC_COLS] for d in dfs], ignore_index=True)
    return {c: float(cat[c].median()) for c in PROC_COLS}


def label_arrays(df: pd.DataFrame):
    """返回 dict: 每目标 y/has/sen/real 掩码 + 4 状态标签。

    4 状态: 0=POR默认(全默认) 1=全非默认 2=SW默认(水层) 3=PERM默认(致密)
    state=-1 表示无标签。
    """
    out = {}
    n = len(df)
    for t in TARGETS:
        v = df[t].to_numpy(dtype=np.float64) if t in df.columns else \
            np.full(n, MISSING)
        has = v > MISSING / 2
        sen = has & np.isclose(v, SENTINEL[t], atol=1e-6)
        out[t] = {"y": v, "has": has, "sen": sen}

    por_sen = out["POR"]["sen"]
    perm_sen = out["PERM"]["sen"]
    sw_sen = out["SW"]["sen"]
    has_all = out["POR"]["has"] & out["PERM"]["has"] & out["SW"]["has"]
    state = np.full(n, -1, dtype=np.int64)
    state[has_all] = 1
    state[has_all & perm_sen] = 3
    state[has_all & sw_sen] = 2
    state[has_all & por_sen] = 0
    out["state"] = state
    return out


def make_folds(well_ids: np.ndarray):
    """按井 GroupKFold 5 折，返回每行的折号（与此前方案同口径）。"""
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold = np.full(len(well_ids), -1, dtype=np.int64)
    dummy = np.zeros(len(well_ids))
    for k, (_, va) in enumerate(gkf.split(dummy, groups=well_ids)):
        fold[va] = k
    return fold
