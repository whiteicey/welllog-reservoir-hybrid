"""特征工程：逐列复刻参考方案的 base(113) / enhanced(237) / extra(181)。

前置: 曲线已按井插值填充（data.interp_missing），{c}_MISS 掩码列存在。
构成:
    base 113: 13 原始列 + log1p/三角/比值/乘积/差值 + 13 缺失掩码
              + 6 关键曲线井级 MEDIAN/STD/RANK + 滚动 9/51/201 均值/std
              + DIFF + TREND(51-201)
    enhanced 237: base + 11 上下文曲线 ZIQR/QNORM(+非关键曲线 RANK)
              + 9 局部曲线 RESID_MEAN/RESID_MEDIAN/RANGE/SLOPE + 7 物理量
    extra 181: 8 曲线 ±1/2/3 滞后差 + MED3/5 及残差 + D1/D1_NEXT/CURV
              + DN_SEP 及滞后差 + 7 曲线 601/1501 长窗 + 井顶底距离
"""
import numpy as np
import pandas as pd

BASE_COLUMNS = ["DEPTH", "GR", "PE", "SP", "CAL", "AC", "DEN", "CNL",
                "RXO", "RT", "DEVI", "AZIM", "BIT"]
KEY_CURVES = ["GR", "AC", "DEN", "CNL", "RT", "RXO"]
CONTEXT_CURVES = ["GR", "PE", "SP", "CAL", "AC", "DEN", "CNL", "RXO",
                  "RT", "DEVI", "BIT"]
LOCAL_CURVES = ["GR", "PE", "SP", "CAL", "AC", "DEN", "CNL", "RXO", "RT"]
LAG_CURVES = ["GR", "AC", "DEN", "CNL", "RT", "RXO", "PE", "SP"]
LONG_CURVES = ["GR", "SP", "DEN", "AC", "CNL", "RT", "PE"]
ROLLING_WINDOWS = (9, 51, 201)


def base_enhanced_features(df: pd.DataFrame):
    """返回 (base 113 列, enhanced 237 列)。df 曲线已插值。"""
    cols = {c: df[c].astype(np.float64) for c in BASE_COLUMNS}

    cols["GR_LOG"] = np.log1p(cols["GR"].clip(lower=0))
    cols["RXO_LOG"] = np.log1p(cols["RXO"].clip(lower=0))
    cols["RT_LOG"] = np.log1p(cols["RT"].clip(lower=0))
    cols["AZIM_SIN"] = np.sin(np.deg2rad(cols["AZIM"]))
    cols["AZIM_COS"] = np.cos(np.deg2rad(cols["AZIM"]))
    cols["DEVI_SQ"] = cols["DEVI"] ** 2

    d_min, d_max = float(cols["DEPTH"].min()), float(cols["DEPTH"].max())
    cols["DEPTH_REL"] = (cols["DEPTH"] - d_min) / max(d_max - d_min, 1.0)

    cols["CAL_MINUS_BIT"] = cols["CAL"] - cols["BIT"]
    cols["CAL_BIT_RATIO"] = cols["CAL"] / cols["BIT"].clip(lower=0.1)
    cols["RT_RXO_RATIO"] = cols["RT"] / cols["RXO"].clip(lower=0.01)
    cols["RT_RXO_LOG_RATIO"] = cols["RT_LOG"] - cols["RXO_LOG"]
    cols["AC_DEN_RATIO"] = cols["AC"] / cols["DEN"].clip(lower=0.5)
    cols["CNL_DEN_RATIO"] = cols["CNL"] / cols["DEN"].clip(lower=0.5)
    cols["GR_CNL_PRODUCT"] = cols["GR"] * cols["CNL"]
    cols["AC_CNL_PRODUCT"] = cols["AC"] * cols["CNL"]
    cols["DEN_CNL_PRODUCT"] = cols["DEN"] * cols["CNL"]
    cols["GR_AC_PRODUCT"] = cols["GR"] * cols["AC"]
    cols["GR_DEN_PRODUCT"] = cols["GR"] * cols["DEN"]
    cols["GR_RTV_PRODUCT"] = cols["GR"] / cols["RT"].clip(lower=0.01)
    cols["SP_GR_DIFF"] = cols["SP"] - cols["GR"]
    cols["DEN_AC_DIFF"] = cols["DEN"] - cols["AC"] / 100.0

    for c in BASE_COLUMNS:
        cols[f"{c}_MISSING"] = df[f"{c}_MISS"].astype(np.float64)

    for c in KEY_CURVES:
        s = cols[c]
        med = float(s.median())
        std = max(float(s.std(ddof=0)), 1e-6)
        cols[f"{c}_WELL_MEDIAN"] = pd.Series(med, index=s.index)
        cols[f"{c}_WELL_STD"] = pd.Series(std, index=s.index)
        cols[f"{c}_WELL_RANK"] = s.rank(method="average", pct=True)

    for c in KEY_CURVES:
        s = cols[c]
        for w in ROLLING_WINDOWS:
            r = s.rolling(w, center=True, min_periods=1)
            cols[f"{c}_ROLL_MEAN_{w}"] = r.mean()
            cols[f"{c}_ROLL_STD_{w}"] = r.std(ddof=0).fillna(0.0)
        cols[f"{c}_DIFF"] = s.diff().fillna(0.0)
        cols[f"{c}_TREND"] = (
            s.rolling(51, center=True, min_periods=1).mean()
            - s.rolling(201, center=True, min_periods=1).mean()).fillna(0.0)

    base = pd.concat(cols, axis=1).replace([np.inf, -np.inf], np.nan)
    base = base.fillna(0.0)

    enh = {name: base[name] for name in base.columns}
    for c in CONTEXT_CURVES:
        s = base[c].astype(np.float64)
        med = float(s.median())
        q05, q25 = float(s.quantile(0.05)), float(s.quantile(0.25))
        q75, q95 = float(s.quantile(0.75)), float(s.quantile(0.95))
        iqr = max(q75 - q25, 1e-6)
        span = max(q95 - q05, 1e-6)
        enh[f"{c}_WELL_ZIQR"] = (s - med) / iqr
        enh[f"{c}_WELL_QNORM"] = (s - q05) / span
        if c not in KEY_CURVES:
            enh[f"{c}_WELL_RANK"] = s.rank(method="average", pct=True)

    for c in LOCAL_CURVES:
        s = base[c].astype(np.float64)
        for w in ROLLING_WINDOWS:
            r = s.rolling(w, center=True, min_periods=1)
            enh[f"{c}_RESID_MEAN_{w}"] = s - r.mean()
            enh[f"{c}_RESID_MEDIAN_{w}"] = s - r.median()
            if w < 201:
                enh[f"{c}_ROLL_RANGE_{w}"] = r.max() - r.min()
        for lag in (5, 20):
            enh[f"{c}_SLOPE_{lag}"] = s.diff(lag).fillna(0.0) / float(lag)

    den, ac = base["DEN"].astype(np.float64), base["AC"].astype(np.float64)
    cnl, gr = base["CNL"].astype(np.float64), base["GR"].astype(np.float64)
    rt, rxo = base["RT"].astype(np.float64), base["RXO"].astype(np.float64)
    dphi = ((2.65 - den) / 1.65 * 100.0).clip(-30.0, 100.0)
    sphi = ((ac - 182.0) / 438.0 * 100.0).clip(-30.0, 100.0)
    g05, g95 = float(gr.quantile(0.05)), float(gr.quantile(0.95))
    shale = ((gr - g05) / max(g95 - g05, 1e-6)).clip(0.0, 1.0)
    proxy = ((cnl.clip(lower=0.1) + dphi.clip(lower=0.1)) / 200.0)\
        .clip(0.001, 0.8)
    enh["DENSITY_POROSITY"] = dphi
    enh["SONIC_POROSITY"] = sphi
    enh["CNL_DENSITY_POROSITY_DIFF"] = cnl - dphi
    enh["CNL_DENSITY_POROSITY_MEAN"] = (cnl + dphi) / 2.0
    enh["SHALE_INDEX_GR"] = shale
    enh["ARCHIE_PROXY"] = (np.log10(rt.clip(lower=0.01))
                           + 2.0 * np.log10(proxy))
    enh["INVASION_LOG_DIFF"] = (np.log10(rt.clip(lower=0.01))
                                - np.log10(rxo.clip(lower=0.01)))

    enh_frame = pd.concat(enh, axis=1).replace([np.inf, -np.inf], np.nan)
    enh_frame = enh_frame.fillna(0.0)
    return base.astype(np.float32), enh_frame.astype(np.float32)


def extra_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = {}
    for c in LAG_CURVES:
        s = df[c].astype(np.float64)
        if c in ("RT", "RXO"):
            s = np.log10(s.clip(lower=0.01))
        for lag in (-3, -2, -1, 1, 2, 3):
            cols[f"{c}_LAG{lag:+d}"] = (s.shift(lag) - s).bfill().ffill()
        for w in (3, 5):
            med = s.rolling(w, center=True, min_periods=1).median()
            cols[f"{c}_MED{w}"] = med
            cols[f"{c}_MED{w}_RESID"] = s - med
        cols[f"{c}_D1"] = s.diff().fillna(0.0)
        cols[f"{c}_D1_NEXT"] = (-s.diff(-1)).fillna(0.0)
        cols[f"{c}_CURV"] = (s.shift(-1) - 2 * s + s.shift(1)).fillna(0.0)

    den = df["DEN"].astype(np.float64)
    cnl = df["CNL"].astype(np.float64)
    dphi = (2.65 - den) / 1.65 * 100.0
    sep = cnl - dphi
    cols["DN_SEP"] = sep
    for lag in (-2, -1, 1, 2):
        cols[f"DN_SEP_LAG{lag:+d}"] = (sep.shift(lag) - sep).bfill().ffill()

    for c in LONG_CURVES:
        s = df[c].astype(np.float64)
        if c == "RT":
            s = np.log10(s.clip(lower=0.01))
        for w in (601, 1501):
            r = s.rolling(w, center=True, min_periods=50)
            med = r.median().bfill().ffill()
            cols[f"{c}_ZMED{w}"] = med
            cols[f"{c}_ZSTD{w}"] = r.std(ddof=0).bfill().ffill()
            cols[f"{c}_ZRESID{w}"] = s - med
            cols[f"{c}_ZQ10_{w}"] = r.quantile(0.1).bfill().ffill()
            cols[f"{c}_ZQ90_{w}"] = r.quantile(0.9).bfill().ffill()

    depth = df["DEPTH"].astype(np.float64)
    cols["DEPTH_FROM_TOP"] = depth - depth.min()
    cols["DEPTH_TO_BOTTOM"] = depth.max() - depth
    out = pd.DataFrame(cols).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.astype(np.float32)


def build_all_features(dfs: list):
    """返回 (base, enhanced, extra) 三个特征矩阵（行序与 dfs 拼接一致）。"""
    bs, es, xs = [], [], []
    for d in dfs:
        b, e = base_enhanced_features(d)
        bs.append(b)
        es.append(e)
        xs.append(extra_features(d))
    return (pd.concat(bs, ignore_index=True),
            pd.concat(es, ignore_index=True),
            pd.concat(xs, ignore_index=True))
