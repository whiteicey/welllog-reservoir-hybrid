"""Feature construction for one well, using only that well's input curves.

FeatureBuilder      : 113 base features / 237 enhanced features (identical to the v2 system)
well_extra_features : 181 additional lag, short-window and formation-scale (60 m / 150 m) features
The training-set medians used for imputation are restored from model_meta.json at inference time,
so no training data is needed to run predict.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import INPUT_COLUMNS


BASE_COLUMNS = [column for column in INPUT_COLUMNS if column != "CASE"]
KEY_CURVES = ["GR", "AC", "DEN", "CNL", "RT", "RXO"]
CONTEXT_CURVES = ["GR", "PE", "SP", "CAL", "AC", "DEN", "CNL", "RXO", "RT", "DEVI", "BIT"]
LOCAL_CURVES = ["GR", "PE", "SP", "CAL", "AC", "DEN", "CNL", "RXO", "RT"]
ROLLING_WINDOWS = (9, 51, 201)


@dataclass(frozen=True)
class FeatureMatrices:
    base: np.ndarray
    enhanced: np.ndarray


class FeatureBuilder:
    """Build deterministic point, well-context, and multi-scale sequence features."""

    def __init__(self) -> None:
        self.medians_: dict[str, float] = {}
        self.base_feature_names_: list[str] = []
        self.enhanced_feature_names_: list[str] = []

    def fit(self, frames: list[pd.DataFrame]) -> "FeatureBuilder":
        if not frames:
            raise ValueError("At least one frame is required")
        combined = pd.concat(frames, ignore_index=True)
        self.medians_ = {
            column: float(combined[column].median())
            for column in BASE_COLUMNS
        }
        base, enhanced = self._transform_frames(frames[0])
        self.base_feature_names_ = base.columns.tolist()
        self.enhanced_feature_names_ = enhanced.columns.tolist()
        return self

    @classmethod
    def from_metadata(cls, meta: dict) -> "FeatureBuilder":
        """Restore a fitted builder from model_meta.json (medians + feature schemas)."""
        builder = cls()
        builder.medians_ = {k: float(v) for k, v in meta["medians"].items()}
        builder.base_feature_names_ = list(meta["base_feature_names"])
        builder.enhanced_feature_names_ = list(meta["enhanced_feature_names"])
        return builder

    def transform(self, frame: pd.DataFrame) -> FeatureMatrices:
        if not self.medians_:
            raise RuntimeError("FeatureBuilder must be fitted before transform")
        base, enhanced = self._transform_frames(frame)
        if base.columns.tolist() != self.base_feature_names_:
            raise RuntimeError("Base feature schema changed during transform")
        if enhanced.columns.tolist() != self.enhanced_feature_names_:
            raise RuntimeError("Enhanced feature schema changed during transform")
        return FeatureMatrices(
            base=np.nan_to_num(base.to_numpy(dtype=np.float32, copy=True)),
            enhanced=np.nan_to_num(enhanced.to_numpy(dtype=np.float32, copy=True)),
        )

    def _impute(self, raw_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
        frame = raw_frame.copy()
        missing = {
            column: frame[column].isna().astype(np.float32)
            for column in BASE_COLUMNS
        }
        for column in BASE_COLUMNS:
            series = frame[column]
            if series.isna().any():
                series = series.interpolate(method="linear", limit_direction="both")
                series = series.fillna(self.medians_[column])
            frame[column] = series.astype(np.float64)
        return frame, missing

    def _transform_frames(self, raw_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        frame, missing = self._impute(raw_frame)
        columns: dict[str, pd.Series] = {
            column: frame[column].astype(np.float32)
            for column in BASE_COLUMNS
        }

        columns["GR_LOG"] = np.log1p(columns["GR"]).astype(np.float32)
        columns["RXO_LOG"] = np.log1p(columns["RXO"]).astype(np.float32)
        columns["RT_LOG"] = np.log1p(columns["RT"]).astype(np.float32)
        columns["AZIM_SIN"] = np.sin(np.deg2rad(columns["AZIM"])).astype(np.float32)
        columns["AZIM_COS"] = np.cos(np.deg2rad(columns["AZIM"])).astype(np.float32)
        columns["DEVI_SQ"] = (columns["DEVI"] ** 2).astype(np.float32)

        depth_min = float(columns["DEPTH"].min())
        depth_max = float(columns["DEPTH"].max())
        depth_span = max(depth_max - depth_min, 1.0)
        columns["DEPTH_REL"] = ((columns["DEPTH"] - depth_min) / depth_span).astype(np.float32)

        columns["CAL_MINUS_BIT"] = columns["CAL"] - columns["BIT"]
        columns["CAL_BIT_RATIO"] = columns["CAL"] / columns["BIT"].clip(lower=0.1)
        columns["RT_RXO_RATIO"] = columns["RT"] / columns["RXO"].clip(lower=0.01)
        columns["RT_RXO_LOG_RATIO"] = columns["RT_LOG"] - columns["RXO_LOG"]
        columns["AC_DEN_RATIO"] = columns["AC"] / columns["DEN"].clip(lower=0.5)
        columns["CNL_DEN_RATIO"] = columns["CNL"] / columns["DEN"].clip(lower=0.5)
        columns["GR_CNL_PRODUCT"] = columns["GR"] * columns["CNL"]
        columns["AC_CNL_PRODUCT"] = columns["AC"] * columns["CNL"]
        columns["DEN_CNL_PRODUCT"] = columns["DEN"] * columns["CNL"]
        columns["GR_AC_PRODUCT"] = columns["GR"] * columns["AC"]
        columns["GR_DEN_PRODUCT"] = columns["GR"] * columns["DEN"]
        columns["GR_RTV_PRODUCT"] = columns["GR"] / columns["RT"].clip(lower=0.01)
        columns["SP_GR_DIFF"] = columns["SP"] - columns["GR"]
        columns["DEN_AC_DIFF"] = columns["DEN"] - columns["AC"] / 100.0

        for column, mask in missing.items():
            columns[f"{column}_MISSING"] = mask

        for column in KEY_CURVES:
            series = columns[column]
            well_median = float(series.median())
            well_std = max(float(series.std(ddof=0)), 1e-6)
            columns[f"{column}_WELL_MEDIAN"] = pd.Series(
                np.full(len(series), well_median, dtype=np.float32), index=series.index
            )
            columns[f"{column}_WELL_STD"] = pd.Series(
                np.full(len(series), well_std, dtype=np.float32), index=series.index
            )
            columns[f"{column}_WELL_RANK"] = series.rank(method="average", pct=True)

        for column in KEY_CURVES:
            series = columns[column]
            for window in ROLLING_WINDOWS:
                rolling = series.rolling(window=window, center=True, min_periods=1)
                columns[f"{column}_ROLL_MEAN_{window}"] = rolling.mean()
                columns[f"{column}_ROLL_STD_{window}"] = rolling.std(ddof=0).fillna(0.0)
            columns[f"{column}_DIFF"] = series.diff().fillna(0.0)
            columns[f"{column}_TREND"] = (
                series.rolling(51, center=True, min_periods=1).mean()
                - series.rolling(201, center=True, min_periods=1).mean()
            ).fillna(0.0)

        base = pd.concat(columns, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        enhanced: dict[str, pd.Series] = {name: base[name] for name in base.columns}

        for column in CONTEXT_CURVES:
            series = base[column].astype(np.float64)
            median = float(series.median())
            q05 = float(series.quantile(0.05))
            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            q95 = float(series.quantile(0.95))
            iqr = max(q75 - q25, 1e-6)
            span = max(q95 - q05, 1e-6)
            enhanced[f"{column}_WELL_ZIQR"] = ((series - median) / iqr).astype(np.float32)
            enhanced[f"{column}_WELL_QNORM"] = ((series - q05) / span).astype(np.float32)
            if column not in KEY_CURVES:
                enhanced[f"{column}_WELL_RANK"] = series.rank(method="average", pct=True)

        for column in LOCAL_CURVES:
            series = base[column].astype(np.float64)
            for window in ROLLING_WINDOWS:
                rolling = series.rolling(window=window, center=True, min_periods=1)
                enhanced[f"{column}_RESID_MEAN_{window}"] = (series - rolling.mean()).astype(np.float32)
                enhanced[f"{column}_RESID_MEDIAN_{window}"] = (series - rolling.median()).astype(np.float32)
                if window < 201:
                    enhanced[f"{column}_ROLL_RANGE_{window}"] = (
                        rolling.max() - rolling.min()
                    ).astype(np.float32)
            for lag in (5, 20):
                enhanced[f"{column}_SLOPE_{lag}"] = (
                    series.diff(lag).fillna(0.0) / float(lag)
                ).astype(np.float32)

        den = base["DEN"].astype(np.float64)
        ac = base["AC"].astype(np.float64)
        cnl = base["CNL"].astype(np.float64)
        gr = base["GR"].astype(np.float64)
        rt = base["RT"].astype(np.float64)
        rxo = base["RXO"].astype(np.float64)
        density_porosity = ((2.65 - den) / 1.65 * 100.0).clip(-30.0, 100.0)
        sonic_porosity = ((ac - 182.0) / 438.0 * 100.0).clip(-30.0, 100.0)
        gr_q05 = float(gr.quantile(0.05))
        gr_q95 = float(gr.quantile(0.95))
        shale_index = ((gr - gr_q05) / max(gr_q95 - gr_q05, 1e-6)).clip(0.0, 1.0)
        porosity_proxy = ((cnl.clip(lower=0.1) + density_porosity.clip(lower=0.1)) / 200.0).clip(0.001, 0.8)

        enhanced["DENSITY_POROSITY"] = density_porosity.astype(np.float32)
        enhanced["SONIC_POROSITY"] = sonic_porosity.astype(np.float32)
        enhanced["CNL_DENSITY_POROSITY_DIFF"] = (cnl - density_porosity).astype(np.float32)
        enhanced["CNL_DENSITY_POROSITY_MEAN"] = ((cnl + density_porosity) / 2.0).astype(np.float32)
        enhanced["SHALE_INDEX_GR"] = shale_index.astype(np.float32)
        enhanced["ARCHIE_PROXY"] = (
            np.log10(rt.clip(lower=0.01)) + 2.0 * np.log10(porosity_proxy)
        ).astype(np.float32)
        enhanced["INVASION_LOG_DIFF"] = (
            np.log10(rt.clip(lower=0.01)) - np.log10(rxo.clip(lower=0.01))
        ).astype(np.float32)

        enhanced_frame = pd.concat(enhanced, axis=1)
        enhanced_frame = enhanced_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return base, enhanced_frame


# ----------------------------------------------------------------------------- extra features
LAG_CURVES = ["GR", "AC", "DEN", "CNL", "RT", "RXO", "PE", "SP"]
LAGS = (-3, -2, -1, 1, 2, 3)


def well_extra_features(frame: pd.DataFrame, medians: dict) -> pd.DataFrame:
    """181 extra features: lag differences, 3/5-sample medians, derivatives, density-neutron separation,
    60 m / 150 m zone descriptors and distance to the top/bottom of the logged interval."""
    cols = {}
    f = frame.copy()
    for c in LAG_CURVES:
        s = f[c]
        if s.isna().any():
            s = s.interpolate(limit_direction="both").fillna(medians.get(c, 0.0))
        if c in ("RT", "RXO"):
            s = np.log10(s.clip(lower=0.01))
        s = s.astype(np.float64)
        for lag in LAGS:
            cols[f"{c}_LAG{lag:+d}"] = (s.shift(lag) - s).bfill().ffill()
        for w in (3, 5):
            med = s.rolling(w, center=True, min_periods=1).median()
            cols[f"{c}_MED{w}"] = med
            cols[f"{c}_MED{w}_RESID"] = s - med
        d1 = s.diff().fillna(0.0)
        cols[f"{c}_D1"] = d1
        cols[f"{c}_D1_NEXT"] = (-s.diff(-1)).fillna(0.0)
        cols[f"{c}_CURV"] = (s.shift(-1) - 2 * s + s.shift(1)).fillna(0.0)
    # neighbourhood density-neutron separation and its lags (lithology/gas indicator)
    den = f["DEN"].interpolate(limit_direction="both").fillna(medians.get("DEN", 2.5))
    cnl = f["CNL"].interpolate(limit_direction="both").fillna(medians.get("CNL", 20.0))
    dphi = (2.65 - den) / 1.65 * 100.0
    sep = cnl - dphi
    cols["DN_SEP"] = sep
    for lag in (-2, -1, 1, 2):
        cols[f"DN_SEP_LAG{lag:+d}"] = (sep.shift(lag) - sep).bfill().ffill()
    # long-window zone descriptors (formation-scale context: 60 m and 150 m)
    for c in ("GR", "SP", "DEN", "AC", "CNL", "RT", "PE"):
        s = f[c]
        if s.isna().any():
            s = s.interpolate(limit_direction="both").fillna(medians.get(c, 0.0))
        if c == "RT":
            s = np.log10(s.clip(lower=0.01))
        s = s.astype(np.float64)
        for w in (601, 1501):
            r = s.rolling(w, center=True, min_periods=50)
            med = r.median().bfill().ffill()
            cols[f"{c}_ZMED{w}"] = med
            cols[f"{c}_ZSTD{w}"] = r.std(ddof=0).bfill().ffill()
            cols[f"{c}_ZRESID{w}"] = s - med
            cols[f"{c}_ZQ10_{w}"] = r.quantile(0.1).bfill().ffill()
            cols[f"{c}_ZQ90_{w}"] = r.quantile(0.9).bfill().ffill()
    depth = f["DEPTH"].astype(np.float64)
    cols["DEPTH_FROM_TOP"] = depth - depth.min()
    cols["DEPTH_TO_BOTTOM"] = depth.max() - depth
    out = pd.DataFrame(cols).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.astype(np.float32)


def feature_sets(mats: FeatureMatrices, extra: np.ndarray) -> dict[str, np.ndarray]:
    """The four feature sets referenced by model_meta.json."""
    return {
        "base": mats.base,
        "enh": mats.enhanced,
        "base+extra": np.c_[mats.base, extra],
        "enh+extra": np.c_[mats.enhanced, extra],
    }
