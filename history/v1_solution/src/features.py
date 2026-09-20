from __future__ import annotations

import numpy as np
import pandas as pd

from .data import INPUT_COLUMNS


BASE_COLUMNS = [column for column in INPUT_COLUMNS if column != "CASE"]
KEY_CURVES = ["GR", "AC", "DEN", "CNL", "RT", "RXO"]
ROLLING_WINDOWS = (9, 51, 201)


class FeatureBuilder:
    """Deterministic feature engineering with optional well-level context."""

    def __init__(self, use_well_context: bool = True) -> None:
        self.use_well_context = use_well_context
        self.medians_: dict[str, float] = {}
        self.feature_names_: list[str] = []

    def fit(self, frames: list[pd.DataFrame]) -> "FeatureBuilder":
        combined = pd.concat(frames, ignore_index=True)
        self.medians_ = {
            column: float(combined[column].median())
            for column in BASE_COLUMNS
        }
        first = self._transform_frame(frames[0])
        self.feature_names_ = first.columns.tolist()
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.medians_:
            raise RuntimeError("FeatureBuilder must be fitted before transform")
        transformed = self._transform_frame(frame)
        if transformed.columns.tolist() != self.feature_names_:
            raise RuntimeError("Feature schema changed during transform")
        values = transformed.to_numpy(dtype=np.float32, copy=True)
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    def _transform_frame(self, raw_frame: pd.DataFrame) -> pd.DataFrame:
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
            frame[column] = series

        columns: dict[str, pd.Series] = {}
        for column in BASE_COLUMNS:
            columns[column] = frame[column].astype(np.float32)

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

        if self.use_well_context:
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

            for column in KEY_CURVES:
                series = columns[column]
                columns[f"{column}_DIFF"] = series.diff().fillna(0.0)
                columns[f"{column}_TREND"] = (
                    series.rolling(51, center=True, min_periods=1).mean()
                    - series.rolling(201, center=True, min_periods=1).mean()
                ).fillna(0.0)

        features = pd.concat(columns, axis=1)
        return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
