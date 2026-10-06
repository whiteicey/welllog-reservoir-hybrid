"""Signed, robust local curve contrast as an input-only side view."""
import numpy as np
import pandas as pd
import torch
from torch import nn

from paired_posterior_router_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES
from paired_well_view_sequence import RAW_CURVES


def signed_local_contrast(frame):
    """Nine signed local contrasts; each well and curve is processed independently."""
    result = np.zeros((len(frame), len(RAW_CURVES)), np.float32)
    for column, name in enumerate(RAW_CURVES):
        raw = frame[name].to_numpy(np.float64)
        finite = np.isfinite(raw)
        if not finite.any():
            continue
        if name in ('RXO', 'RT'):
            raw = np.where(finite, np.log1p(np.maximum(raw, 0)), np.nan)
        series = pd.Series(raw)
        center = series.rolling(31, center=True, min_periods=1).median()
        residual = series - center
        local_mad = residual.abs().rolling(101, center=True, min_periods=1).median().to_numpy(np.float64)
        q25, q75 = np.quantile(raw[finite], [.25, .75])
        floor = max(.1 * (q75 - q25), 1e-6)
        denominator = np.maximum(1.4826 * np.nan_to_num(local_mad, nan=0), floor)
        contrast = np.clip(residual.to_numpy(np.float64) / denominator, -8, 8) / 8
        result[finite, column] = np.nan_to_num(contrast[finite], nan=0, posinf=0, neginf=0)
    assert np.isfinite(result).all() and np.max(np.abs(result)) <= 1.0
    return result


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices):
        assert features == 206
        super().__init__(features, initial_means, retain_indices=retain_indices,
                         numeric_indices=numeric_indices, conditioning_indices=conditioning_indices)
        with torch.random.fork_rng(devices=[]):
            self.contrast_adapter = nn.Linear(9, 206)
            nn.init.zeros_(self.contrast_adapter.weight)
            nn.init.zeros_(self.contrast_adapter.bias)

    def corrected(self, features, valid):
        assert features.shape[-1] == 215
        original = features[..., :206]
        contrast = features[..., 206:].float().masked_fill(~valid[..., None], 0)
        correction = self.contrast_adapter(contrast).masked_fill(~valid[..., None], 0)
        return (original.float() + correction).to(original.dtype)

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        return super().forward(self.corrected(features, valid), relative_depth, valid, auxiliary, absolute_depth)

    def prediction_parts(self, features, relative_depth, valid, absolute_depth=None):
        return super().prediction_parts(self.corrected(features, valid), relative_depth, valid, absolute_depth)
