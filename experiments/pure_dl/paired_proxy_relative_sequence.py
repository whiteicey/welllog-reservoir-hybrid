"""Within-well relative coordinates of four non-affine composite proxies."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from paired_posterior_router_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES

PROXIES=('CNL_DENSITY_POROSITY_DIFF','CNL_DENSITY_POROSITY_MEAN','ARCHIE_PROXY','INVASION_LOG_DIFF')


def proxy_relative_features(enhanced, names):
    assert all(names.count(name)==1 for name in PROXIES)
    sides=[]
    for name in PROXIES:
        series=pd.Series(enhanced[:,names.index(name)].astype(np.float64))
        assert np.isfinite(series.to_numpy()).all()
        median=float(series.median())
        q05,q25,q75,q95=[float(series.quantile(q)) for q in (.05,.25,.75,.95)]
        iqr=max(q75-q25,1e-6);span=max(q95-q05,1e-6)
        sides.extend([((series-median)/iqr).clip(-8.,8.).to_numpy(np.float32),
                      ((series-q05)/span).clip(-8.,8.).to_numpy(np.float32),
                      series.rank(method='average',pct=True).to_numpy(np.float32)])
    result=np.column_stack(sides)
    assert result.shape==(len(enhanced),12) and np.isfinite(result).all()
    return result


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices):
        assert features==206
        super().__init__(features,initial_means,retain_indices=retain_indices,
                         numeric_indices=numeric_indices,conditioning_indices=conditioning_indices)
        with torch.random.fork_rng(devices=[]):
            self.proxy_adapter=nn.Linear(12,206)
            nn.init.zeros_(self.proxy_adapter.weight);nn.init.zeros_(self.proxy_adapter.bias)

    def corrected(self, features, valid):
        assert features.shape[-1]==218
        # Base206 was already rounded exactly as original float16 preprocessing.
        original=features[...,:206].to(torch.float16 if features.device.type=='cuda' else features.dtype)
        side=features[...,206:].float().masked_fill(~valid[...,None],0)
        correction=self.proxy_adapter(side).masked_fill(~valid[...,None],0)
        return (original+correction).to(original.dtype)

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        return super().forward(self.corrected(features,valid),relative_depth,valid,auxiliary,absolute_depth)

    def prediction_parts(self, features, relative_depth, valid, absolute_depth=None):
        return super().prediction_parts(self.corrected(features,valid),relative_depth,valid,absolute_depth)
