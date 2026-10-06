"""Input-only per-curve spike scores with zero-initialized feature correction."""
import warnings
import numpy as np
import torch
from torch import nn
from paired_posterior_router_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES

CURVES=('GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT')


def spike_scores(frame):
    result=np.zeros((len(frame),9),np.float32)
    for column,name in enumerate(CURVES):
        values=frame[name].to_numpy(np.float64)
        valid=np.isfinite(values)
        if not valid.any():continue
        series=frame[name].where(np.isfinite(frame[name]))
        median11=series.rolling(11,center=True,min_periods=1).median().to_numpy()
        residual=values-median11
        residual[~valid]=np.nan
        windows=np.lib.stride_tricks.sliding_window_view(np.pad(residual,(25,25),constant_values=np.nan),51)
        mad=np.empty(len(values),np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            for start in range(0,len(values),4096):
                block=windows[start:start+4096]
                center=np.nanmedian(block,axis=1)
                mad[start:start+len(block)]=np.nanmedian(np.abs(block-center[:,None]),axis=1)
        q25,q75=np.percentile(values[valid],[25,75])
        floor=max(.1*(q75-q25),1e-6)
        denominator=np.maximum(np.nan_to_num(mad,nan=0.)*1.4826,floor)
        score=np.minimum(np.abs(residual)/denominator,10.)/10.
        result[valid,column]=score[valid]
    assert np.isfinite(result).all() and ((result>=0)&(result<=1)).all()
    return result


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices):
        assert features==206
        super().__init__(features,initial_means,retain_indices=retain_indices,
                         numeric_indices=numeric_indices,conditioning_indices=conditioning_indices)
        with torch.random.fork_rng(devices=[]):
            self.spike_adapter=nn.Linear(9,206)
            nn.init.zeros_(self.spike_adapter.weight);nn.init.zeros_(self.spike_adapter.bias)

    def corrected(self, features, valid):
        assert features.shape[-1]==215
        original=features[...,:206]
        side=features[...,206:].masked_fill(~valid[...,None],0)
        correction=self.spike_adapter(side).masked_fill(~valid[...,None],0)
        return (original+correction).to(original.dtype)

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        return super().forward(self.corrected(features,valid),relative_depth,valid,auxiliary,absolute_depth)

    def prediction_parts(self, features, relative_depth, valid, absolute_depth=None):
        return super().prediction_parts(self.corrected(features,valid),relative_depth,valid,absolute_depth)
