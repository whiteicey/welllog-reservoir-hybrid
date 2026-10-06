"""Input-conditioned smoothing/sharpening of frozen within-well profiles."""
import numpy as np
import torch
from torch import nn
from frozen_numeric_patch import apply_patch_values


def profile_descriptors(prediction, fixed_default, row_ids, valid):
    result=np.zeros((len(prediction),3,2),np.float32)
    limits=[.04,.03,.025]
    for wi,ids in enumerate(row_ids):
        raw_positions=np.flatnonzero(valid[wi])
        for j in range(3):
            values=prediction[ids,j]
            transformed=np.log10(np.maximum(values,.01)) if j==1 else values
            available=~fixed_default[ids,j]
            raw=np.zeros(len(valid[wi]),np.float64)
            mask=np.zeros_like(raw)
            raw[raw_positions]=np.where(available,transformed,0)
            mask[raw_positions]=available
            for k,size in enumerate((9,33)):
                count=np.convolve(mask,np.ones(size),mode='same')[raw_positions]
                total=np.convolve(raw,np.ones(size),mode='same')[raw_positions]
                mean=np.divide(total,count,out=transformed.copy(),where=count>0)
                delta=mean-transformed if j==1 else mean/np.maximum(values,1e-5)-1
                result[ids,j,k]=np.where(available,np.clip(delta/limits[j],-1,1),0)
    return result.reshape(len(prediction),6)


class NumericPatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(218,32),nn.GELU(),nn.Linear(32,6))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('limits',torch.tensor([.04,.03,.025]))

    def forward(self,inputs):
        coefficient=self.net(inputs).tanh().reshape(-1,3,2)
        evidence=inputs[:,-6:].reshape(-1,3,2)
        return (coefficient*evidence).mean(-1)*self.limits
