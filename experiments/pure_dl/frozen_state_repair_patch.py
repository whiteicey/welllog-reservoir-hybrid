"""Train expected utility gain of a fixed, input-only alternate action."""
import math
import numpy as np
import torch
from torch import nn


def alternate_predictions(prediction,defaults,row_ids,valid,default_values,bounds):
    alternate=prediction.copy();support=np.zeros_like(prediction)
    for wi,ids in enumerate(row_ids):
        positions=np.flatnonzero(valid[wi])
        for j in range(3):
            available=~defaults[ids,j]
            values=prediction[ids,j]
            transformed=np.log10(np.maximum(values,.01)) if j==1 else values
            raw=np.zeros(len(valid[wi]));mask=np.zeros_like(raw)
            raw[positions]=np.where(available,transformed,0);mask[positions]=available
            count=np.convolve(mask,np.ones(33),mode='same')[positions]
            total=np.convolve(raw,np.ones(33),mode='same')[positions]
            mean=np.divide(total,count,out=transformed.copy(),where=count>0)
            value=10**mean if j==1 else mean
            candidate=np.where(defaults[ids,j],np.where(count>0,value,values),default_values[j])
            alternate[ids,j]=np.clip(candidate,*bounds[j]);support[ids,j]=count/33
    return alternate,support


class RepairPatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(218,32),nn.GELU(),nn.Linear(32,3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias,math.atanh(-.1))
    def forward(self,features):return self.net(features).tanh()


def rewards(value,truth):
    error=torch.stack(((value[:,0]-truth[:,0]).abs()/(.08*(truth[:,0].abs()+1e-5)),
                       (value[:,1]/truth[:,1].clamp_min(1e-5)).clamp_min(1e-5).log10().abs(),
                       (value[:,2]-truth[:,2]).abs()/(.05*(truth[:,2].abs()+1e-5))),-1)
    return (1-error).clamp(0,1)
