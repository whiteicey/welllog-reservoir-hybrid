"""Main prediction receives a learned residual from aligned auxiliary parameters."""
import torch
from torch import nn
from torch.nn import functional as F
from global_cosine_sequence import Network as Baseline


def pack_prediction(output):
    gate,raw=output
    return torch.cat([gate,raw.flatten(-3)],-1).transpose(1,2)


class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        with torch.random.fork_rng(devices=[]):self.prediction_refine=nn.Conv1d(150,75,1)
        nn.init.zeros_(self.prediction_refine.weight)
        nn.init.zeros_(self.prediction_refine.bias)

    def forward(self,features,relative_depth,valid,auxiliary=False,absolute_depth=None):
        main,mid,coarse=super().forward(features,relative_depth,valid,True,absolute_depth=absolute_depth)
        length=main[0].shape[1]
        aligned=[F.interpolate(pack_prediction(output),size=length,mode='linear',align_corners=False) for output in (mid,coarse)]
        correction=self.prediction_refine(torch.cat(aligned,1)).transpose(1,2)
        gate,raw=main
        main=(gate+correction[...,:3],raw+correction[...,3:].reshape_as(raw))
        return (main,mid,coarse) if auxiliary else main
