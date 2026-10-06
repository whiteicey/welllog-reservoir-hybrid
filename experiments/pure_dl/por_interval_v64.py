"""POR physical tolerance-mass training; retained point-adapter control is reused."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from por_internal_adapter_v63 import ROOT,Network as Parent,load_data,Evaluator
import numpy as np

STUDY=ROOT/'results/deep_learning/por_interval_v64_20260929'
PREVIOUS=ROOT/'results/deep_learning/por_internal_adapter_v63_20260929'
CONTROL=ROOT/'outputs/pure_dl_por_internal_adapter_v63_POR_point_fold5'
MODES=('interval',)

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

class Network(Parent):
    def __init__(self,target,master,mode):
        if mode!='interval':raise ValueError(mode)
        super().__init__(target,master,'point');self.mode=mode
        self.manifest=self.manifest|dict(training_auxiliary='conditional mass inside true physical POR plus/minus8percent, original log1p normalization')

class IntervalLoss(nn.Module):
    def __init__(self,master):
        super().__init__();self.location=float(master['regression_median'][0]);self.scale=float(master['regression_scale'][0])
    def forward(self,output,truth,default,mask,quality):
        raw=output[1][mask][:,0].float();d=default[mask][:,0]
        y=truth[mask][:,0].float().masked_fill(d,0)
        physical=torch.expm1(y*self.scale+self.location)
        radius=.08*(physical.abs()+1e-5)
        lower=(torch.log1p((physical-radius).clamp_min(0))-self.location)/self.scale
        upper=(torch.log1p((physical+radius).clamp_min(0))-self.location)/self.scale
        mu=raw[:,1];sd=F.softplus(raw[:,2])+.025
        phi_low=torch.atan((lower[:,None]-mu)/sd);phi_high=torch.atan((upper[:,None]-mu)/sd)
        # Same df3 CDF used in original por_student3_sequence, now differentiable.
        cdf_low=.5+(phi_low+phi_low.sin()*phi_low.cos())/math.pi
        cdf_high=.5+(phi_high+phi_high.sin()*phi_high.cos())/math.pi
        mass=(raw[:,0].softmax(-1)*(cdf_high-cdf_low).clamp_min(0)).sum(-1)
        penalty=-mass.clamp_min(1e-8).log();penalty=penalty.masked_fill(d,0)
        result=(quality[mask].detach().float()*penalty).mean()/3
        return result,dict(numeric_mass_mean=float(mass[~d].detach().mean()) if bool((~d).any()) else None,requires_grad=result.requires_grad)
