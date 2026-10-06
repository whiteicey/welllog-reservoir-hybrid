"""Correct POR log1p inverse; original active v38 SW source remains immutable."""
import torch
from action_value_v38 import *
from action_value_v38 import Objective as OriginalObjective

class Objective(OriginalObjective):
    @torch.no_grad()
    def labels(self,normalized,default,mask):
        m=self.model
        if m.target!='POR':raise ValueError('This correction is scoped to POR')
        d=default[mask][:,0]
        y=torch.expm1(normalized[mask][:,0].float()*m.scale+m.median)
        y=torch.where(d,torch.full_like(y,.1),y)
        return (1-(m.actions[None]-y[:,None]).abs()/(.08*(y.abs()[:,None]+1e-5))).clamp_min(0)
