"""Predeclared frozen default-atom/conditional-density factorization, no voting."""
import hashlib
import numpy as np
import torch
from torch import nn
from sw_sam_radius_v94 import ROOT,Network as SamNetwork,Evaluator,load_data
from sw_utility_strength_v82 import Network as GaussianNetwork

STUDY=ROOT/'results/deep_learning/sw_factorization_v95_20260930'
PREVIOUS=ROOT/'results/deep_learning/sw_sam_radius_v94_20260930'
GATE=ROOT/'outputs/pure_dl_sw_sam_radius_v94_SW_r100_fold5/model_best.pt'
DENSITY=ROOT/'outputs/pure_dl_sw_utility_strength_v82_SW_a100_fold5/model_best.pt'
MODES=('same','separate')

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

class Network(nn.Module):
    def __init__(self,mode,metadata):
        super().__init__()
        if mode not in MODES:raise ValueError(mode)
        self.mode=mode;self.c=2;self.length=1024
        self.atom=SamNetwork('r100',metadata)
        self.density=SamNetwork('r100',metadata) if mode=='same' else GaussianNetwork('a100',metadata)
        for branch,path in ((self.atom,GATE),(self.density,GATE if mode=='same' else DENSITY)):
            saved=torch.load(path,map_location='cpu',weights_only=False)
            branch.load_state_dict(saved['state_dict'],strict=True)
        self.requires_grad_(False);self.eval()

    def forward(self,x,relative,valid,absolute_depth=None,return_aux=False,generator=None,well_inputs=None):
        # One coherent conditional law: g at the atom, (1-g)*p(y|numeric).
        # The same formula and fixed source assignments apply to every point.
        gate,_=self.atom(x,relative,valid,well_inputs=well_inputs)
        _,raw=self.density(x,relative,valid,well_inputs=well_inputs)
        out=(gate,raw)
        return (out,raw.new_zeros(()),None) if return_aux else out
