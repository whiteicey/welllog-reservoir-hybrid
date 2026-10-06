"""Small pointwise or depth-local residual readouts with identical initial predictions."""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sw_utility_strength_v82 import ROOT,Network as Parent,Evaluator,load_data,SOURCE,DecisionObjective as ParentObjective

STUDY=ROOT/'results/deep_learning/sw_residual_readout_v85_20260930'
PREVIOUS=ROOT/'results/deep_learning/sw_frozen_gate_utility_v84r1_20260930'
CONTROL=ROOT/'outputs/pure_dl_sw_utility_strength_v82_SW_a100_fold5'
MODES={mode:dict(target='SW',c=2,length=1024,rate=.4,utility_weight=1.) for mode in ('point64','local64')}

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

class ResidualHead(nn.Module):
    def __init__(self,base,mode):
        super().__init__();self.base=base;self.hidden=nn.Linear(128,64)
        self.local=nn.Conv1d(64,64,3,padding=1,groups=64) if mode=='local64' else nn.Identity()
        self.project=nn.Linear(64,25);self.valid=None
        nn.init.zeros_(self.project.weight);nn.init.zeros_(self.project.bias)
        if mode=='local64':
            with torch.no_grad():
                self.local.weight.zero_();self.local.weight[:,0,1]=1.;self.local.bias.zero_()
    @property
    def weight(self):return self.base.weight
    @property
    def bias(self):return self.base.bias
    def forward(self,h):
        if self.valid is None:raise RuntimeError('Explicit original window validity required')
        u=self.hidden(h).masked_fill(~self.valid[...,None],0)
        u=self.local(u.transpose(1,2)).transpose(1,2).masked_fill(~self.valid[...,None],0)
        delta=self.project(F.gelu(u)).masked_fill(~self.valid[...,None],0)
        return self.base(h)+delta

class Network(Parent):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('a100',metadata);self.study_mode=mode
        with torch.random.fork_rng(devices=[]):self.output=ResidualHead(self.output,mode)
        added=sum(p.numel() for p in self.output.parameters())-sum(p.numel() for p in self.output.base.parameters())
        self.parent_info.update(readout_mode=mode,added_parameters=added,hidden_width=64,
            loaded_v82_trained_weights=False,historical_linear_control=str(CONTROL.relative_to(ROOT)),
            residual_output_zero_initialized=True,local_initial_center_identity=mode=='local64',
            original_rng_preserved=True,invalid_neighbors_masked=True,
            only_hyperparameter_changed='residual final readout structure; original body/loss/alpha1/budget/dropkey retained')
    def base_state_hash(self):
        values=[]
        for name,value in self.state_dict().items():
            if name.startswith(('output.hidden.','output.local.','output.project.')):continue
            values.append(value.detach().cpu().numpy().tobytes())
        return hashlib.sha256(b''.join(values)).hexdigest()
    def forward(self,x,relative,valid,absolute_depth=None,return_aux=False,generator=None,well_inputs=None):
        self.output.valid=valid
        try:return super().forward(x,relative,valid,absolute_depth,return_aux,generator,well_inputs)
        finally:self.output.valid=None

class DecisionObjective(ParentObjective):
    def __init__(self,evaluator,metadata,mode):
        super().__init__(evaluator,metadata,'a100')
