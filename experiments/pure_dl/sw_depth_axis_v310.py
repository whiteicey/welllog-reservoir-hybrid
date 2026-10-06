"""Matched depth-neighbor versus same-point latent-channel residuals."""
import torch
from torch import nn
from sw_raw_well_condition_v297 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,DecisionObjective,Network as Parent
STUDY=ROOT/'results/deep_learning/sw_depth_axis_v310_20261002'
OUT=ROOT/'outputs/pure_dl_sw_depth_axis_v310_fold5'
STAGES={s:('SW',1024,False,0.,3e-4) for s in ('depth','channel')}

class AxisResidual(nn.Module):
    def __init__(self,base,kernel,axis):
        super().__init__();self.base=base;self.kernel=kernel;self.axis=axis;self.valid=None
        self.norm=nn.LayerNorm(48);self.gate=nn.Linear(48,48)
        self.dw=nn.Conv1d(48,48,kernel,padding=kernel//2,groups=48)
        nn.init.zeros_(self.dw.weight);nn.init.zeros_(self.dw.bias)
    def forward(self,h):
        if self.valid is None:raise RuntimeError('Explicit valid mask required')
        invalid=~self.valid.unsqueeze(-1)
        h=self.base(h).masked_fill(invalid,0)
        n=self.norm(h).masked_fill(invalid,0)
        if self.axis=='depth':delta=self.dw(n.transpose(1,2)).transpose(1,2)
        else:
            weights=self.dw.weight[:,0,:]
            delta=sum(torch.roll(n,shifts=-(j-self.kernel//2),dims=-1)*weights[:,j] for j in range(self.kernel))+self.dw.bias
        return (h+torch.sigmoid(self.gate(n))*delta).masked_fill(invalid,0)

class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,'well');self.axis_mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261310)
            self.branches=nn.ModuleList([AxisResidual(b,k,mode) for b,k in zip(self.branches,(3,7,15))])
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        active=sum(p.numel() for p in self.parameters() if p.requires_grad)
        assert active==73953
        info.update(axis_mode=self.axis_mode,new_active_parameters=active,new_axis_parameters=8688,kernels=[3,7,15],
                    all_taps_active_in_both_arms=True,zero_DW_initial_residual=True,original104_only=True,
                    no297_trained_weights_loaded=True,one_SW_additional_phase=True,not_first_depth_context_experiment=True)
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        for branch in self.branches:branch.valid=valid
        try:return super().forward(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
        finally:
            for branch in self.branches:branch.valid=None

def make_model(stage,master,metadata):return Network(master,metadata,stage)
