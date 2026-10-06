"""Compact PERM nonlinear full-density residual, cold original-source initialization."""
import torch
from torch import nn
from torch.nn import functional as F
from compact_five_stage_v100 import ROOT,SOURCE,CompactReadout,PackedHead,load_data as original_data
STUDY=ROOT/'results/deep_learning/compact_perm_density_head_v117_20261001'
OUT=ROOT/'outputs/pure_dl_compact_perm_density_head_v117_fold5'
STAGES={'joint_density':('PERM',256,False,0.,3e-5)}
class DensityHead(PackedHead):
    def forward(self,h,side,valid):
        gate,raw=super().forward(h,side,valid)
        x=torch.cat((h,side.transpose(1,2).to(h.dtype)),1).masked_fill(~valid[:,None],0)
        proposal=self.density_delta(F.gelu(self.density_reduce(x))).transpose(1,2).reshape(len(h),-1,1,3,8).float()
        width=F.softplus(raw[:,:,:,2].detach())+.025
        delta=torch.stack((.25*torch.tanh(proposal[:,:,:,0]/.25),(.25*width)*torch.tanh(proposal[:,:,:,1]/(.25*width)),.25*torch.tanh(proposal[:,:,:,2]/.25)),3)
        return gate,(raw+delta).masked_fill(~valid[:,:,None,None,None],0)
class Network(CompactReadout):
    def __init__(self,master):
        super().__init__(master,1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261117)
            for h in self.heads:
                h.__class__=DensityHead;h.density_reduce=nn.Conv1d(191,32,1);h.density_delta=nn.Conv1d(32,24,1)
                nn.init.zeros_(h.density_delta.weight);nn.init.zeros_(h.density_delta.bias)
    def initialize_priors(self,data):
        return dict(source_only=True,old_downstream_not_loaded=True,new_density_head='191-32-24',zero_initial_function_matches_v100=True,
            bounded_delta_weights=.25,bounded_location_fraction=.25,bounded_raw_scale=.25,source_density_and_backbone_frozen=True,
            old_gate_side_routes_train=True,proper_Gaussian_likelihood_and_CDF=True,no_extra_training_stage=True)
def load_data(stage):return original_data('perm')
def make_model(stage,master,metadata):return Network(master)
