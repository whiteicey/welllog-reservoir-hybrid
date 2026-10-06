"""Matched additive/multiplicative masked multi-order detail residuals."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from perm_private_width128_v185 import ROOT,SOURCE,PARENT,Network as Parent,load_data,Evaluator,channel_objective,prepare_sampling,sample_anchors

STUDY=ROOT/'results/deep_learning/perm_multiorder_interaction_v192_20261001'
OUT=ROOT/'outputs/pure_dl_perm_multiorder_interaction_v192_fold5'
STAGES={k:('PERM',256,True,0.,6e-4) for k in ('add','multiply')}

class InteractionDetail(nn.Module):
    def __init__(self,base,mode):
        super().__init__();self.base=base;self.mode=mode
        self.norm=nn.LayerNorm(128)
        self.gate=nn.Conv1d(128,128,1)
        self.filters=nn.ModuleList([nn.Conv1d(c,c,5,padding=2*d,dilation=d,groups=c) for c,d in zip((16,48,64),(1,2,3))])
        self.value=nn.Conv1d(128,128,1)
        self.project=nn.Conv1d(128,128,1)
        nn.init.zeros_(self.project.weight);nn.init.zeros_(self.project.bias)

    def forward(self,h,valid):
        z=self.norm(h).masked_fill(~valid[...,None],0).transpose(1,2)
        pieces=[layer(part).masked_fill(~valid[:,None],0) for layer,part in zip(self.filters,z.split((16,48,64),dim=1))]
        g=F.silu(self.gate(z)).masked_fill(~valid[:,None],0)
        v=F.silu(self.value(torch.cat(pieces,dim=1))).masked_fill(~valid[:,None],0)
        joined=g*v if self.mode=='multiply' else (g+v)/math.sqrt(2.)
        residual=.1*self.project(joined).transpose(1,2)
        return (self.base(h,valid)+residual).masked_fill(~valid[...,None],0)

class Network(Parent):
    def __init__(self,master,mode):
        super().__init__(master,mode);self.study_mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010)
            self.body.detail[0]=InteractionDetail(self.body.detail[0],mode)

    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(interaction=self.study_mode,channel_splits=[16,48,64],dilations=[1,2,3],kernel_size=5,
                    first_detail_residual_only=True,residual_scale=.1,project_zero_initialized=True,
                    matched_pair_parameters=True,same_initial_hidden_and_predictive_function=True,
                    same_initial_parameter_tensors=True,no_trained_v185_weights_loaded=True,
                    no_new_training_stage=True,not_exact_MogaNet_reproduction=True)
        return info

def make_model(stage,master,metadata):return Network(master,stage)
