"""Matched-capacity residual depth ablation on positive POR distribution readout."""
import torch
from torch import nn
from torch.nn import functional as F
from por_ranked_probability_v151 import ROOT,SOURCE,PARENT,Network as Previous,load_data,Evaluator,channel_objective
from por_joint_density_width_v140 import ContextHead as OriginalHead
STUDY=ROOT/'results/deep_learning/por_deep_readout_v164_20261001'
OUT=ROOT/'outputs/pure_dl_por_deep_readout_v164_fold5'
STAGES={name:('POR',256,False,0.,3e-4) for name in ('linear','nonlinear')}

class DeepHead(OriginalHead):
    def __init__(self,mode):
        super().__init__();self.mode=mode
        self.reduce=nn.Conv1d(256,128,1)
        self.residual_output=nn.Conv1d(128,256,1)
        nn.init.zeros_(self.residual_output.weight);nn.init.zeros_(self.residual_output.bias)
    def forward(self,x,valid):
        h=F.gelu(self.stem(x)).masked_fill(~valid[:,None],0)
        z=self.norm(h.transpose(1,2)).transpose(1,2).masked_fill(~valid[:,None],0)
        h=(h+F.gelu(F.conv1d(z,self.depth.weight.sum(-1,keepdim=True),self.depth.bias,groups=256))).masked_fill(~valid[:,None],0)
        a,b=self.expand(h).chunk(2,1);h=F.gelu(a)*b.sigmoid()
        reduced=self.reduce(h)
        if self.mode=='nonlinear':reduced=F.gelu(reduced)
        delta=self.residual_output(reduced).float()
        bound=.1*h.detach().float().square().mean(1,keepdim=True).sqrt().clamp_min(.01)
        h=h+(bound*torch.tanh(delta/bound)).to(h.dtype)
        return self.output(h).transpose(1,2).float().masked_fill(~valid[...,None],0)

class Network(Previous):
    def __init__(self,master,mode):
        super().__init__(master,mode)
        common=self.context_head.state_dict();self.context_head=DeepHead(mode)
        self.context_head.load_state_dict(common,strict=False)
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(head_residual_mode=self.context_head.mode,residual_bound_hidden_rms=.1,
                    extra_parameters=65920,same_capacity_between_variants=True,
                    common_head_raw_initialization_preserved=True,trained_v151_head_not_loaded=True,
                    historical_v151_patience=18,current_patience=14)
        return info

def make_model(stage,master,metadata):return Network(master,stage)
