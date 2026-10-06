"""Augment effective256 distribution readout with matched additive/product features."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from por_joint_density_width_v140 import ROOT,SOURCE,PARENT,Network as Parent,ContextHead as BaseHead,load_data,Evaluator,channel_objective
STUDY=ROOT/'results/deep_learning/por_expert_interaction_v152_20261001'
OUT=ROOT/'outputs/pure_dl_por_expert_interaction_v152_fold5'
STAGES={s:('POR',256,False,0.,3e-4) for s in ('additive','product')}

class Head(BaseHead):
    def __init__(self,mode):
        super().__init__();self.mode=mode
        self.fine_project=nn.Linear(128,32);self.coarse_project=nn.Linear(128,32)
        self.interaction_norm=nn.LayerNorm(32);self.interaction_output=nn.Linear(32,256)
        nn.init.zeros_(self.interaction_output.weight);nn.init.zeros_(self.interaction_output.bias)
    def forward(self,x,valid):
        h=F.gelu(self.stem(x)).masked_fill(~valid[:,None],0)
        z=self.norm(h.transpose(1,2)).transpose(1,2).masked_fill(~valid[:,None],0)
        h=(h+F.gelu(F.conv1d(z,self.depth.weight.sum(-1,keepdim=True),self.depth.bias,groups=256))).masked_fill(~valid[:,None],0)
        fine=F.layer_norm(x[:,:128].transpose(1,2),(128,))
        coarse=F.layer_norm(x[:,128:256].transpose(1,2),(128,))
        a=self.fine_project(fine);b=self.coarse_project(coarse)
        interaction=a*b if self.mode=='product' else (a+b)/math.sqrt(2)
        branch=self.interaction_output(self.interaction_norm(interaction)).transpose(1,2)
        h=(h+branch).masked_fill(~valid[:,None],0)
        a,b=self.expand(h).chunk(2,1)
        return self.output(F.gelu(a)*b.sigmoid()).transpose(1,2).float().masked_fill(~valid[...,None],0)

class Network(Parent):
    def __init__(self,master,mode):
        super().__init__(master,mode)
        # Preserve the exact effective256 common initialization.
        state=self.context_head.state_dict()
        head=Head(mode);head.load_state_dict(state,strict=False)
        # Both arms execute the same random initialization sequence.
        # Shared body values are restored from the existing common head.
        self.context_head=head
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(expert_low_rank_interaction=self.context_head.mode,rank=32,
            matched_candidate_capacity_and_initialization=True,common_effective256_weights_preserved=True,
            added_parameters=16768,branch_zero_initialization=True,trained_v140_weights_not_loaded=True)
        return info

def make_model(stage,master,metadata):return Network(master,stage)
