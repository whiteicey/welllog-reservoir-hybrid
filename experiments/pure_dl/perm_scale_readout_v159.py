"""PERM joint conditional location, weight and bounded raw-scale readout."""
import torch
from torch import nn
from torch.nn import functional as F
from perm_ranked_probability_v154 import ROOT,SOURCE,PARENT,Network as Previous,load_data,Evaluator,channel_objective
from compact_five_stage_v100 import CompactReadout
STUDY=ROOT/'results/deep_learning/perm_scale_readout_v159_20261001'
OUT=ROOT/'outputs/pure_dl_perm_scale_readout_v159_fold5'
STAGES={'perm_conditional':('PERM',256,False,0.,3e-4)}

class Network(Previous):
    def __init__(self,master,mode):
        super().__init__(master,mode)
        old=self.context_head.output
        new=nn.Conv1d(128,48,1)
        nn.init.zeros_(new.weight);nn.init.zeros_(new.bias)
        with torch.no_grad():
            new.weight[:32].copy_(old.weight);new.bias[:32].copy_(old.bias)
        self.context_head.output=new
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(same_capacity_as_v136=False,new_raw_scale_bound=.5,
                    additional_parameters=2064,trained_v154_head_not_loaded=True,
                    matched_initial_function=True,all_original_parameters_frozen=True)
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.local_captured.clear()
        with torch.no_grad():
            gate,raw=CompactReadout.forward(self,x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
        features=torch.cat((self.local_captured[0],self.local_captured[1],x[...,206:].transpose(1,2)),1)
        self.local_captured.clear();u=self.context_head(features,valid)
        own=raw[:,:,0];reference=F.softplus(own[:,:,2].detach())+.025
        delta=reference*torch.tanh(u[...,:16]/reference)
        weight=torch.tanh(u[...,16:32])
        scale=.5*torch.tanh(u[...,32:]/.5)
        corrected=torch.stack((own[:,:,0]+weight,own[:,:,1]+delta,own[:,:,2]+scale),2)
        return gate,corrected.unsqueeze(2)

def make_model(stage,master,metadata):return Network(master,stage)
