"""Augment positive256/RPS readout with one bounded default-gate correction."""
import torch
from torch import nn
from torch.nn import functional as F
from por_ranked_probability_v151 import ROOT,SOURCE,PARENT,Network as Parent,load_data,Evaluator,channel_objective
STUDY=ROOT/'results/deep_learning/por_bounded_gate_rps_v157_20261001'
OUT=ROOT/'outputs/pure_dl_por_bounded_gate_rps_v157_fold5'
STAGES={'joint_rps':('POR',256,False,0.,3e-4)}
class Network(Parent):
    def __init__(self,master,mode):
        super().__init__(master,mode);self.context_head.output=nn.Conv1d(256,33,1)
        nn.init.zeros_(self.context_head.output.weight);nn.init.zeros_(self.context_head.output.bias)
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(new_default_gate_correction=True,new_gate_logodds_bound=.125,
            original_gate_parameters_frozen=True,additional_trainable_parameters=257,
            old_all_predictor_weights_frozen=True,no_trained_v151_head_loaded=True)
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        from compact_por_mixed_family_v113 import Network as Frozen
        self.local_captured.clear()
        with torch.no_grad():gate,raw=Frozen.forward(self,x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
        features=torch.cat((self.local_captured[0],self.local_captured[1],x[...,206:].transpose(1,2)),1)
        self.local_captured.clear();u=self.context_head(features,valid);own=raw[:,:,0]
        reference=F.softplus(own[:,:,2].detach())+.025
        corrected=torch.stack((own[:,:,0]+torch.tanh(u[...,16:32]),
            own[:,:,1]+reference*torch.tanh(u[...,:16]/reference),own[:,:,2]),2).unsqueeze(2)
        return gate+.125*torch.tanh(u[...,32:]/.125),corrected
def make_model(stage,master,metadata):return Network(master,stage)
