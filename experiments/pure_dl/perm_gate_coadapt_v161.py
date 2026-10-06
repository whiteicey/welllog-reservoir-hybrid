"""PERM positive conditional readout with own original expert gate coadaptation."""
import torch
from torch.nn import functional as F
from perm_ranked_probability_v154 import ROOT,SOURCE,PARENT,Network as Previous,load_data,Evaluator,channel_objective
from compact_five_stage_v100 import CompactReadout
STUDY=ROOT/'results/deep_learning/perm_gate_coadapt_v161_20261001'
OUT=ROOT/'outputs/pure_dl_perm_gate_coadapt_v161_fold5'
STAGES={'perm_conditional':('PERM',256,False,0.,3e-4)}

class Network(Previous):
    def __init__(self,master,mode):
        super().__init__(master,mode)
        for head in self.heads:
            head.gate_weight.requires_grad_(True)
            head.gate_bias.requires_grad_(True)
    def groups(self):
        gates=[p for head in self.heads for p in (head.gate_weight,head.gate_bias)]
        return [dict(params=list(self.context_head.parameters()),lr=3e-4,name='PERM_location_weight_head'),
                dict(params=gates,lr=3e-5,name='PERM_original_gate_rows')]
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(old_predictor_fully_frozen=False,own_expert_gate_rows_trainable=True,
                    original_backbone_router_scale_frozen=True,gate_lr=3e-5,
                    active_parameter_increase=258,total_architecture_capacity_unchanged=True,
                    trained_v154_head_not_loaded=True)
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.local_captured.clear()
        # CompactReadout freezes the backbone internally, but runs own packed
        # target heads outside no_grad so the selected gate rows receive loss.
        gate,raw=CompactReadout.forward(self,x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
        features=torch.cat((self.local_captured[0],self.local_captured[1],x[...,206:].transpose(1,2)),1)
        self.local_captured.clear();u=self.context_head(features,valid)
        own=raw[:,:,0];reference=F.softplus(own[:,:,2].detach())+.025
        corrected=torch.stack((own[:,:,0]+torch.tanh(u[...,16:]),
                    own[:,:,1]+reference*torch.tanh(u[...,:16]/reference),own[:,:,2]),2)
        return gate,corrected.unsqueeze(2)

def make_model(stage,master,metadata):return Network(master,stage)
