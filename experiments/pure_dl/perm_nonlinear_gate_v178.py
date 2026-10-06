"""Joint-view nonlinear default-state correction on positive conditionalCDF readout."""
import torch
from torch import nn
from torch.nn import functional as F
from perm_conditional_cdf_v174 import ROOT,SOURCE,PARENT,Network as Parent,load_data,Evaluator,channel_objective
from compact_five_stage_v100 import CompactReadout
STUDY=ROOT/'results/deep_learning/perm_nonlinear_gate_v178_20261001'
OUT=ROOT/'outputs/pure_dl_perm_nonlinear_gate_v178_fold5'
STAGES={'perm_conditional':('PERM',256,True,0.,3e-4)}
class Network(Parent):
    def __init__(self,master,mode):
        super().__init__(master,mode)
        self.context_head.output=nn.Conv1d(128,33,1)
        nn.init.zeros_(self.context_head.output.weight);nn.init.zeros_(self.context_head.output.bias)
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(new_nonlinear_gate_logit_correction=True,gate_logit_bound=1.,
                    original_gate_parameters_frozen=True,gate_function_can_change=True,
                    shared_conditional_state_readout=True,active_extra_parameters=129,
                    same_capacity_as_v154=False,no_trained_v174_weights_loaded=True)
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.local_captured.clear()
        with torch.no_grad():gate,raw=CompactReadout.forward(self,x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
        features=torch.cat((self.local_captured[0],self.local_captured[1],x[...,206:].transpose(1,2)),1)
        self.local_captured.clear();u=self.context_head(features,valid);own=raw[:,:,0]
        reference=F.softplus(own[:,:,2].detach())+.025
        corrected=torch.stack((own[:,:,0]+torch.tanh(u[...,16:32]),own[:,:,1]+reference*torch.tanh(u[...,:16]/reference),own[:,:,2]),2)
        return gate+torch.tanh(u[...,32:33]),corrected.unsqueeze(2)
def make_model(stage,master,metadata):return Network(master,stage)
