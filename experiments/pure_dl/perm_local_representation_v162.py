"""Bounded depth-local feature adaptation of the original private PERM experts."""
import torch
from torch import nn
from torch.nn import functional as F
from perm_ranked_probability_v154 import ROOT,SOURCE,PARENT,Network as Previous,load_data,Evaluator,channel_objective
from compact_five_stage_v100 import CompactReadout
STUDY=ROOT/'results/deep_learning/perm_local_representation_v162_20261001'
OUT=ROOT/'outputs/pure_dl_perm_local_representation_v162_fold5'
STAGES={'perm_conditional':('PERM',256,False,0.,3e-4)}

class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce=nn.Conv1d(128,32,1)
        self.depth=nn.Conv1d(32,32,7,padding=3,groups=32)
        self.output=nn.Conv1d(32,128,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,h,valid):
        z=F.gelu(self.reduce(h)).masked_fill(~valid[:,None],0)
        u=self.output(F.gelu(self.depth(z))).float()
        bound=.1*h.detach().float().square().mean(1,keepdim=True).sqrt().clamp_min(.01)
        delta=bound*torch.tanh(u/bound)
        return (h+delta.to(h.dtype)).masked_fill(~valid[:,None],0)

class Network(Previous):
    def __init__(self,master,mode):
        super().__init__(master,mode)
        self.adapters=nn.ModuleList([Adapter(),Adapter()])
        self.adapter_valid=None
        self.adapter_handles=[m.main_head.register_forward_pre_hook(self.adapt(i),prepend=True)
                              for i,m in enumerate(self.backbone.members)]
    def adapt(self,i):
        def hook(module,args):
            # The base body intentionally runs in no_grad. Re-enable gradients
            # only for the new adapter, never for the original backbone.
            with torch.set_grad_enabled(self.training):
                h=self.adapters[i](args[0].detach(),self.adapter_valid)
            return (h,*args[1:])
        return hook
    def capture(self,e):
        def hook(module,args):self.captured[e]=args[0]
        return hook
    def capture_new(self,e):
        def hook(module,args):self.local_captured[e]=args[0]
        return hook
    def groups(self):
        return [dict(params=list(self.context_head.parameters()),lr=3e-4,name='PERM_location_weight_head'),
                dict(params=list(self.adapters.parameters()),lr=1e-4,name='PERM_depth_local_representation')]
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(same_capacity_as_v136=False,original_backbone_frozen=True,
                    bounded_hidden_residual=.1,depth_kernel=7,private_adapters=True,
                    original_gate_rows_frozen=True,gate_function_can_change_through_features=True,
                    trained_v154_head_not_loaded=True,same_initial_predictive_function=True)
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.adapter_valid=valid;self.local_captured.clear()
        try:
            gate,raw=CompactReadout.forward(self,x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
            features=torch.cat((self.local_captured[0],self.local_captured[1],x[...,206:].transpose(1,2)),1)
            self.local_captured.clear();u=self.context_head(features,valid)
            own=raw[:,:,0];reference=F.softplus(own[:,:,2].detach())+.025
            corrected=torch.stack((own[:,:,0]+torch.tanh(u[...,16:]),
                        own[:,:,1]+reference*torch.tanh(u[...,:16]/reference),own[:,:,2]),2)
            return gate,corrected.unsqueeze(2)
        finally:self.adapter_valid=None

def make_model(stage,master,metadata):return Network(master,stage)
