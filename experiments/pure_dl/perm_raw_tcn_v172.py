"""New private raw-feature TCN encoder for PERM residual distribution; SOURCE fixed."""
import torch
from torch import nn
from torch.nn import functional as F
from perm_ranked_probability_v154 import ROOT,SOURCE,PARENT,Network as Parent,load_data,Evaluator,channel_objective
STUDY=ROOT/'results/deep_learning/perm_raw_tcn_v172_20261001'
OUT=ROOT/'outputs/pure_dl_perm_raw_tcn_v172_fold5'
STAGES={'perm_conditional':('PERM',256,False,0.,3e-4)}
class RawEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.stem=nn.Conv1d(269,64,1)
        self.depth=nn.ModuleList([nn.Conv1d(64,64,5,padding=2*d,dilation=d,groups=64) for d in (1,2,4,8)])
        self.mix=nn.ModuleList([nn.Conv1d(64,128,1) for _ in range(4)])
        self.output=nn.Conv1d(64,32,1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,x,valid):
        h=F.gelu(self.stem(x)).masked_fill(~valid[:,None],0)
        for depth,mix in zip(self.depth,self.mix):
            z=F.gelu(depth(h)).masked_fill(~valid[:,None],0);a,b=mix(z).chunk(2,1)
            h=(h+.25*F.gelu(a)*b.sigmoid()).masked_fill(~valid[:,None],0)
        return self.output(h).transpose(1,2).float().masked_fill(~valid[...,None],0)
class Network(Parent):
    def __init__(self,master,mode):
        super().__init__(master,mode);self.context_head=RawEncoder()
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(new_raw_feature_encoder=True,original_backbone_retained_as_frozen_distribution_prior=True,
                    input_hidden=0,input_raw=269,width=64,depth_kernel=5,stored_kernel=5,effective_point_kernel_is_sum=False,dilations=[1,2,4,8],same_capacity_as_v136=False,
                    no_trained_v154_head_loaded=True,no_extra_training_phase=True)
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.local_captured.clear()
        with torch.no_grad():
            # Skip the parent residual head; use its frozen SOURCE predictor only.
            from compact_five_stage_v100 import CompactReadout
            gate,raw=CompactReadout.forward(self,x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
        self.local_captured.clear();u=self.context_head(x.transpose(1,2),valid);own=raw[:,:,0]
        reference=F.softplus(own[:,:,2].detach())+.025
        corrected=torch.stack((own[:,:,0]+torch.tanh(u[...,16:]),own[:,:,1]+reference*torch.tanh(u[...,:16]/reference),own[:,:,2]),2)
        return gate,corrected.unsqueeze(2)
def make_model(stage,master,metadata):return Network(master,stage)
