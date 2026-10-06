"""Small additive local curve-core patch; keep the effective154 readout intact."""
import torch
from torch import nn
from torch.nn import functional as F
from perm_ranked_probability_v154 import ROOT,SOURCE,PARENT,Network as Parent,load_data,Evaluator,channel_objective
from compact_five_stage_v100 import CompactReadout
STUDY=ROOT/'results/deep_learning/perm_local_core_patch_v196_20261001'
OUT=ROOT/'outputs/pure_dl_perm_local_core_patch_v196_fold5'
STAGES={k:('PERM',256,False,0.,3e-4) for k in ('mean','weighted')}
CURVES=('GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT')

class SmallCore(nn.Module):
    def __init__(self,indices,mode):
        super().__init__();self.mode=mode;self.register_buffer('indices',torch.tensor(indices,dtype=torch.long))
        self.embed=nn.Linear(16,16);self.curve_id=nn.Parameter(torch.zeros(9,16))
        self.gen1=nn.Linear(16,16);self.gen2=nn.Linear(16,8)
        self.gen3=nn.Linear(24,16);self.gen4=nn.Linear(16,16)
        self.project=nn.Linear(144,128)
        nn.init.zeros_(self.project.weight);nn.init.zeros_(self.project.bias)
    def forward(self,x,valid):
        b,l,_=x.shape;assert l%16==0
        patches=x.index_select(-1,self.indices).masked_fill(~valid[...,None],0).reshape(b,l//16,16,9).transpose(-1,-2)
        token=F.gelu(self.embed(patches))+self.curve_id
        values=self.gen2(F.gelu(self.gen1(token)))
        core=(values.softmax(-2)*values).sum(-2,keepdim=True) if self.mode=='weighted' else values.mean(-2,keepdim=True)
        updated=self.gen4(F.gelu(self.gen3(torch.cat((token,core.expand(-1,-1,9,-1)),-1))))
        residual=.1*self.project(updated.flatten(-2))
        residual=residual.masked_fill(~valid.reshape(b,l//16,16).any(-1)[...,None],0)
        return F.interpolate(residual.transpose(1,2),size=l,mode='linear',align_corners=False).masked_fill(~valid[:,None],0)

class AugmentedHead(nn.Module):
    def __init__(self,original,indices,mode):
        super().__init__()
        for name in ('stem','norm','depth','expand','output'):setattr(self,name,getattr(original,name))
        self.patch=SmallCore(indices,mode)
    def forward(self,features,valid,x):
        h=F.gelu(self.stem(features)).masked_fill(~valid[:,None],0)
        h=(h+self.patch(x,valid)).masked_fill(~valid[:,None],0)
        z=self.norm(h.transpose(1,2)).transpose(1,2).masked_fill(~valid[:,None],0)
        h=(h+F.gelu(F.conv1d(z,self.depth.weight.sum(-1,keepdim=True),self.depth.bias,groups=128))).masked_fill(~valid[:,None],0)
        a,b=self.expand(h).chunk(2,1)
        return self.output(F.gelu(a)*b.sigmoid()).transpose(1,2).float().masked_fill(~valid[...,None],0)

class Network(Parent):
    def __init__(self,master,mode):
        super().__init__(master,mode);self.study_mode=mode
        names=list(master['selected_feature_names']);indices=[names.index(k) for k in CURVES]
        assert len(names)==206 and len(set(indices))==9
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010);self.context_head=AugmentedHead(self.context_head,indices,mode)
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(additive_local_curve_core=self.study_mode,curve_names=list(CURVES),curve_indices=self.context_head.patch.indices.tolist(),
                    patch_length=16,series_width=16,core_width=8,residual_scale=.1,zero_initialized_projection=True,
                    extra_active_parameters=sum(p.numel() for p in self.context_head.patch.parameters()),
                    original154_trunk_locations_logweights_NLL_jointRPS_retained=True,
                    SOURCE_backbone_gate_scale_frozen=True,full_resolution_point_head_preserved=True,
                    same_initial_base_parameter_tensors=True,same_initial_hidden_and_predictive_function=True,
                    no_stochastic_pooling=True,no_trained_v154_head_loaded=True,not_exact_SOFTS_reproduction=True)
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.local_captured.clear()
        with torch.no_grad():
            gate,raw=CompactReadout.forward(self,x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
        features=torch.cat((self.local_captured[0],self.local_captured[1],x[...,206:].transpose(1,2)),1)
        self.local_captured.clear();u=self.context_head(features,valid,x)
        own=raw[:,:,0];reference=F.softplus(own[:,:,2].detach())+.025
        corrected=torch.stack((own[:,:,0]+u[...,16:].tanh(),own[:,:,1]+reference*torch.tanh(u[...,:16]/reference),own[:,:,2]),2)
        return gate,corrected.unsqueeze(2)

def make_model(stage,master,metadata):return Network(master,stage)
