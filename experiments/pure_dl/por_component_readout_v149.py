"""Matched component readouts: shared point decoding versus component communication."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from por_joint_density_width_v140 import ROOT,SOURCE,PARENT,ContextHead,Network as Parent,load_data,Evaluator,channel_objective

STUDY=ROOT/'results/deep_learning/por_component_readout_v149_20261001'
OUT=ROOT/'outputs/pure_dl_por_component_readout_v149_fold5'
STAGES={s:('POR',256,False,0.,3e-4) for s in ('shared','communicate')}

class ComponentHead(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode
        self.body=ContextHead();self.body.output=nn.Identity()
        self.project=nn.Linear(256,256)
        self.identity=nn.Parameter(torch.randn(16,16)*.02)
        self.old_distribution=nn.Linear(4,16)
        self.norm=nn.LayerNorm(16);self.qkv=nn.Linear(16,48);self.mix=nn.Linear(16,16)
        self.decode=nn.Sequential(nn.LayerNorm(16),nn.Linear(16,32),nn.GELU(),nn.Linear(32,2))
        nn.init.zeros_(self.decode[-1].weight);nn.init.zeros_(self.decode[-1].bias)

    def forward(self,x,valid,raw):
        h=self.body(x,valid);tokens=self.project(h).reshape(*h.shape[:2],16,16)
        sd=F.softplus(raw[:,:,2])+.025
        family=torch.cat((torch.ones_like(sd[...,:8]),-torch.ones_like(sd[...,8:])),dim=-1)
        distribution=torch.stack((raw[:,:,0].log_softmax(-1).clamp_min(-12),raw[:,:,1],sd.log(),family),-1).detach()
        tokens=tokens+self.identity+self.old_distribution(distribution)
        q,k,v=self.qkv(self.norm(tokens)).chunk(3,-1)
        logits=(q@k.transpose(-1,-2))/math.sqrt(16)
        if self.mode=='shared':
            allowed=torch.eye(16,device=logits.device,dtype=torch.bool)
            logits=logits.masked_fill(~allowed,float('-inf'))
        tokens=tokens+self.mix(logits.softmax(-1)@v)
        output=self.decode(tokens).float()
        return torch.cat((output[...,0],output[...,1]),-1).masked_fill(~valid[...,None],0)

class Network(Parent):
    def __init__(self,master,mode):
        super().__init__(master,mode);self.context_head=ComponentHead(mode)
    def groups(self):return [dict(params=list(self.context_head.parameters()),lr=3e-4,name='component_shared_density_readout')]
    def initialize_priors(self,data):
        return dict(parent_checkpoint=str(PARENT),old_predictor_fully_frozen=True,same_initial_predictive_function=True,
            component_tokens=16,token_width=16,hidden_width=256,mode=self.context_head.mode,
            matched_capacity_and_initialization_across_modes=True,only_attention_connectivity_differs=True,
            new_component_locations_and_logweights=True,location_reference_width_bound=1.,logweight_bound=1.,
            gate_scales_families_frozen=True,no_trained_v140_head_loaded=True,executed_training_stages_including_source=5)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        from compact_por_mixed_family_v113 import Network as Frozen
        self.local_captured.clear()
        with torch.no_grad():gate,raw=Frozen.forward(self,x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
        features=torch.cat((self.local_captured[0],self.local_captured[1],x[...,206:].transpose(1,2)),1)
        self.local_captured.clear();own=raw[:,:,0]
        u=self.context_head(features,valid,own);reference=F.softplus(own[:,:,2].detach())+.025
        location=own[:,:,1]+reference*torch.tanh(u[...,:16]/reference)
        weights=own[:,:,0]+torch.tanh(u[...,16:])
        return gate,torch.stack((weights,location,own[:,:,2]),2).unsqueeze(2)

def make_model(stage,master,metadata):return Network(master,stage)
