"""Five total stages: source, combined POR, combined PERM, SW joint, SW density.

Only the original dual source is reused; no trained downstream weights enter
the POR/PERM constructions or the randomly initialized final SW architecture.
"""
from pathlib import Path
import copy,types
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from train_dual8222_heads80 import SOURCE
from paired_local_context_router_sequence import Network as Original,retained_indices,calibration_indices
from paired_detached_router_sequence import routed_merge
from dual8222_location_readout_v5 import BoundedLocationResidual
from divergent_models_v45 import Network as RandomSW
from well_conditioned_v46 import WellEncoder,Network as Conditioned
from sw_interaction_v56 import InteractionFF
from sw_dropout_rate_v76 import attention_forward

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/compact_five_stage_v100_20260930'
OUT=ROOT/'outputs/pure_dl_compact_v100_fold5'
STAGES=dict(por=('POR',256,False,0.,3e-5),perm=('PERM',256,False,0.,3e-5),
            sw_joint=('SW',1024,True,1.,3e-4),sw_density=('SW',1024,True,0.,1e-4))

class BoundedSide(nn.Module):
    def __init__(self,outputs):
        super().__init__();self.input=nn.Conv1d(191,32,1);self.output=nn.Conv1d(32,outputs,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,h,side,valid,bound):
        x=torch.cat((h,side.transpose(1,2).to(h.dtype)),1).masked_fill(~valid[:,None],0)
        raw=self.output(F.gelu(self.input(x))).float()
        return (bound*torch.tanh(raw/bound)).masked_fill(~valid[:,None],0)

class Route(nn.Module):
    def __init__(self,bound,seed):
        super().__init__();self.bound=bound
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);self.input=nn.Conv1d(319,32,1);self.output=nn.Conv1d(32,1,1)
            nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,h,side,valid):
        x=torch.cat((*h,side.transpose(1,2).to(h[0].dtype)),1).masked_fill(~valid[:,None],0)
        raw=self.output(F.gelu(self.input(x))).squeeze(1).float()
        return (self.bound*torch.tanh(raw/self.bound)).masked_fill(~valid,0)

def shift(logits,delta):return logits.float()+torch.stack((.5*delta,-.5*delta),-1).unsqueeze(2)

class PackedHead(nn.Module):
    def __init__(self,original,c):
        super().__init__();self.c=c
        for stem,value,lo,hi in [('gate_weight',original.gate.weight,c,c+1),('gate_bias',original.gate.bias,c,c+1),
                ('density_weight',original.density.weight,24*c,24*(c+1)),('density_bias',original.density.bias,24*c,24*(c+1))]:
            self.register_parameter(stem,nn.Parameter(value[lo:hi].detach().clone(),requires_grad=(c==0 and stem.startswith('density')) or (c==1 and stem.startswith('gate'))))
            self.register_buffer(stem+'_before',value[:lo].detach().clone());self.register_buffer(stem+'_after',value[hi:].detach().clone())
        self.point=BoundedLocationResidual(False) if c==0 else None
        self.side=BoundedSide(8 if c==0 else 1)
    def packed(self,stem):return torch.cat((getattr(self,stem+'_before'),getattr(self,stem),getattr(self,stem+'_after')),0)
    def forward(self,h,side,valid):
        gate=F.conv1d(h,self.packed('gate_weight'),self.packed('gate_bias')).transpose(1,2)[:,:,self.c:self.c+1]
        raw=F.conv1d(h,self.packed('density_weight'),self.packed('density_bias')).transpose(1,2).reshape(len(h),-1,3,3,8)[:,:,self.c:self.c+1].float()
        if self.c==0:
            scale=raw[:,:,0,2].transpose(1,2)
            delta=self.point(h,valid,scale)+self.side(h,side,valid,.25*(F.softplus(scale.detach())+.025))
            own=raw[:,:,0];raw=torch.stack((own[:,:,0],own[:,:,1]+delta.transpose(1,2),own[:,:,2]),2).unsqueeze(2)
        else:gate=gate.float()+self.side(h,side,valid,.25).transpose(1,2)
        return gate,raw

class CompactReadout(nn.Module):
    def __init__(self,master,c):
        super().__init__();self.c=c
        numeric,condition=calibration_indices(master['selected_feature_names'])
        self.backbone=Original(retain_indices=retained_indices(master['selected_feature_names']),numeric_indices=numeric,conditioning_indices=condition)
        self.backbone.load_state_dict(master['state_dict'],strict=True);self.backbone.eval().requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010);self.heads=nn.ModuleList([PackedHead(m.main_head,c) for m in self.backbone.members])
        self.routes=nn.ModuleList([Route(.25,20261104),Route(.125,20261105),Route(.125,20261106)])
        self.captured={};self.handles=[m.main_head.register_forward_pre_hook(self.capture(e)) for e,m in enumerate(self.backbone.members)]
    def capture(self,e):
        def hook(module,args):self.captured[e]=args[0].detach()
        return hook
    def train(self,mode=True):super().train(mode);self.backbone.eval();return self
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.captured.clear()
        with torch.no_grad():
            for m in self.backbone.members:m(x[...,:206].contiguous(),relative,valid,absolute_depth=absolute_depth)
            logits=self.backbone.route_logits(x[...,:206].contiguous(),valid)[:,:,self.c:self.c+1]
        hidden=[self.captured[e] for e in range(2)];side=x[...,206:]
        outputs=[head(h,side,valid) for head,h in zip(self.heads,hidden)]
        common,a,b=[route(hidden,side,valid) for route in self.routes]
        if self.c==0:atom=shift(logits,a);density=shift(logits,common+b)
        else:atom=density=shift(logits,common+(a+b)*.5)
        gate,_=routed_merge(outputs,atom);_,raw=routed_merge(outputs,density)
        self.captured.clear();return gate,raw
    def groups(self):
        linear=[p for h in self.heads for n,p in h.named_parameters() if p.requires_grad and n in ('gate_weight','gate_bias','density_weight','density_bias')]
        nonlinear=[p for h in self.heads for n,p in h.named_parameters() if p.requires_grad and n not in ('gate_weight','gate_bias','density_weight','density_bias')]+list(self.routes.parameters())
        return [dict(params=linear,lr=3e-5,name='original_target_rows'),dict(params=nonlinear,lr=3e-4,name='combined_new_heads_routes')]

class CompactSW(RandomSW):
    def __init__(self,metadata):
        super().__init__('long_section',metadata);self.mode='well';self.study_mode='compact_joint'
        self.condition=WellEncoder()
        self.register_buffer('reference_inputs',metadata['reference_inputs'].detach().clone(),persistent=False)
        for block in self.body.global_blocks:
            block.ff=InteractionFF('additive');block.use_key_dropout=True;block.drop_probability=.4
            block.forward=types.MethodType(attention_forward,block)
    forward=Conditioned.forward

class DensityStage(nn.Module):
    def __init__(self,metadata,state):
        super().__init__();self.c=2
        self.atom=CompactSW(metadata);self.atom.load_state_dict(state,strict=True);self.atom.requires_grad_(False).eval()
        self.density=CompactSW(metadata);self.density.load_state_dict(state,strict=True)
    def train(self,mode=True):super().train(mode);self.atom.eval();return self
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        with torch.no_grad():gate,_=self.atom(x,relative,valid,well_inputs=well_inputs)
        _,raw=self.density(x,relative,valid,well_inputs=well_inputs)
        return gate,raw

def load_data(stage):
    from champion_scaling_v21 import load_data as original_data
    from fullcv_context_v97 import add_well_bank
    data,master=original_data()
    train_wells=np.unique(data['wells'][data['fold']!=4]).tolist();heldout=np.unique(data['wells'][data['fold']==4]).tolist()
    data['population_rows']=dict(train_wells=train_wells,heldout_wells=heldout)
    metadata={k:master[k] for k in ('centers','density_edges','selected_feature_names')};metadata.update(add_well_bank(data))
    metadata.update(sw_target_median=float(master['regression_median'][2]),sw_target_scale=float(master['regression_scale'][2]))
    if stage in ('por','perm'):
        with np.load(STUDY/'amplitude.npz') as z:side=torch.as_tensor(z['amplitude'],device='cuda')
        data['x']=torch.cat((data['x'],side),-1)
    return data,master,metadata

def make_model(stage,master,metadata):
    if stage in ('por','perm'):return CompactReadout(master,0 if stage=='por' else 1)
    if stage=='sw_joint':return CompactSW(metadata)
    if stage=='sw_density':
        saved=torch.load(OUT/'sw_joint/model_best.pt',map_location='cpu',weights_only=False)
        return DensityStage(metadata,saved['state_dict'])
    raise ValueError(stage)
