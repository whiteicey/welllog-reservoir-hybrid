"""Matched within-patch order ablation with two fixed positional moments."""
import hashlib
import numpy as np
import torch
from torch import nn
from sw_interaction_v56 import ROOT,Network as Parent,Evaluator,load_data
from divergent_models_v45 import Network as PriorNetwork
from well_conditioned_v46 import modulate
from torch.nn import functional as F

STUDY=ROOT/'results/deep_learning/sw_patch_moments_v73_20260929'
PREVIOUS=ROOT/'results/deep_learning/perm_observation_v72_20260929'
SOURCE=ROOT/'outputs/pure_dl_sw_interaction_v56_SW_additive_fold5/model_best.pt'
MODES={m:dict(target='SW',c=2,length=1024) for m in ('permuted','ordered')}
PERM=(5,12,0,9,3,15,7,1,14,4,10,6,2,13,8,11)

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

class Network(Parent):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('additive',metadata)
        saved=torch.load(SOURCE,map_location='cpu',weights_only=False)
        self.load_state_dict(saved['state_dict'],strict=True)
        self.study_mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261173)
            self.patch_project=nn.Linear(192,96,bias=False)
            nn.init.zeros_(self.patch_project.weight)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.parent_info=dict(checkpoint=str(SOURCE.relative_to(ROOT)),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            best_epoch=saved['best_epoch'],target_score=saved['target_score'],
            initialization='v56 full additive backbone and FiLM; output reset to training-only SW prior; zero new patch projection',
            optimizer_reset=True,all_network_trainable=True,patch_project_parameters=18432,permutation=list(PERM))

    def initialize_priors(self,data):
        return PriorNetwork.initialize_priors(self,data)|self.parent_info

    def patch_summary(self,h,pv):
        b,t,s=pv.shape;z=h.reshape(b,t,s,96).float();v=pv
        if self.study_mode=='permuted':
            ix=torch.tensor(PERM,device=h.device);z=z.index_select(2,ix);v=v.index_select(2,ix)
        pos=torch.linspace(-1,1,16,device=h.device)
        basis=torch.stack((pos,.5*(3*pos.square()-1)),-1)
        count=v.sum(-1).clamp_min(1)[...,None,None]
        centered=basis[None,None]-(basis[None,None]*v[...,None]).sum(2,keepdim=True)/count
        moments=torch.einsum('btsc,btsk->btkc',z,centered*v[...,None])/count
        return self.patch_project(moments.flatten(-2).to(h.dtype))

    def forward(self,x,relative,valid,absolute_depth=None,return_aux=False,generator=None,well_inputs=None):
        if well_inputs is None:raise ValueError('Explicit input-only well bank required')
        b,l,_=x.shape;c=self.body;modulation=self.condition(well_inputs)
        h=F.gelu(c.stem(torch.cat((x,relative[...,None]),-1))).masked_fill(~valid[...,None],0)
        h=modulate(h,modulation[0],valid)
        for block in c.local:h=block(h,valid)
        pv=valid.reshape(b,l//16,16);n=pv.sum(-1).clamp_min(1)
        token=(h.reshape(b,l//16,16,96)*pv[...,None]).sum(2)/n[...,None]
        token=token+self.patch_summary(h,pv)
        pos=torch.linspace(-1,1,l,device=x.device).expand(b,-1)
        loc=torch.stack((relative,pos),-1).reshape(b,l//16,16,2)
        loc=(loc*pv[...,None]).sum(2)/n[...,None]
        tv=pv.any(-1);token=(token+c.token_pos(loc)).masked_fill(~tv[...,None],0)
        for j,block in enumerate(c.global_blocks):token=modulate(block(token,tv),modulation[j+1],tv)
        global_h=F.interpolate(token.transpose(1,2),size=l,mode='linear',align_corners=False).transpose(1,2)
        h=F.gelu(c.fuse(torch.cat((h,global_h),-1))).masked_fill(~valid[...,None],0)
        h=modulate(h,modulation[5],valid)
        for block in c.detail:h=block(h,valid)
        y=self.output(self.norm(h)).float();output=(y[...,:1],y[...,1:].reshape(*y.shape[:2],1,3,8))
        return (output,y.new_zeros(()),None) if return_aux else output
