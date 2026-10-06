"""Matched learned pooling for private PERM: positional versus content scores."""
import torch
from torch import nn
from torch.nn import functional as F
from perm_transfer_v50 import ROOT,Network as Parent,Evaluator,load_data
from well_conditioned_v46 import modulate
import numpy as np

STUDY=ROOT/'results/deep_learning/perm_token_pool_v52_20260929'
PREVIOUS=ROOT/'results/deep_learning/perm_moment_mix_v51_20260929'
MODES={m:dict(target='PERM',c=1,length=1024) for m in ('position','content')}

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

class Network(Parent):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('sw_transfer',metadata)
        self.pool_kind=mode
        self.pool_score=nn.Linear(96,4,bias=False);nn.init.zeros_(self.pool_score.weight)
        self.initialization_info.update(pool_kind=mode,pooling_heads=4,points_per_token=16,
            pool_initialization='zero scores => masked mean in each24-channel group',
            comparison='same SW source/reset PERM head/all trainable; scoring input is content or nonlinear position encoding')

    def forward(self,x,relative,valid,absolute_depth=None,return_aux=False,generator=None,well_inputs=None):
        if well_inputs is None:raise ValueError('Input-only current-well bank required')
        b,l,_=x.shape;c=self.body;modulation=self.condition(well_inputs)
        h=F.gelu(c.stem(torch.cat((x,relative[...,None]),-1))).masked_fill(~valid[...,None],0)
        h=modulate(h,modulation[0],valid)
        for block in c.local:h=block(h,valid)
        pv=valid.reshape(b,l//16,16);n=pv.sum(-1).clamp_min(1);tv=pv.any(-1)
        pos=torch.linspace(-1,1,l,device=x.device).expand(b,-1)
        point_position=torch.stack((relative,pos),-1)
        score_input=h if self.pool_kind=='content' else F.gelu(c.token_pos(point_position))
        logits=self.pool_score(score_input).float().reshape(b,l//16,16,4)
        weights=logits.masked_fill(~pv[...,None],-1e30).softmax(2)*pv[...,None]
        # Fully padded tokens remain exactly zero without a NaN softmax row.
        weights=weights/weights.sum(2,keepdim=True).clamp_min(1e-12)
        grouped=h.reshape(b,l//16,16,4,24)
        token=(grouped.float()*weights[...,None]).sum(2).reshape(b,l//16,96).to(h.dtype)
        loc=point_position.reshape(b,l//16,16,2);loc=(loc*pv[...,None]).sum(2)/n[...,None]
        token=(token+c.token_pos(loc)).masked_fill(~tv[...,None],0)
        for j,block in enumerate(c.global_blocks):token=modulate(block(token,tv),modulation[j+1],tv)
        global_h=F.interpolate(token.transpose(1,2),size=l,mode='linear',align_corners=False).transpose(1,2)
        h=F.gelu(c.fuse(torch.cat((h,global_h),-1))).masked_fill(~valid[...,None],0)
        h=modulate(h,modulation[5],valid)
        for block in c.detail:h=block(h,valid)
        y=self.output(self.norm(h)).float();output=(y[...,:1],y[...,1:].reshape(b,l,1,3,8))
        info=None
        if return_aux:
            entropy=-(weights*weights.clamp_min(1e-12).log()).sum(2)
            info=dict(pool_kind=self.pool_kind,mean_valid_head_entropy=float(entropy[tv].mean()),max_pool_weight=float(weights.max()))
        return (output,y.new_zeros(()),info) if return_aux else output
