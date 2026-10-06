"""Matched coarse versus local queries for neural depth upsampling."""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sw_interaction_v56 import ROOT,Network as Parent,Evaluator,load_data
from well_conditioned_v46 import modulate

STUDY=ROOT/'results/deep_learning/sw_upsample_v61_20260929'
PREVIOUS=ROOT/'results/deep_learning/sw_brier_v60_20260929'
SOURCE=ROOT/'outputs/pure_dl_sw_interaction_v56_SW_additive_fold5/model_best.pt'
PARENT_PRED=SOURCE.with_name('channel_pred.npy')
MODES={m:dict(target='SW',c=2,length=1024) for m in ('coarse_query','local_query')}

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

class LearnedUpsample(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode
        self.query=nn.Linear(96,16);self.key=nn.Linear(96,16)
        nn.init.zeros_(self.query.weight);nn.init.zeros_(self.query.bias)
    def forward(self,local,token,global_h):
        b,l,_=local.shape;t=token.shape[1]
        # Same align_corners=False coordinates as original linear interpolation.
        pos=((torch.arange(l,device=local.device,dtype=torch.float32)+.5)*t/l-.5).clamp(0,t-1)
        left=pos.floor().long();right=(left+1).clamp_max(t-1);fraction=pos-left
        prior=torch.stack((1-fraction,fraction),-1)
        logits=prior.clamp_min(1e-12).log();base=logits.softmax(-1)
        query_input=local if self.mode=='local_query' else global_h
        q=self.query(F.layer_norm(query_input,(96,)))
        keys=self.key(F.layer_norm(token,(96,)))
        score=torch.stack(((q*keys[:,left]).sum(-1),(q*keys[:,right]).sum(-1)),-1).float()/4
        delta=(logits[None]+score).softmax(-1)-base[None]
        # At zero query this is identically zero, preserving parent BF16 outputs.
        correction=delta[...,:1]*token[:,left].float()+delta[...,1:]*token[:,right].float()
        return global_h+correction.to(global_h.dtype)

class Network(Parent):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('additive',metadata)
        saved=torch.load(SOURCE,map_location='cpu',weights_only=False)
        self.load_state_dict(saved['state_dict'],strict=True);self.study_mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261161);self.upsample=LearnedUpsample(mode)
        self.initialization_info=dict(checkpoint=str(SOURCE.relative_to(ROOT)),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            best_epoch=saved['best_epoch'],target_score=saved['target_score'],optimizer_reset=True,
            initialization='exact v56 additive full function; zero query leaves linear interpolation unchanged',
            new_parameters=3104,query_source=mode,all_parameters_trainable=True)
    def initialize_priors(self,data):return self.initialization_info
    def core_hash(self):
        return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for n,v in self.state_dict().items() if not n.startswith('upsample.'))).hexdigest()

    def forward(self,x,relative,valid,absolute_depth=None,return_aux=False,generator=None,well_inputs=None):
        if well_inputs is None:raise ValueError('Explicit input-only well bank required')
        b,l,_=x.shape;c=self.body
        context=well_inputs if self.mode=='well' else self.reference_inputs[None].expand(b,-1,-1)
        modulation=self.condition(context)
        h=F.gelu(c.stem(torch.cat((x,relative[...,None]),-1))).masked_fill(~valid[...,None],0)
        h=modulate(h,modulation[0],valid)
        for block in c.local:h=block(h,valid)
        pv=valid.reshape(b,l//16,16);n=pv.sum(-1).clamp_min(1)
        token=(h.reshape(b,l//16,16,96)*pv[...,None]).sum(2)/n[...,None]
        pos=torch.linspace(-1,1,l,device=x.device).expand(b,-1)
        loc=torch.stack((relative,pos),-1).reshape(b,l//16,16,2)
        loc=(loc*pv[...,None]).sum(2)/n[...,None]
        tv=pv.any(-1);token=(token+c.token_pos(loc)).masked_fill(~tv[...,None],0)
        for j,block in enumerate(c.global_blocks):token=modulate(block(token,tv),modulation[j+1],tv)
        global_h=F.interpolate(token.transpose(1,2),size=l,mode='linear',align_corners=False).transpose(1,2)
        global_h=self.upsample(h,token,global_h)
        h=F.gelu(c.fuse(torch.cat((h,global_h),-1))).masked_fill(~valid[...,None],0)
        h=modulate(h,modulation[5],valid)
        for block in c.detail:h=block(h,valid)
        y=self.output(self.norm(h)).float();output=(y[...,:1],y[...,1:].reshape(*y.shape[:2],1,3,8))
        return (output,y.new_zeros(()),None) if return_aux else output
