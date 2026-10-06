"""Task-specific input-only whole-well memory reads on a shared trainable backbone."""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from well_conditioned_v46 import ROOT,Network as Parent,Evaluator,load_data,modulate

STUDY=ROOT/'results/deep_learning/well_memory_v47_20260929'
PREVIOUS=ROOT/'results/deep_learning/well_conditioned_v46_20260929'
PARENT=ROOT/'outputs/pure_dl_well_conditioned_v46_SW_well_fold5/model_best.pt'
MODES={m:dict(target='SW',c=2,length=1024) for m in ('global_query','section_query')}

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

class TaskMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(207,96),nn.GELU(),nn.Linear(96,64),nn.LayerNorm(64))
        self.key=nn.Linear(64,64);self.value=nn.Linear(64,64)
        self.queries=nn.ModuleList([nn.Sequential(nn.LayerNorm(96),nn.Linear(96,64)) for _ in range(2)])
        self.projections=nn.ModuleList([nn.Linear(64,128) for _ in range(2)])
        for projection in self.projections:
            nn.init.zeros_(projection.weight);nn.init.zeros_(projection.bias)

    def forward(self,token,tv,bank,mode,length):
        b,n,_=bank.shape
        # Deterministic depth rank, not well identity or a target-derived feature.
        depth=((torch.arange(n,device=bank.device,dtype=torch.float32)+.5)/n*2-1)[None,:,None].expand(b,-1,-1)
        memory=self.encoder(torch.cat((bank,depth),-1))
        key=self.key(memory).reshape(b,n,4,16).transpose(1,2)
        value=self.value(memory).reshape(b,n,4,16).transpose(1,2)
        if mode=='global_query':
            token=((token*tv[...,None]).sum(1)/tv.sum(1).clamp_min(1)[:,None])[:,None].expand_as(token)
        result=[]
        for query,projection in zip(self.queries,self.projections):
            q=query(token).reshape(b,-1,4,16).transpose(1,2)
            read=F.scaled_dot_product_attention(q,key,value,dropout_p=0.)
            read=read.transpose(1,2).reshape(b,-1,64)
            delta=projection(read).masked_fill(~tv[...,None],0)
            result.append(F.interpolate(delta.transpose(1,2),size=length,mode='linear',align_corners=False).transpose(1,2))
        return result

class Network(Parent):
    def __init__(self,mode,metadata):
        super().__init__('well',metadata)
        saved=torch.load(PARENT,map_location='cpu',weights_only=False)
        self.load_state_dict(saved['state_dict'],strict=True)
        self.mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261147);self.memory=TaskMemory()
        self.parent_info=dict(checkpoint=str(PARENT.relative_to(ROOT)),sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),
            best_epoch=saved['best_epoch'],target_score=saved['target_score'],optimizer_reset=True,
            initialization='v46 current-well network plus identical zero-projection task memory; all parameters trainable')

    def forward(self,x,relative,valid,absolute_depth=None,return_aux=False,generator=None,well_inputs=None):
        if well_inputs is None:raise ValueError('Input-only current-well bank required')
        b,l,_=x.shape;c=self.body;modulation=self.condition(well_inputs)
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
        h=F.gelu(c.fuse(torch.cat((h,global_h),-1))).masked_fill(~valid[...,None],0)
        h=modulate(h,modulation[5],valid)
        for block in c.detail:h=block(h,valid)
        gate_delta,density_delta=self.memory(token,tv,well_inputs,self.mode,l)
        # Full linear heads keep the original GEMM shape at zero initialization.
        gate=self.output(self.norm(h+gate_delta)).float()[...,:1]
        density=self.output(self.norm(h+density_delta)).float()[...,1:].reshape(b,l,1,3,8)
        output=(gate,density)
        return (output,gate.new_zeros(()),None) if return_aux else output
