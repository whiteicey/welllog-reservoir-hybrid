"""Matched SW attention parametrizations retaining positive local depth filter."""
import math,types
import torch
from torch import nn
from torch.nn import functional as F
from compact_five_stage_v100 import ROOT,SOURCE,load_data as original_data
from compact_frequency_v104 import Network as LocalNetwork

STUDY=ROOT/'results/deep_learning/compact_attention_v107_20261001'
OUT=ROOT/'outputs/pure_dl_compact_attention_v107_fold5'
STAGES={name:('SW',1024,True,1.,3e-4) for name in ('dot','cosine')}

def attention_forward(self,x,valid):
    b,t,c=x.shape
    q,k,v=self.qkv(self.norm(x)).reshape(b,t,3,self.heads,c//self.heads).permute(2,0,3,1,4).unbind(0)
    if self.attention_kind=='cosine':
        q=F.normalize(q.float(),dim=-1).to(q.dtype);k=F.normalize(k.float(),dim=-1).to(k.dtype)
    temperature=self.log_temperature.exp().clamp(.01,100.).to(q.dtype)[None,:,None,None]
    q=q*temperature
    keys=valid
    if self.training:
        proposed=valid&(torch.rand(valid.shape,device=x.device)>=self.drop_probability)
        keys=torch.where(proposed.any(-1,keepdim=True),proposed,valid)
    z=F.scaled_dot_product_attention(q,k,v,attn_mask=keys[:,None,None,:],dropout_p=0.,scale=1.)
    x=(x+self.project(z.transpose(1,2).reshape(b,t,c))).masked_fill(~valid[...,None],0)
    return (x+self.ff(self.ffnorm(x))).masked_fill(~valid[...,None],0)

class Network(LocalNetwork):
    def __init__(self,metadata,mode):
        super().__init__(metadata,'local')
        for block in self.body.global_blocks:
            block.attention_kind=mode
            initial=10. if mode=='cosine' else 1/math.sqrt(block.width//block.heads)
            block.log_temperature=nn.Parameter(torch.full((block.heads,),math.log(initial)))
            block.forward=types.MethodType(attention_forward,block)

def load_data(stage):return original_data('sw_joint')
def make_model(stage,master,metadata):return Network(metadata,stage)
