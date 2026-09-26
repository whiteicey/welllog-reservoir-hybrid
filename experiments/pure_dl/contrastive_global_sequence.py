"""Residual contrast of global attention maps; an original inspired adaptation."""
import torch
from torch import nn
from global_cosine_sequence import Network as Baseline
from cosine_patch_sequence import SmallAttention,cosine_scores


class ContrastiveAttention(SmallAttention):
    def __init__(self,width=32,heads=4,maximum_tokens=16):
        super().__init__(width,heads,maximum_tokens)
        self.contrast_qk=nn.Linear(width,2*width,bias=False)
        self.contrast_alpha=nn.Parameter(torch.zeros(heads))

    def attention_maps(self,x,valid):
        batch,length,width=x.shape;assert length<=self.maximum_tokens
        z=self.norm(x).masked_fill(~valid[...,None],0)
        q,k,v=self.qkv(z).reshape(batch,length,3,self.heads,width//self.heads).permute(2,0,3,1,4)
        q2,k2=self.contrast_qk(z).reshape(batch,length,2,self.heads,width//self.heads).permute(2,0,3,1,4)
        pos=torch.arange(length,device=x.device)
        bias=self.relative_bias[:,pos[:,None]-pos[None,:]+self.maximum_tokens-1]
        a1=(cosine_scores(q,k,self.log_scale)+bias).masked_fill(~valid[:,None,None,:],-1e4).softmax(-1)
        a2=(cosine_scores(q2,k2,self.log_scale)+bias).masked_fill(~valid[:,None,None,:],-1e4).softmax(-1)
        coefficient=.5*self.contrast_alpha.tanh()[None,:,None,None]
        # May be signed; these are attention coefficients, not predictive probabilities.
        mixed=a1+coefficient*(a1-a2)
        return mixed,v,a1,a2

    def forward(self,x,valid):
        batch,length,width=x.shape
        mixed,v,_,_=self.attention_maps(x,valid)
        attended=(mixed.to(v.dtype)@v).transpose(1,2).reshape(batch,length,width)
        return self.project(attended).masked_fill(~valid[...,None],0)


class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        old=self.stem.hierarchy.global_patches
        with torch.random.fork_rng(devices=[]):new=ContrastiveAttention()
        result=new.load_state_dict(old.state_dict(),strict=False)
        assert set(result.missing_keys)=={'contrast_qk.weight','contrast_alpha'} and not result.unexpected_keys
        self.stem.hierarchy.global_patches=new
