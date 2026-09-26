"""Fine local sequence plus coarse input-only context from outside the window."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from patch_hierarchy_sequence import Network as Baseline


class ExternalContext(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_norm=nn.LayerNorm(64);self.query=nn.Linear(64,32,bias=False)
        self.token_encoder=nn.Sequential(nn.Linear(31,32),nn.GELU(),nn.LayerNorm(32))
        self.kv=nn.Linear(32,64,bias=False);self.output=nn.Linear(32,64,bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self,x,valid,tokens,token_valid):
        batch,channels,length=x.shape;patches=length//16
        local=x.transpose(1,2).reshape(batch,patches,16,channels)
        mask=valid.reshape(batch,patches,16)
        pooled=local.masked_fill(~mask[...,None],0).sum(2)/mask.sum(2)[...,None].clamp_min(1)
        q=self.query(self.query_norm(pooled)).reshape(batch,patches,4,8).transpose(1,2)
        encoded=self.token_encoder(tokens.float().masked_fill(~token_valid[...,None],0))
        encoded=encoded.masked_fill(~token_valid[...,None],0)
        k,v=self.kv(encoded).reshape(batch,-1,2,4,8).permute(2,0,3,1,4)
        scores=(q@k.transpose(-1,-2)).float()/math.sqrt(8)
        scores=scores.masked_fill(~token_valid[:,None,None,:],-1e4)
        context=(scores.softmax(-1).to(v.dtype)@v).transpose(1,2).reshape(batch,patches,32)
        context=self.output(context).repeat_interleave(16,dim=1)
        return context.masked_fill(~valid[...,None],0).transpose(1,2)


class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        with torch.random.fork_rng(devices=[]):self.external=ExternalContext()

    def forward(self,features,relative_depth,valid,auxiliary=False,absolute_depth=None,
                external_tokens=None,external_valid=None):
        assert external_tokens is not None and external_valid is not None
        mask=valid[:,None].to(features.dtype);point=features.transpose(1,2)*mask
        x=torch.cat([point,relative_depth[:,None]*mask,mask,absolute_depth[:,None]*mask],dim=1)
        stem=self.stem(x)
        stem=stem+self.external(stem,valid,external_tokens,external_valid)
        e0=self.enc0(stem);e1=self.enc1(self.down1(e0));e2=self.enc2(self.down2(e1))
        h=self.bottleneck(self.down3(e2))
        h=self.dec2(self.merge2(torch.cat([F.interpolate(h,size=e2.shape[-1],mode='linear',align_corners=False),e2],1)));coarse=h
        h=self.dec1(self.merge1(torch.cat([F.interpolate(h,size=e1.shape[-1],mode='linear',align_corners=False),e1],1)));mid=h
        h=self.dec0(self.merge0(torch.cat([F.interpolate(h,size=e0.shape[-1],mode='linear',align_corners=False),e0],1)))
        main=self.main_head(self.combine(torch.cat([h,self.point(point)],1)))
        return (main,self.aux_mid(mid),self.aux_coarse(coarse)) if auxiliary else main
