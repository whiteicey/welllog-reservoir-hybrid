"""Local point attention plus global patch-token attention, with masked pooling."""
import math
import torch
from torch import nn
from relative_only_sequence import Network as Baseline, RelativeOnlyStem


class SmallAttention(nn.Module):
    def __init__(self, width=32, heads=4, maximum_tokens=16):
        super().__init__();self.heads=heads;self.maximum_tokens=maximum_tokens
        self.norm=nn.LayerNorm(width)
        self.qkv=nn.Linear(width,width*3,bias=False)
        self.project=nn.Linear(width,width,bias=False)
        self.relative_bias=nn.Parameter(torch.zeros(heads,2*maximum_tokens-1))

    def forward(self,x,valid):
        batch,length,width=x.shape;assert length<=self.maximum_tokens
        z=self.norm(x).masked_fill(~valid[...,None],0)
        q,k,v=self.qkv(z).reshape(batch,length,3,self.heads,width//self.heads).permute(2,0,3,1,4)
        scores=(q@k.transpose(-2,-1)).float()/math.sqrt(width//self.heads)
        pos=torch.arange(length,device=x.device)
        bias=self.relative_bias[:,pos[:,None]-pos[None,:]+self.maximum_tokens-1]
        scores=(scores+bias).masked_fill(~valid[:,None,None,:],-1e4)
        attended=(scores.softmax(-1).to(v.dtype)@v).transpose(1,2).reshape(batch,length,width)
        return self.project(attended).masked_fill(~valid[...,None],0)


class PatchHierarchy(nn.Module):
    def __init__(self):
        super().__init__();self.patch_size=16
        self.norm=nn.LayerNorm(64);self.input=nn.Linear(64,32,bias=False)
        self.local=SmallAttention();self.global_patches=SmallAttention()
        self.output=nn.Linear(32,64,bias=False)
        nn.init.zeros_(self.output.weight)
        with torch.random.fork_rng(devices=[]):
            self.pool_norm=nn.LayerNorm(32);self.pool_score=nn.Linear(32,1,bias=False)
            nn.init.zeros_(self.pool_score.weight)

    def forward(self,x,valid):
        batch,channels,length=x.shape;assert length%16==0
        patches=length//16
        z=self.input(self.norm(x.transpose(1,2))).masked_fill(~valid[...,None],0)
        local=z.reshape(batch*patches,16,32);mask=valid.reshape(batch*patches,16)
        local=(local+self.local(local,mask)).masked_fill(~mask[...,None],0)
        scores=self.pool_score(self.pool_norm(local)).squeeze(-1).float()
        scores=(4*torch.tanh(scores/4)).masked_fill(~mask,-1e4)
        weights=scores.softmax(-1)*mask
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
        token=(local*weights.to(local.dtype)[...,None]).sum(1)
        token=token.reshape(batch,patches,32);token_valid=mask.any(1).reshape(batch,patches)
        global_context=self.global_patches(token,token_valid)
        mixed=local.reshape(batch,patches,16,32)+global_context[:,:,None,:]
        return self.output(mixed.reshape(batch,length,32)).masked_fill(~valid[...,None],0).transpose(1,2)


class PatchStem(RelativeOnlyStem):
    def __init__(self,old):
        super().__init__(old.in_channels,old.out_channels,1)
        with torch.no_grad():self.weight.copy_(old.weight);self.bias.copy_(old.bias)
        self.hierarchy=PatchHierarchy()

    def forward(self,x):
        stem=super().forward(x)
        return stem+self.hierarchy(stem,x[:,-2,:]>0)


class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        with torch.random.fork_rng(devices=[]):self.stem=PatchStem(self.stem)


