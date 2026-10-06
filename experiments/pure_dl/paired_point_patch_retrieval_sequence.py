"""Point queries retrieve patch context, alongside original patch broadcast."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from paired_posterior_router_sequence import (
    Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES)

class RetrievalHierarchy(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.patch_size=original.patch_size
        for name,child in original.named_children():
            self.add_module(name,child)
        with torch.random.fork_rng(devices=[]):
            self.retrieve_norm=nn.LayerNorm(32)
            self.query=nn.Linear(32,32,bias=False)
            self.key=nn.Linear(32,32,bias=False)
            self.value=nn.Linear(32,32,bias=False)
            self.retrieve_output=nn.Linear(32,32,bias=False)
            nn.init.zeros_(self.retrieve_output.weight)
            self.retrieve_bias=nn.Parameter(torch.zeros(4,31))

    def forward(self,x,valid):
        batch,channels,length=x.shape
        patches=length//16
        z=self.input(self.norm(x.transpose(1,2))).masked_fill(~valid[...,None],0)
        local=z.reshape(batch*patches,16,32)
        mask=valid.reshape(batch*patches,16)
        local=(local+self.local(local,mask)).masked_fill(~mask[...,None],0)
        token=(local.sum(1)/mask.sum(1,keepdim=True).clamp_min(1)).reshape(batch,patches,32)
        token_valid=mask.any(1).reshape(batch,patches)
        context=self.global_patches(token,token_valid)
        points=local.reshape(batch,length,32)
        q=self.query(self.retrieve_norm(points)).reshape(batch,length,4,8).transpose(1,2)
        memory=self.retrieve_norm(token+context).masked_fill(~token_valid[...,None],0)
        k=self.key(memory).reshape(batch,patches,4,8).transpose(1,2)
        v=self.value(memory).reshape(batch,patches,4,8).transpose(1,2)
        scores=(F.normalize(q.float(),dim=-1)@F.normalize(k.float(),dim=-1).transpose(-2,-1))*math.sqrt(8)
        offset=torch.arange(length,device=x.device)//16
        offset=offset[:,None]-torch.arange(patches,device=x.device)[None,:]+15
        scores=(scores+self.retrieve_bias[:,offset]).masked_fill(~token_valid[:,None,None,:],-1e4)
        retrieved=(scores.softmax(-1).to(v.dtype)@v).transpose(1,2).reshape(batch,length,32)
        retrieved=self.retrieve_output(retrieved).masked_fill(~valid[...,None],0)
        mixed=points+context[:,:,None,:].expand(-1,-1,16,-1).reshape(batch,length,32)+retrieved
        return self.output(mixed).masked_fill(~valid[...,None],0).transpose(1,2)

class Network(Baseline):
    def __init__(self,features=206,initial_means=None,*,retain_indices,numeric_indices,conditioning_indices):
        super().__init__(features,initial_means,retain_indices=retain_indices,
                         numeric_indices=numeric_indices,conditioning_indices=conditioning_indices)
        self.members[0].stem.hierarchy=RetrievalHierarchy(self.members[0].stem.hierarchy)
