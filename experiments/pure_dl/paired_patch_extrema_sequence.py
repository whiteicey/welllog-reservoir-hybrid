"""Bounded high/low contrasts augment mean tokens in full-feature expert."""
import torch
from torch import nn
from paired_posterior_router_sequence import (
    Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES)


class ExtremaHierarchy(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.patch_size = original.patch_size
        for name, child in original.named_children():
            self.add_module(name, child)
        with torch.random.fork_rng(devices=[]):
            self.extrema_projection = nn.Linear(64,32,bias=False)
            nn.init.zeros_(self.extrema_projection.weight)

    def forward(self,x,valid):
        batch,channels,length=x.shape
        patches=length//16
        z=self.input(self.norm(x.transpose(1,2))).masked_fill(~valid[...,None],0)
        local=z.reshape(batch*patches,16,32)
        mask=valid.reshape(batch*patches,16)
        local=(local+self.local(local,mask)).masked_fill(~mask[...,None],0)
        mean=local.sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        present=mask.any(1)
        high=local.masked_fill(~mask[...,None],torch.finfo(local.dtype).min).amax(1)
        low=local.masked_fill(~mask[...,None],torch.finfo(local.dtype).max).amin(1)
        high=high.masked_fill(~present[:,None],0)
        low=low.masked_fill(~present[:,None],0)
        contrast=torch.cat([(high-mean).tanh(),(mean-low).tanh()],-1)
        token=mean+self.extrema_projection(contrast)
        token=token.reshape(batch,patches,32)
        global_context=self.global_patches(token,present.reshape(batch,patches))
        mixed=local.reshape(batch,patches,16,32)+global_context[:,:,None,:]
        return self.output(mixed.reshape(batch,length,32)).masked_fill(
            ~valid[...,None],0).transpose(1,2)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices,
                 numeric_indices, conditioning_indices):
        super().__init__(features,initial_means,retain_indices=retain_indices,
                         numeric_indices=numeric_indices,conditioning_indices=conditioning_indices)
        self.members[0].stem.hierarchy=ExtremaHierarchy(self.members[0].stem.hierarchy)
