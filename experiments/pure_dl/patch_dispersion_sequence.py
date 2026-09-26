"""Global patch tokens retain means and learn a residual from within-patch dispersion."""
import torch
from torch import nn
from global_cosine_sequence import Network as Baseline
from patch_hierarchy_sequence import PatchHierarchy
from cosine_patch_sequence import SmallAttention

def masked_dispersion(local,mask):
    count=mask.sum(1,keepdim=True).clamp_min(1)
    values=local.float().masked_fill(~mask[...,None],0)
    mean=values.sum(1)/count
    centered=(values-mean[:,None,:]).masked_fill(~mask[...,None],0)
    variance=centered.square().sum(1)/count
    # Smooth at zero variance; no sqrt(0) derivative for empty/singleton patches.
    return (torch.sqrt(variance+1e-6)-.001).masked_fill(~mask.any(1)[:,None],0)

class DispersionHierarchy(PatchHierarchy):
    def __init__(self):
        super().__init__()
        self.global_patches=SmallAttention()
        self.dispersion=nn.Linear(32,32,bias=False)
        nn.init.zeros_(self.dispersion.weight)
    def forward(self,x,valid):
        batch,channels,length=x.shape;assert length%16==0
        patches=length//16
        z=self.input(self.norm(x.transpose(1,2))).masked_fill(~valid[...,None],0)
        local=z.reshape(batch*patches,16,32);mask=valid.reshape(batch*patches,16)
        local=(local+self.local(local,mask)).masked_fill(~mask[...,None],0)
        token=local.sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        token=token+self.dispersion(masked_dispersion(local,mask).to(local.dtype))
        token=token.reshape(batch,patches,32);token_valid=mask.any(1).reshape(batch,patches)
        context=self.global_patches(token,token_valid)
        mixed=local.reshape(batch,patches,16,32)+context[:,:,None,:]
        return self.output(mixed.reshape(batch,length,32)).masked_fill(~valid[...,None],0).transpose(1,2)

class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        old=self.stem.hierarchy
        with torch.random.fork_rng(devices=[]): new=DispersionHierarchy()
        missing=new.load_state_dict(old.state_dict(),strict=False)
        assert missing.missing_keys==['dispersion.weight'] and not missing.unexpected_keys
        self.stem.hierarchy=new
