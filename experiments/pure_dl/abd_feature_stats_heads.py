"""Per-target readout restores frozen branch mean/scale discarded by LayerNorm."""
import torch
from torch import nn
from torch.nn import functional as F
from abd_feature_backbone_heads import FeatureBackbone, Network as Baseline


class StatsBackbone(FeatureBackbone):
    @torch.no_grad()
    def forward(self, features, relative_depth, valid, absolute_depth=None):
        self.captured.clear()
        for branch in self.branches:
            branch(features,relative_depth,valid,absolute_depth=absolute_depth)
        normalized=[];statistics=[]
        for i in range(5):
            h=self.captured[i].transpose(1,2)
            value=h.float()
            normalized.append(F.layer_norm(value,(128,)).to(h.dtype))
            mean=value.mean(-1,keepdim=True)
            log_std=.5*(value.var(-1,keepdim=True,unbiased=False)+1e-5).log()
            statistics.extend([mean.clamp(-8,8).to(h.dtype),log_std.clamp(-8,8).to(h.dtype)])
        self.captured.clear()
        return torch.cat(normalized+statistics,-1).masked_fill(~valid[...,None],0)


class StatsHead(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.hidden=original.hidden
        self.output=original.output
        with torch.random.fork_rng(devices=[]):
            self.statistics=nn.Linear(10,self.hidden.out_features,bias=False)
            nn.init.zeros_(self.statistics.weight)

    def forward(self, features):
        # Identical base head at initialization; target-specific statistics path starts at zero.
        hidden=self.hidden(features[...,:640])+self.statistics(features[...,640:])
        output=self.output(F.gelu(hidden))
        return output[...,0],output[...,1:].reshape(*features.shape[:2],3,8)


class Network(Baseline):
    def __init__(self, sources, initial_means, default_probabilities):
        super().__init__(sources,initial_means,default_probabilities)
        for handle in self.backbone.handles:handle.remove()
        self.backbone=StatsBackbone(sources)
        self.heads=nn.ModuleList([StatsHead(head) for head in self.heads])
