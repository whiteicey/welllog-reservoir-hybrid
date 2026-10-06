"""Five frozen feature branches from A6/B43/D52, then independent new target heads."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TargetHead(nn.Module):
    def __init__(self, means, default_probability, hidden=128):
        super().__init__()
        self.hidden = nn.Linear(640, hidden)
        self.output = nn.Linear(hidden, 25)
        nn.init.normal_(self.output.weight, 0, .005)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            p=float(default_probability)
            self.output.bias[0]=math.log(p/(1-p))
            density=self.output.bias[1:].view(3, 8)
            density[1].copy_(torch.as_tensor(means))
            density[2].fill_(math.log(math.expm1(.5-.025)))

    def forward(self, features):
        output=self.output(F.gelu(self.hidden(features)))
        return output[..., 0], output[..., 1:].reshape(*features.shape[:2], 3, 8)


class FeatureBackbone(nn.Module):
    def __init__(self, sources):
        super().__init__()
        self.sources=nn.ModuleList(sources).eval().requires_grad_(False)
        self.branches=list(sources[0].members)+list(sources[1].members)+[sources[2]]
        assert len(self.branches)==5
        self.captured={}
        self.handles=[branch.main_head.register_forward_pre_hook(self.capture(i))
                      for i,branch in enumerate(self.branches)]

    def capture(self, index):
        def hook(module, inputs):
            self.captured[index]=inputs[0].detach()
        return hook

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, features, relative_depth, valid, absolute_depth=None):
        self.captured.clear()
        for branch in self.branches:
            # Original output is discarded. Neither old heads nor posterior routers feed new heads.
            branch(features,relative_depth,valid,absolute_depth=absolute_depth)
        features_by_branch=[]
        for i in range(5):
            h=self.captured[i].transpose(1, 2)
            assert h.shape[-1]==128
            features_by_branch.append(F.layer_norm(h.float(),(128,)).to(h.dtype))
        self.captured.clear()
        return torch.cat(features_by_branch,-1).masked_fill(~valid[...,None],0)


class Network(nn.Module):
    def __init__(self, sources, initial_means, default_probabilities):
        super().__init__()
        self.backbone=FeatureBackbone(sources)
        self.heads=nn.ModuleList([TargetHead(initial_means[j],default_probabilities[j]) for j in range(3)])

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        assert not auxiliary
        frozen_features=self.backbone(features,relative_depth,valid,absolute_depth)
        outputs=[head(frozen_features) for head in self.heads]
        return torch.stack([g for g,_ in outputs],-1),torch.stack([r for _,r in outputs],2)
