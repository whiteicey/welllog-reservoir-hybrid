"""Original global-cosine U-Net plus a pointwise learned MLP member."""
import torch
from torch import nn
from torch.nn import functional as F
from global_cosine_sequence import Network as UNetMember
from multiscale_sequence import DensityHead
from paired_backbone_sequence import objective, merge_predictions


def masked_pool(features, valid, factor):
    mask = valid[:,None].to(features.dtype)
    denominator = F.avg_pool1d(mask,factor,stride=factor)
    numerator = F.avg_pool1d(features*mask,factor,stride=factor)
    return (numerator/denominator.clamp_min(1./factor)).masked_fill(denominator==0,0)


class PointMLPMember(nn.Module):
    def __init__(self, features=206, initial_means=None):
        super().__init__()
        self.stem = nn.Linear(features+2,256)
        self.norm = nn.LayerNorm(256)
        self.expand = nn.Linear(256,256)
        self.project = nn.Linear(256,256)
        self.gain = nn.Parameter(torch.full((256,),.1))
        self.output = nn.Linear(256,128)
        self.main_head = DensityHead(128,initial_means)
        self.aux_mid = DensityHead(128,initial_means)
        self.aux_coarse = DensityHead(128,initial_means)

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        mask = valid[...,None]
        x = torch.cat([features.masked_fill(~mask,0),
                       relative_depth[...,None].masked_fill(~mask,0),
                       mask.to(features.dtype)],dim=-1)
        h = F.gelu(self.stem(x)).masked_fill(~mask,0)
        z = self.norm(h).masked_fill(~mask,0)
        z = F.gelu(self.expand(z)).masked_fill(~mask,0)
        z = self.project(z).masked_fill(~mask,0)
        h = (h+self.gain*z).masked_fill(~mask,0)
        fine = F.gelu(self.output(h)).masked_fill(~mask,0).transpose(1,2)
        main = self.main_head(fine)
        if not auxiliary:
            return main
        return main,self.aux_mid(masked_pool(fine,valid,2)),self.aux_coarse(masked_pool(fine,valid,4))


class Network(nn.Module):
    def __init__(self, features=206, initial_means=None):
        super().__init__()
        self.members = nn.ModuleList([UNetMember(features,initial_means),
                                      PointMLPMember(features,initial_means)])

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        outputs = [member(features,relative_depth,valid,auxiliary,absolute_depth)
                   for member in self.members]
        return outputs if auxiliary else merge_predictions(outputs)
