"""Fixed equal posterior pair: original global-cosine U-Net and full-resolution TCN."""
import torch
from torch import nn
from torch.nn import functional as F
from global_cosine_sequence import Network as UNetMember
from depthwise_norm_sequence import DepthNorm
from multiscale_sequence import DensityHead
from paired_backbone_sequence import objective, merge_predictions


class ResidualBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.norm = DepthNorm(128, eps=1e-5)
        self.depthwise = nn.Conv1d(128,128,7,padding=3*dilation,dilation=dilation,groups=128)
        self.expand = nn.Conv1d(128,128,1)
        self.project = nn.Conv1d(128,128,1)
        self.gain = nn.Parameter(torch.full((1,128,1),.1))

    def forward(self, x, valid):
        mask = valid[:,None]
        normalized = self.norm(x).masked_fill(~mask,0)
        mixed = self.depthwise(normalized).masked_fill(~mask,0)
        residual = self.project(.5*F.gelu(self.expand(mixed)))
        return (x+self.gain*residual).masked_fill(~mask,0)


def masked_pool(features, valid, factor):
    mask = valid[:,None].to(features.dtype)
    numerator = F.avg_pool1d(features*mask,factor,stride=factor)
    denominator = F.avg_pool1d(mask,factor,stride=factor)
    return (numerator/denominator.clamp_min(1./factor)).masked_fill(denominator==0,0)


class TCNMember(nn.Module):
    def __init__(self, features=206, initial_means=None):
        super().__init__()
        self.stem = nn.Conv1d(features+2,128,1)
        self.blocks = nn.ModuleList([ResidualBlock(d) for d in (1,2,4,8,16,32)])
        self.main_head = DensityHead(128,initial_means)
        self.aux_mid = DensityHead(128,initial_means)
        self.aux_coarse = DensityHead(128,initial_means)

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        mask = valid[:,None]
        x = torch.cat([features.transpose(1,2).masked_fill(~mask,0),
                       relative_depth[:,None].masked_fill(~mask,0),
                       mask.to(features.dtype)],dim=1)
        h = self.stem(x).masked_fill(~mask,0)
        for block in self.blocks:
            h = block(h,valid)
        main = self.main_head(h)
        if not auxiliary:
            return main
        return main,self.aux_mid(masked_pool(h,valid,2)),self.aux_coarse(masked_pool(h,valid,4))


class Network(nn.Module):
    def __init__(self, features=206, initial_means=None):
        super().__init__()
        # Original member0 is initialized first; the TCN then draws its own
        # weights from the fixed run seed. No fitted parameters are loaded.
        self.members = nn.ModuleList([UNetMember(features,initial_means),
                                      TCNMember(features,initial_means)])

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        outputs = [member(features,relative_depth,valid,auxiliary,absolute_depth)
                   for member in self.members]
        return outputs if auxiliary else merge_predictions(outputs)
