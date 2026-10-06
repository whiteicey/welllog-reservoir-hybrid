"""Neighbor-conditioned hidden residual in the calibrated68 expert only."""
import torch
from torch import nn
from torch.nn import functional as F
from paired_posterior_router_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES
from paired_well_view_sequence import WellViewMember


class ContextCalibrationMember(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
        with torch.random.fork_rng(devices=[]):
            self.neighbor_correction = nn.Linear(32,32,bias=False)
            nn.init.zeros_(self.neighbor_correction.weight)

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        member = self.original
        clean = features.masked_fill(~valid[...,None],0)
        context = clean.index_select(-1,member.conditioning_indices)
        hidden = member.calibration[1](member.calibration[0](context))
        hidden = hidden.masked_fill(~valid[...,None],0)
        # Average only valid input positions, without mixing wells or padded duplicates.
        count = F.avg_pool1d(valid[:,None].float(),17,stride=1,padding=8)
        average = F.avg_pool1d(hidden.float().transpose(1,2),17,stride=1,padding=8)
        average = (average/count.clamp_min(1e-6)).transpose(1,2)
        residual = (average-hidden.float()).masked_fill(~valid[...,None],0)
        adapted = hidden + self.neighbor_correction(residual).to(hidden.dtype)
        gamma,beta = member.calibration[2](adapted).chunk(2,-1)
        numeric = clean.index_select(-1,member.numeric_indices)
        replacement = (numeric*(1+gamma.tanh())+beta.tanh()).masked_fill(~valid[...,None],0).to(clean.dtype)
        updated = clean.index_copy(-1,member.numeric_indices,replacement)
        # The original calibration was already applied; enter at its inherited view/backbone.
        return WellViewMember.forward(member,updated,relative_depth,valid,auxiliary,absolute_depth)


class Network(Baseline):
    def __init__(self,features=206,initial_means=None,*,retain_indices,numeric_indices,conditioning_indices):
        super().__init__(features,initial_means,retain_indices=retain_indices,
                         numeric_indices=numeric_indices,conditioning_indices=conditioning_indices)
        self.members[1]=ContextCalibrationMember(self.members[1])
