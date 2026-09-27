"""Posterior router input gains pairwise member-disagreement descriptors."""
import torch
from torch import nn
from torch.nn import functional as F
from paired_calibrated_view_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES
from paired_detached_router_sequence import routed_merge
from paired_posterior_router_sequence import posterior_descriptors


def disagreement_descriptors(outputs):
    # Pairwise disagreement between the two members' detached main posteriors.
    # Order: target then |mean diff| / |log std ratio| / |gate diff|.
    (gate0,raw0),(gate1,raw1)=outputs
    gate0=gate0.detach().float();raw0=raw0.detach().float()
    gate1=gate1.detach().float();raw1=raw1.detach().float()
    weight0=raw0[...,0,:].softmax(-1);mu0=raw0[...,1,:];sd0=F.softplus(raw0[...,2,:])+.025
    weight1=raw1[...,0,:].softmax(-1);mu1=raw1[...,1,:];sd1=F.softplus(raw1[...,2,:])+.025
    mean0=(weight0*mu0).sum(-1);mean1=(weight1*mu1).sum(-1)
    var0=(weight0*(sd0.square()+(mu0-mean0[...,None]).square())).sum(-1)
    var1=(weight1*(sd1.square()+(mu1-mean1[...,None]).square())).sum(-1)
    std0=var0.clamp_min(0).sqrt();std1=var1.clamp_min(0).sqrt()
    log_std_ratio=(std1.clamp_min(1e-6).log()-std0.clamp_min(1e-6).log()).clamp(-8,8)
    gate_diff=(gate0.sigmoid()-gate1.sigmoid()).abs()
    values=torch.stack([(mean0-mean1).abs().clamp(max=8),log_std_ratio.abs(),gate_diff],-1)
    return values.flatten(-2)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices):
        # Same construction order as the posterior router: experts, then router.
        super().__init__(features,initial_means,retain_indices=retain_indices,
                         numeric_indices=numeric_indices,conditioning_indices=conditioning_indices)
        self.register_buffer('router_indices',torch.tensor(retain_indices,dtype=torch.long))
        with torch.random.fork_rng(devices=[]):
            self.router=nn.Sequential(nn.Linear(101,32),nn.GELU(),nn.Linear(32,6))
            nn.init.zeros_(self.router[-1].weight);nn.init.zeros_(self.router[-1].bias)

    def route_logits(self, features, valid, main_outputs):
        original=features.index_select(-1,self.router_indices).float()
        descriptors=posterior_descriptors(main_outputs)
        disagreement=disagreement_descriptors(main_outputs)
        x=torch.cat([original,descriptors,disagreement],-1).masked_fill(~valid[...,None],0)
        return self.router(x).reshape(*features.shape[:2],3,2)

    def prediction_parts(self, features, relative_depth, valid, absolute_depth=None):
        members=[m(features,relative_depth,valid,absolute_depth=absolute_depth) for m in self.members]
        logits=self.route_logits(features,valid,members)
        return members,logits.float().softmax(-1),routed_merge(members,logits)

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        members=[m(features,relative_depth,valid,auxiliary,absolute_depth) for m in self.members]
        main=[output[0] for output in members] if auxiliary else members
        logits=self.route_logits(features,valid,main)
        if auxiliary:
            detached=[tuple(value.detach() for value in output) for output in main]
            return members,routed_merge(detached,logits)
        return routed_merge(main,logits)
