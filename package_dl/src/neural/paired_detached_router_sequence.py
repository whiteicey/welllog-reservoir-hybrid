"""Train-only gradient-isolated router over two calibrated68 experts."""
import torch
from torch import nn
from torch.nn import functional as F
from .paired_calibrated_view_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES


def routed_merge(outputs, logits):
    gates=torch.stack([g.float() for g,_ in outputs],-1)
    routing=logits.float().log_softmax(-1)
    survival=routing+F.logsigmoid(-gates)
    gate=(routing+F.logsigmoid(gates)).logsumexp(-1)-survival.logsumexp(-1)
    shares=survival.log_softmax(-1)
    weights=torch.cat([raw[...,0,:].float().log_softmax(-1)+shares[...,i,None]
                       for i,(_,raw) in enumerate(outputs)],-1)
    means=torch.cat([raw[...,1,:].float() for _,raw in outputs],-1)
    scales=torch.cat([raw[...,2,:].float() for _,raw in outputs],-1)
    return gate,torch.stack([weights,means,scales],-2)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices):
        super().__init__(features,initial_means,retain_indices=retain_indices,
                         numeric_indices=numeric_indices,conditioning_indices=conditioning_indices)
        self.register_buffer('router_indices',torch.tensor(retain_indices,dtype=torch.long))
        with torch.random.fork_rng(devices=[]):
            self.router=nn.Sequential(nn.Linear(68,32),nn.GELU(),nn.Linear(32,6))
            nn.init.zeros_(self.router[-1].weight);nn.init.zeros_(self.router[-1].bias)

    def route_logits(self, features, valid):
        x=features.index_select(-1,self.router_indices).masked_fill(~valid[...,None],0)
        return self.router(x).reshape(*features.shape[:2],3,2)

    def prediction_parts(self, features, relative_depth, valid, absolute_depth=None):
        members=[m(features,relative_depth,valid,absolute_depth=absolute_depth) for m in self.members]
        logits=self.route_logits(features,valid)
        return members,logits.float().softmax(-1),routed_merge(members,logits)

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        members=[m(features,relative_depth,valid,auxiliary,absolute_depth) for m in self.members]
        logits=self.route_logits(features,valid)
        if auxiliary:
            detached=[tuple(value.detach() for value in output[0]) for output in members]
            return members,routed_merge(detached,logits)
        return routed_merge(members,logits)
