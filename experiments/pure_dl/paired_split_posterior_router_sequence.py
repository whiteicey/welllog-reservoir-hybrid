"""Separate default-mass and conditional-density routing over calibrated experts."""
import torch
from torch import nn
from torch.nn import functional as F
from paired_posterior_router_sequence import Network as Baseline, posterior_descriptors, retained_indices, calibration_indices, VIEW_NAMES


def split_merge(outputs, logits):
    gates=torch.stack([g.float() for g,_ in outputs],-1)
    routes=logits.float().log_softmax(-1)
    gate_route=routes[...,0,:]
    density_route=routes[...,1,:]
    log_default=(gate_route+F.logsigmoid(gates)).logsumexp(-1)
    log_survival=(gate_route+F.logsigmoid(-gates)).logsumexp(-1)
    gate=log_default-log_survival
    conditional=(density_route+F.logsigmoid(-gates)).log_softmax(-1)
    weights=torch.cat([raw[...,0,:].float().log_softmax(-1)+conditional[...,i,None]
                       for i,(_,raw) in enumerate(outputs)],-1)
    means=torch.cat([raw[...,1,:].float() for _,raw in outputs],-1)
    scales=torch.cat([raw[...,2,:].float() for _,raw in outputs],-1)
    return gate,torch.stack([weights,means,scales],-2)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices):
        super().__init__(features,initial_means,retain_indices=retain_indices,
                         numeric_indices=numeric_indices,conditioning_indices=conditioning_indices)
        old=self.router[2]
        with torch.random.fork_rng(devices=[]):
            new=nn.Linear(32,12,device=old.weight.device,dtype=old.weight.dtype)
            with torch.no_grad():
                new.weight.copy_(old.weight.reshape(3,2,32)[:,None].expand(3,2,2,32).reshape(12,32))
                new.bias.copy_(old.bias.reshape(3,2)[:,None].expand(3,2,2).reshape(12))
        self.router[2]=new

    def route_logits(self, features, valid, main_outputs):
        original=features.index_select(-1,self.router_indices).float()
        descriptors=posterior_descriptors(main_outputs)
        x=torch.cat([original,descriptors],-1).masked_fill(~valid[...,None],0)
        return self.router(x).reshape(*features.shape[:2],3,2,2)

    def prediction_parts(self, features, relative_depth, valid, absolute_depth=None):
        members=[m(features,relative_depth,valid,absolute_depth=absolute_depth) for m in self.members]
        logits=self.route_logits(features,valid,members)
        return members,logits.float().softmax(-1),split_merge(members,logits)

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        members=[m(features,relative_depth,valid,auxiliary,absolute_depth) for m in self.members]
        main=[output[0] for output in members] if auxiliary else members
        logits=self.route_logits(features,valid,main)
        if auxiliary:
            detached=[tuple(value.detach() for value in output) for output in main]
            return members,split_merge(detached,logits)
        return split_merge(main,logits)
