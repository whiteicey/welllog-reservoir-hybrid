"""Detached own POR/PERM posterior summaries condition only main SW outputs."""
import torch
from torch import nn
from paired_posterior_router_sequence import Network as Baseline, posterior_descriptors, retained_indices, calibration_indices, VIEW_NAMES
from paired_detached_router_sequence import routed_merge


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices):
        super().__init__(features,initial_means,retain_indices=retain_indices,
                         numeric_indices=numeric_indices,conditioning_indices=conditioning_indices)
        with torch.random.fork_rng(devices=[]):
            self.sw_conditioners=nn.ModuleList([nn.Sequential(nn.Linear(8,16),nn.GELU(),nn.Linear(16,9)) for _ in range(2)])
            for layer in self.sw_conditioners:
                nn.init.zeros_(layer[-1].weight);nn.init.zeros_(layer[-1].bias)

    def correct_main(self, output, index, valid):
        gate,raw=output
        # Existing descriptor helper explicitly detaches gate/raw, and orders
        # target summaries gate/entropy/mean/log1p(total std). First8=POR/PERM.
        descriptors=posterior_descriptors([output])[...,:8].masked_fill(~valid[...,None],0)
        delta=self.sw_conditioners[index](descriptors).masked_fill(~valid[...,None],0)
        sw_gate=(gate[...,2]+delta[...,0]).to(gate.dtype)
        new_gate=torch.cat([gate[...,:2],sw_gate[...,None]],-1)
        sw=raw[...,2,:,:]
        sw_mean=(sw[...,1,:]+delta[...,1:]).to(raw.dtype)
        new_sw=torch.stack([sw[...,0,:],sw_mean,sw[...,2,:]],-2)
        new_raw=torch.cat([raw[...,:2,:,:],new_sw.unsqueeze(-3)],-3)
        return new_gate,new_raw

    def corrected_members(self, features, relative_depth, valid, auxiliary, absolute_depth):
        outputs=[]
        for i,member in enumerate(self.members):
            output=member(features,relative_depth,valid,auxiliary,absolute_depth)
            if auxiliary:
                outputs.append((self.correct_main(output[0],i,valid),output[1],output[2]))
            else:
                outputs.append(self.correct_main(output,i,valid))
        return outputs

    def prediction_parts(self, features, relative_depth, valid, absolute_depth=None):
        members=self.corrected_members(features,relative_depth,valid,False,absolute_depth)
        logits=self.route_logits(features,valid,members)
        return members,logits.float().softmax(-1),routed_merge(members,logits)

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        members=self.corrected_members(features,relative_depth,valid,auxiliary,absolute_depth)
        main=[output[0] for output in members] if auxiliary else members
        logits=self.route_logits(features,valid,main)
        if auxiliary:
            detached=[tuple(value.detach() for value in output) for output in main]
            return members,routed_merge(detached,logits)
        return routed_merge(main,logits)
