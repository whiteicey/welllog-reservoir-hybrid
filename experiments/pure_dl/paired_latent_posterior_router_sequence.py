"""Detached expert readout representations provide a residual routing signal."""
import torch
from torch import nn
from torch.nn import functional as F
from paired_posterior_router_sequence import Network as Baseline, posterior_descriptors, retained_indices, calibration_indices, VIEW_NAMES
from paired_detached_router_sequence import routed_merge


def forward_features(member, features, relative_depth, valid, absolute_depth):
    # Explicit equivalent of the inherited absolute_depth_sequence forward up
    # to the head inputs. RelativeOnlyStem itself removes the absolute channel.
    mask = valid[:, None].to(features.dtype)
    point = features.transpose(1, 2) * mask
    assert absolute_depth is not None and absolute_depth.shape == valid.shape
    x = torch.cat([point, relative_depth[:, None]*mask, mask,
                   absolute_depth[:, None]*mask], dim=1)
    e0 = member.enc0(member.stem(x))
    e1 = member.enc1(member.down1(e0))
    e2 = member.enc2(member.down2(e1))
    h = member.bottleneck(member.down3(e2))
    h = member.dec2(member.merge2(torch.cat([
        F.interpolate(h, size=e2.shape[-1], mode='linear', align_corners=False), e2], 1)))
    coarse = h
    h = member.dec1(member.merge1(torch.cat([
        F.interpolate(h, size=e1.shape[-1], mode='linear', align_corners=False), e1], 1)))
    mid = h
    h = member.dec0(member.merge0(torch.cat([
        F.interpolate(h, size=e0.shape[-1], mode='linear', align_corners=False), e0], 1)))
    return member.combine(torch.cat([h, member.point(point)], 1)), mid, coarse


class LatentRoute(nn.Module):
    def __init__(self):
        super().__init__()
        self.projections=nn.ModuleList([nn.Sequential(nn.LayerNorm(128),nn.Linear(128,16),nn.GELU()) for _ in range(2)])
        self.output=nn.Linear(32,6)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self, latents, valid):
        compressed=[project(value.detach().transpose(1,2)).masked_fill(~valid[...,None],0)
                    for project,value in zip(self.projections,latents)]
        return self.output(torch.cat(compressed,-1)).masked_fill(~valid[...,None],0)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices):
        super().__init__(features,initial_means,retain_indices=retain_indices,
                         numeric_indices=numeric_indices,conditioning_indices=conditioning_indices)
        with torch.random.fork_rng(devices=[]):
            branch=LatentRoute()
        self.router.add_module('latent_readout',branch)

    def expert_outputs(self, features, relative_depth, valid, auxiliary, absolute_depth):
        outputs=[];latents=[]
        for i,member in enumerate(self.members):
            view=features
            if i==1:
                # Exact explicit preprocessing of calibrated68 member before
                # inherited U-Net. No hook or temporary module mutation.
                clean=features.masked_fill(~valid[...,None],0)
                context=clean.index_select(-1,member.conditioning_indices)
                gamma,beta=member.calibration(context).chunk(2,-1)
                numeric=clean.index_select(-1,member.numeric_indices)
                replacement=(numeric*(1+gamma.tanh())+beta.tanh()).masked_fill(~valid[...,None],0).to(clean.dtype)
                view=clean.index_copy(-1,member.numeric_indices,replacement)
                view=view.masked_fill(~member.feature_view_mask[None,None,:],0)
            fine,mid,coarse=forward_features(member,view,relative_depth,valid,absolute_depth)
            latents.append(fine)
            main=member.main_head(fine)
            outputs.append((main,member.aux_mid(mid),member.aux_coarse(coarse)) if auxiliary else main)
        return outputs,latents

    def route_logits_with_latents(self, features, valid, main, latents):
        original=features.index_select(-1,self.router_indices).float()
        x=torch.cat([original,posterior_descriptors(main)],-1).masked_fill(~valid[...,None],0)
        base=self.router[2](self.router[1](self.router[0](x)))
        correction=self.router.latent_readout(latents,valid)
        return (base+correction).reshape(*features.shape[:2],3,2)

    def prediction_parts(self, features, relative_depth, valid, absolute_depth=None):
        members,latents=self.expert_outputs(features,relative_depth,valid,False,absolute_depth)
        logits=self.route_logits_with_latents(features,valid,members,latents)
        return members,logits.float().softmax(-1),routed_merge(members,logits)

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        members,latents=self.expert_outputs(features,relative_depth,valid,auxiliary,absolute_depth)
        main=[output[0] for output in members] if auxiliary else members
        logits=self.route_logits_with_latents(features,valid,main,latents)
        if auxiliary:
            detached=[tuple(value.detach() for value in output) for output in main]
            return members,routed_merge(detached,logits)
        return routed_merge(main,logits)
