"""Fixed full206/calibrated68/uncalibrated68 expert ensemble with detached posterior router."""
import torch
from torch import nn
from paired_posterior_router_sequence import Network as Baseline, posterior_descriptors, retained_indices, calibration_indices, VIEW_NAMES
from paired_well_view_sequence import WellViewMember as PointViewMember, retained_indices as point_indices_from_schema
from global_cosine_sequence import Network as Member
from paired_detached_router_sequence import routed_merge
from por_t3_patch_loss import objective as member_objective


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices, point_indices):
        # Complete unchanged original two-member initialization first.
        super().__init__(features,initial_means,retain_indices=retain_indices,
                         numeric_indices=numeric_indices,conditioning_indices=conditioning_indices)
        with torch.random.fork_rng(devices=[]):
            third=Member(features,initial_means)
            self.members.append(PointViewMember(third,point_indices,features))
            self.router=nn.Sequential(nn.Linear(104,32),nn.GELU(),nn.Linear(32,9))
            nn.init.zeros_(self.router[-1].weight);nn.init.zeros_(self.router[-1].bias)

    def route_logits(self, features, valid, main_outputs):
        assert len(main_outputs)==3
        original=features.index_select(-1,self.router_indices).float()
        descriptors=posterior_descriptors(main_outputs)
        x=torch.cat([original,descriptors],-1).masked_fill(~valid[...,None],0)
        return self.router(x).reshape(*features.shape[:2],3,3)


def objective(outputs,target,default,mask,deep_supervision=False,weights=None,*,epoch):
    assert len(outputs)==3 and deep_supervision
    return sum(member_objective(output,target,default,mask,True,weights,epoch=epoch)
               for output in outputs)/3
