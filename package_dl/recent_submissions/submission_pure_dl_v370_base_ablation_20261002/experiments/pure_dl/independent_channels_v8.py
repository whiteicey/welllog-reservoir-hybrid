"""Private, end-to-end single-target models; no backbone sharing between targets.

Architecture candidates: full-view single backbone; full+calibrated-view dual
backbones with a PRIVATE input router. We reuse pretrained initialization, not
live parameters. Each model returns only its assigned target distribution.
Packed frozen output rows preserve the pretrained convolution arithmetic; they
never supply other target labels or losses and are not trainable.
"""
from __future__ import annotations
import copy
import torch
from torch import nn
from torch.nn import functional as F
from paired_local_context_router_sequence import Network as Original, retained_indices, calibration_indices
from dual8222_location_readout_v5 import Network as Previous
from paired_detached_router_sequence import routed_merge

TARGETS = ('POR','PERM','SW')
ARCHITECTURES = ('dual','single')


class TargetReadout(nn.Module):
    def __init__(self, groups, expert: int, target: int):
        super().__init__()
        self.target = target
        self.valid = None
        # Keep the same packed 3/72-row convolutions as the retained checkpoint.
        gate_w = torch.cat([groups[c][expert].gate_weight for c in range(3)]).detach()
        gate_b = torch.cat([groups[c][expert].gate_bias for c in range(3)]).detach()
        rows = [groups[c][expert].density_rows() for c in range(3)]
        density_w = torch.cat([r[0] for r in rows]).detach()
        density_b = torch.cat([r[1] for r in rows]).detach()
        for stem, tensor, lo, hi in [
            ('gate_weight',gate_w,target,target+1),('gate_bias',gate_b,target,target+1),
            ('density_weight',density_w,24*target,24*(target+1)),
            ('density_bias',density_b,24*target,24*(target+1))]:
            self.register_parameter(stem,nn.Parameter(tensor[lo:hi].clone()))
            self.register_buffer(stem+'_before',tensor[:lo].clone())
            self.register_buffer(stem+'_after',tensor[hi:].clone())
        if target == 0:
            self.residual = copy.deepcopy(groups[0][expert].location_residual)
            self.residual.requires_grad_(True)
        else:
            self.residual = None

    def packed(self,name):
        return torch.cat([getattr(self,name+'_before'),getattr(self,name),getattr(self,name+'_after')],0)

    def forward(self,h):
        g = F.conv1d(h,self.packed('gate_weight'),self.packed('gate_bias')).transpose(1,2)
        r = F.conv1d(h,self.packed('density_weight'),self.packed('density_bias'))
        r = r.transpose(1,2).reshape(len(h),-1,3,3,8).float()
        c=self.target
        own=r[:,:,c:c+1]
        if self.residual is not None:
            if self.valid is None:raise RuntimeError('Valid-position mask was not set')
            delta=self.residual(h,self.valid,own[:,:,0,2].transpose(1,2)).transpose(1,2)
            p=own[:,:,0]
            own=torch.stack([p[:,:,0],p[:,:,1]+delta,p[:,:,2]],2).unsqueeze(2)
        return g[:,:,c:c+1],own


class TargetRouterOutput(nn.Module):
    def __init__(self,layer,target):
        super().__init__()
        self.target=target
        for name in ('weight','bias'):
            value=getattr(layer,name).detach()
            self.register_parameter(name,nn.Parameter(value[2*target:2*(target+1)].clone()))
            self.register_buffer(name+'_before',value[:2*target].clone())
            self.register_buffer(name+'_after',value[2*(target+1):].clone())

    def forward(self,x):
        w=torch.cat([self.weight_before,self.weight,self.weight_after],0)
        b=torch.cat([self.bias_before,self.bias,self.bias_after],0)
        return F.linear(x,w,b)


class Network(nn.Module):
    def __init__(self,master:dict,previous:dict,target:str,architecture:str):
        super().__init__()
        if target not in TARGETS or architecture not in ARCHITECTURES:
            raise ValueError('Invalid target/architecture')
        self.target=target;self.target_index=TARGETS.index(target);self.architecture=architecture
        names=master['selected_feature_names']
        numeric,condition=calibration_indices(names)
        original=Original(retain_indices=retained_indices(names),numeric_indices=numeric,conditioning_indices=condition)
        original.load_state_dict(master['state_dict'],strict=True)
        old=Previous(original,previous['initialization_manifest'],'point32')
        old.heads.load_state_dict(previous['readout']['head_state_dict'],strict=True)
        # Remove the old inference-only capture hooks: no detach/no_grad path remains.
        for handle in old.handles:handle.remove()
        count=2 if architecture=='dual' else 1
        selected=[]
        for e in range(count):
            member=old.backbone.members[e]
            member.main_head=TargetReadout(old.heads,e,self.target_index)
            member.requires_grad_(True)
            # Unused auxiliary heads are excluded rather than trained on other targets.
            member.aux_mid=nn.Identity();member.aux_coarse=nn.Identity()
            selected.append(member)
        self.members=nn.ModuleList(selected)
        if architecture=='dual':
            self.router=old.backbone.router
            self.router[2]=TargetRouterOutput(self.router[2],self.target_index)
            self.router.requires_grad_(True)
            self.register_buffer('router_indices',old.backbone.router_indices.detach().clone())
        del old,original

    def forward(self,x,relative,valid,absolute_depth=None):
        outputs=[]
        for member in self.members:
            member.main_head.valid=valid
            try:
                outputs.append(member(x,relative,valid,auxiliary=False,absolute_depth=absolute_depth))
            finally:
                member.main_head.valid=None
        if self.architecture=='single':return outputs[0]
        clean=x.index_select(-1,self.router_indices).masked_fill(~valid[...,None],0)
        h=self.router[1](self.router[0](clean)).masked_fill(~valid[...,None],0)
        context=self.router.local_context(h.transpose(1,2)).transpose(1,2).masked_fill(~valid[...,None],0)
        logits=self.router[2]((h+context).masked_fill(~valid[...,None],0))
        logits=logits.reshape(*x.shape[:2],3,2)[:,:,self.target_index:self.target_index+1]
        return routed_merge(outputs,logits)

    def parameter_groups(self):
        # One optimizer and schedule per independently constructed target model.
        head=[];body=[]
        for name,p in self.named_parameters():
            if not p.requires_grad:continue
            (head if '.main_head.' in name else body).append(p)
        return [{'params':body,'lr':1e-5,'name':'private_backbone_and_router'},
                {'params':head,'lr':3e-5,'name':'private_target_readout'}]

    def backbone_parameters(self):
        return [(name,p) for name,p in self.named_parameters() if '.main_head.' not in name]
