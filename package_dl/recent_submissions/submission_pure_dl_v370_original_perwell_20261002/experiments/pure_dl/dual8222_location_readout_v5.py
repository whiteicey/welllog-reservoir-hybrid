"""Bounded residual LOCATION readouts on the retained two-expert model.

Three predeclared arms share trainable linear POR locations, fixed gate/mixture/
scale/PERM/SW and the same source function. Only the added nonlinear residual is
bounded; the common linear location update is not bounded.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from dual8222_por_subspace import Network as LocationBase
from paired_detached_router_sequence import routed_merge

MODES = {'linear':('mean',), 'point32':('mean','bounded_point_residual'),
         'local32':('mean','bounded_local_residual')}
COUNTS = {'linear':2064, 'point32':10848, 'local32':11232}


class BoundedLocationResidual(nn.Module):
    """h B,128,L; fixed raw_scale B,8,L; correction B,8,L."""
    def __init__(self, local: bool):
        super().__init__()
        self.reduce = nn.Conv1d(128,32,1)
        self.output = nn.Conv1d(32,8,1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        # Context construction must not change the next expert's initialization.
        with torch.random.fork_rng(devices=[]):
            self.context = nn.Conv1d(32,32,5,padding=2,groups=32) if local else None
            if self.context is not None:
                nn.init.zeros_(self.context.weight)
                nn.init.zeros_(self.context.bias)
        self.bound_fraction = 0.25

    def forward(self,h,valid,raw_scale):
        hidden = F.gelu(self.reduce(h)).masked_fill(~valid[:,None],0)
        if self.context is not None:
            hidden = hidden + self.context(hidden).masked_fill(~valid[:,None],0)
        proposal = self.output(hidden).float()
        bound = self.bound_fraction*(F.softplus(raw_scale.detach().float())+.025)
        # Derivative at proposal==0 is one. Preserve a comparable early update scale.
        return (bound*torch.tanh(proposal/bound)).masked_fill(~valid[:,None],0)


class Network(LocationBase):
    def __init__(self,original,selected,mode):
        if mode not in MODES:
            raise ValueError('Unknown location-readout arm: '+str(mode))
        super().__init__(original,selected,'mean_only')
        self.mode = mode
        if mode != 'linear':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20261010)
                for head in self.heads[0]:
                    head.location_residual = BoundedLocationResidual(local=mode=='local32')
        assert sum(p.numel() for p in self.parameters() if p.requires_grad)==COUNTS[mode]

    def forward(self,features,relative_depth,valid,auxiliary=False,absolute_depth=None):
        if auxiliary:
            raise ValueError('No new auxiliary supervision')
        self.captured.clear()
        with torch.no_grad():
            for member in self.backbone.members:
                member(features,relative_depth,valid,absolute_depth=absolute_depth)
            logits=self.backbone.route_logits(features,valid)
        outputs=[]
        for e in range(2):
            h=self.captured[e]
            heads=[self.heads[c][e] for c in range(3)]
            gate=F.conv1d(h,torch.cat([head.gate_weight for head in heads]),
                         torch.cat([head.gate_bias for head in heads]))
            rows=[head.density_rows() for head in heads]
            raw=F.conv1d(h,torch.cat([v[0] for v in rows]),torch.cat([v[1] for v in rows]))
            raw=raw.transpose(1,2).reshape(len(h),-1,3,3,8).float()
            if self.mode!='linear':
                delta=heads[0].location_residual(h,valid,raw[:,:,0,2,:].transpose(1,2))
                por=raw[:,:,0]
                updated_por=torch.stack([por[:,:,0],por[:,:,1]+delta.transpose(1,2),por[:,:,2]],dim=2)
                raw=torch.cat([updated_por.unsqueeze(2),raw[:,:,1:]],dim=2)
            outputs.append((gate.transpose(1,2),raw))
        self.captured.clear()
        return routed_merge(outputs,logits)

    def export_readout(self):
        """Nonlinear readouts cannot be flattened into the legacy linear matrix."""
        return {'format_version':1, 'model_module':'dual8222_location_readout_v5',
                'mode':self.mode, 'head_state_dict':{
                    k:v.detach().cpu().clone() for k,v in self.heads.state_dict().items()},
                'requires_initialization_manifest':True,
                'nonlinear_residual_bound_fraction':0.25 if self.mode!='linear' else None}
