"""POR-only density subspace updates from the verified 80.257638 fixed-head recipe.

Frozen rows are buffers, not gradient-masked rows of a larger Parameter, so AdamW
cannot move a supposedly fixed mixture weight or gate through weight decay.
Only the original two experts are used. PERM and SW retain their selected heads.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from paired_detached_router_sequence import routed_merge

MODES = {'mean_only': ('mean',), 'mean_scale': ('mean', 'scale'),
         'density_all': ('mixture', 'mean', 'scale')}


class FactorHead(nn.Module):
    def __init__(self, state: dict, prefix: str, train_factors: tuple[str, ...]):
        super().__init__()
        for part in ['gate_weight', 'gate_bias']:
            self.register_buffer(part, state[prefix+part].detach().clone())
        weight, bias = state[prefix+'density_weight'], state[prefix+'density_bias']
        if weight.shape != (24,128,1) or bias.shape != (24,):
            raise ValueError('Unexpected source density rows')
        for j, factor in enumerate(['mixture', 'mean', 'scale']):
            for suffix, value in [('weight',weight[j*8:(j+1)*8]),('bias',bias[j*8:(j+1)*8])]:
                name=factor+'_'+suffix
                value=value.detach().clone()
                if factor in train_factors:
                    self.register_parameter(name,nn.Parameter(value))
                else:
                    self.register_buffer(name,value)

    def density_rows(self):
        return (torch.cat([getattr(self,k+'_weight') for k in ['mixture','mean','scale']],0),
                torch.cat([getattr(self,k+'_bias') for k in ['mixture','mean','scale']],0))


class Network(nn.Module):
    def __init__(self, original: nn.Module, selected: dict, mode: str):
        super().__init__()
        if mode not in MODES or len(original.members)!=2:
            raise ValueError('Invalid POR subspace/source')
        if selected.get('source_kinds') != ['target_group']*3:
            raise ValueError('This study expects the verified linear factor-head recipe')
        self.mode=mode
        self.backbone=original.eval().requires_grad_(False)
        self.heads=nn.ModuleList([
            nn.ModuleList([FactorHead(selected['head_state_dicts'][c],f'branches.{e}.',
                                      MODES[mode] if c==0 else ()) for e in range(2)])
            for c in range(3)
        ])
        self.captured={}
        self.handles=[member.main_head.register_forward_pre_hook(self._capture(e))
                      for e,member in enumerate(self.backbone.members)]
        expected=2064*len(MODES[mode])
        assert sum(p.numel() for p in self.parameters() if p.requires_grad)==expected

    def _capture(self,index):
        def hook(_module,inputs):self.captured[index]=inputs[0].detach()
        return hook

    def train(self,mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self,features,relative_depth,valid,auxiliary=False,absolute_depth=None):
        if auxiliary:raise ValueError('No auxiliary restart in the readout study')
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
            outputs.append((gate.transpose(1,2),raw.transpose(1,2).reshape(len(h),-1,3,3,8)))
        self.captured.clear()
        return routed_merge(outputs,logits)

    def export_legacy_heads(self):
        """Compatible with the prior selected-head manifest; only fixed whole targets."""
        result=[]
        for group in self.heads:
            state={}
            for e,head in enumerate(group):
                weight,bias=head.density_rows()
                for name,value in [('gate_weight',head.gate_weight),('gate_bias',head.gate_bias),
                                   ('density_weight',weight),('density_bias',bias)]:
                    state[f'branches.{e}.'+name]=value.detach().cpu().clone()
            result.append(state)
        return result
