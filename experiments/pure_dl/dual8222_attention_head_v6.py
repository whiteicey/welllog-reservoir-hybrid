"""Deeper point and attention readouts on the verified v5 dual-expert predictor.

Only POR location parameters change. All arms share trainable linear locations;
old v5 residuals and all gates/mixture/scales/PERM/SW are frozen. The new branch
uses both internal expert feature vectors, with no external model or well data.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from dual8222_location_readout_v5 import Network as PreviousNetwork
from paired_detached_router_sequence import routed_merge

MODES = {'linear':('mean',), 'deep_mlp':('mean','deep_joint_residual'),
         'attention':('mean','local_attention_joint_residual')}


class TokenMixer(nn.Module):
    """Two near-parameter-budget alternatives. Input/output: B,L,64."""
    def __init__(self, attention: bool):
        super().__init__()
        self.attention = attention
        if attention:
            self.qkv = nn.Linear(64, 192)
            self.proj = nn.Linear(64, 64)
            self.relative_bias = nn.Parameter(torch.zeros(4,33))
        else:
            self.point = nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Linear(128,64))

    def forward(self, x, valid):
        if not self.attention:
            return self.point(x).masked_fill(~valid[...,None],0)
        b,l,_ = x.shape
        q,k,v = self.qkv(x).reshape(b,l,3,4,16).permute(2,0,3,1,4).unbind(0)
        position = torch.arange(l,device=x.device)
        offset = position[None,:]-position[:,None]
        window = offset.abs() <= 16
        allowed = window[None] & valid[:,None,:] & valid[:,:,None]
        # Invalid queries get only their own dummy key, then the output is zeroed.
        # This avoids all-masked softmax rows without exposing padding to real queries.
        allowed = allowed | ((~valid)[:,:,None] & torch.eye(l,device=x.device,dtype=torch.bool)[None])
        bias = self.relative_bias[:,(offset+16).clamp(0,32)]
        # Explicit float32 attention avoids backend changes across inference/training.
        with torch.autocast(device_type=x.device.type, enabled=False):
            score = (q.float() @ k.float().transpose(-1,-2))/math.sqrt(16)
            score = score+bias.float()[None]
            score = score.masked_fill(~allowed[:,None],float('-inf'))
            weights = score.softmax(-1)
            mixed = (weights @ v.float()).transpose(1,2).reshape(b,l,64)
        return self.proj(mixed.to(x.dtype)).masked_fill(~valid[...,None],0)


class ReadoutBlock(nn.Module):
    def __init__(self, attention: bool):
        super().__init__()
        # Keep FFN/norm initializations identical despite different mixer construction.
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(64)
        self.ffn = nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Linear(128,64))
        with torch.random.fork_rng(devices=[]):
            self.mixer = TokenMixer(attention)

    def forward(self,x,valid):
        x = (x+self.mixer(self.norm1(x),valid)).masked_fill(~valid[...,None],0)
        return (x+self.ffn(self.norm2(x))).masked_fill(~valid[...,None],0)


class RichLocationReadout(nn.Module):
    """Joint two-expert input B,L,256; two blocks; corrections B,L,16."""
    def __init__(self,attention: bool):
        super().__init__()
        self.input = nn.Linear(256,64)
        self.blocks = nn.ModuleList([ReadoutBlock(attention) for _ in range(2)])
        self.output_norm = nn.LayerNorm(64)
        self.output = nn.Linear(64,16)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self,features,valid):
        x = self.input(features).masked_fill(~valid[...,None],0)
        for block in self.blocks:
            x = block(x,valid)
        return self.output(self.output_norm(x)).float().masked_fill(~valid[...,None],0)


class Network(PreviousNetwork):
    def __init__(self,original,previous_checkpoint,mode):
        if mode not in MODES:
            raise ValueError('Unknown v6 head mode: '+str(mode))
        if previous_checkpoint['readout']['mode'] != 'point32':
            raise ValueError('Expected verified v5 point32 source')
        super().__init__(original,previous_checkpoint['initialization_manifest'],'point32')
        self.heads.load_state_dict(previous_checkpoint['readout']['head_state_dict'],strict=True)
        self.requires_grad_(False)
        for head in self.heads[0]:
            head.mean_weight.requires_grad_(True)
            head.mean_bias.requires_grad_(True)
        self.mode = mode
        if mode != 'linear':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20261010)
                self.heads[0][0].rich_location = RichLocationReadout(attention=mode=='attention')

    def forward(self,features,relative_depth,valid,auxiliary=False,absolute_depth=None):
        if auxiliary:
            raise ValueError('No auxiliary loss restart')
        self.captured.clear()
        with torch.no_grad():
            for member in self.backbone.members:
                member(features,relative_depth,valid,absolute_depth=absolute_depth)
            logits = self.backbone.route_logits(features,valid)
        proposal = None
        if self.mode != 'linear':
            joined = torch.cat([self.captured[0],self.captured[1]],dim=1).transpose(1,2)
            proposal = self.heads[0][0].rich_location(joined,valid)
        outputs=[]
        for e in range(2):
            h=self.captured[e]
            heads=[self.heads[c][e] for c in range(3)]
            gate=F.conv1d(h,torch.cat([head.gate_weight for head in heads]),torch.cat([head.gate_bias for head in heads]))
            rows=[head.density_rows() for head in heads]
            raw=F.conv1d(h,torch.cat([v[0] for v in rows]),torch.cat([v[1] for v in rows]))
            raw=raw.transpose(1,2).reshape(len(h),-1,3,3,8).float()
            por=raw[:,:,0]
            old_delta=heads[0].location_residual(h,valid,por[:,:,2].transpose(1,2)).transpose(1,2)
            mean=por[:,:,1]+old_delta
            if proposal is not None:
                bound=.25*(F.softplus(por[:,:,2].detach())+.025)
                delta=bound*torch.tanh(proposal[:,:,e*8:(e+1)*8]/bound)
                mean=mean+delta.masked_fill(~valid[...,None],0)
            updated=torch.stack([por[:,:,0],mean,por[:,:,2]],dim=2)
            raw=torch.cat([updated.unsqueeze(2),raw[:,:,1:]],dim=2)
            outputs.append((gate.transpose(1,2),raw))
        self.captured.clear()
        return routed_merge(outputs,logits)

    def export_readout(self):
        return {'format_version':1,'model_module':'dual8222_attention_head_v6','mode':self.mode,
                'head_state_dict':{k:v.detach().cpu().clone() for k,v in self.heads.state_dict().items()},
                'requires_initialization_manifest':True,'new_residual_bound_fraction':.25,
                'existing_v5_residual_frozen':True}
