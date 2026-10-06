"""Private POR: capacity-near TCN104 versus local+latent64, same t3 head.

New network weights are random. Original training-fold target statistics and
256-bin/800-action decoder metadata are reused, never the old network weights.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from independent_sw_scratch_v10 import TemporalBlock
from architecture_zoo_v15 import LatentMemory, clean
from por_student3_sequence import por_probability

MODES = ('tcn104', 'latent64')


class Network(nn.Module):
    def __init__(self, mode: str, metadata: dict):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.width = 104 if mode == 'tcn104' else 64
        w = self.width
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010)
            self.stem = nn.Linear(207, w)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261027)
            self.blocks = (nn.ModuleList([TemporalBlock(w,d) for d in (1,2,4,8)]) if mode == 'tcn104'
                else nn.ModuleList([TemporalBlock(w,1), LatentMemory(w,16), TemporalBlock(w,2)]))
        self.norm = nn.LayerNorm(w)
        self.output = nn.Linear(w,25)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer('centers',torch.as_tensor(metadata['centers'],dtype=torch.float32).clone())
        self.register_buffer('density_edges',torch.as_tensor(metadata['density_edges'],dtype=torch.float32).clone())
        self.target_median = float(metadata['target_median'])
        self.target_scale = float(metadata['target_scale'])
        self.metadata = metadata
        with torch.no_grad():
            rate = float(metadata['training_default_rate'])
            self.output.bias[0] = math.log(rate/(1-rate))
            self.output.bias[9:17].copy_(torch.as_tensor(metadata['initial_means'],dtype=torch.float32))
        assert self.centers.numel()==256 and self.density_edges.numel()==257

    def forward(self,x,relative,valid):
        inp=clean(torch.cat((x.float(),relative[...,None].float()),dim=-1),valid)
        h=clean(F.gelu(self.stem(inp)),valid)
        for block in self.blocks:
            h=block(h,valid)
        o=clean(self.output(self.norm(h)).float(),valid)
        return o[...,0],o[...,1:].reshape(*o.shape[:2],3,8)

    @staticmethod
    def distribution_parameters(raw):
        return raw[...,0,:].log_softmax(-1),raw[...,1,:],F.softplus(raw[...,2,:])+.025

    @staticmethod
    def log_components(raw,z):
        lw,mu,sd=Network.distribution_parameters(raw.float())
        # df3 with Student-t scale=sd/sqrt(3); sd is the actual std.
        logdensity=math.log(2/math.pi)-sd.log()-2*torch.log1p(((z[...,None]-mu)/sd).square())
        return -(lw+logdensity).logsumexp(-1)

    @torch.no_grad()
    def masses(self,raw):
        lw,mu,sd=self.distribution_parameters(raw.float())
        return por_probability(lw.exp(),mu,sd,self.density_edges)

    def loss(self,output,z,default,mask):
        gate,raw=output
        g=gate[mask].float();d=default[mask]
        zz=z[mask].float().masked_fill(d,0)
        nll=self.log_components(raw[mask],zz).masked_fill(d,0)
        bce=F.binary_cross_entropy_with_logits(g,d.float(),reduction='none')
        return (bce+nll).mean()/3

    @torch.no_grad()
    def mechanism_statistics(self):
        result={'mode':self.mode,'width':self.width,'all_parameters_trainable':True,'head':'8xStudentT_df3+default_gate'}
        if self.mode=='latent64':
            result['latent_norm']=float(self.blocks[1].latents.float().norm())
        return result
