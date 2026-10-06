"""Grow internal nonlinear widths while retaining the actual v5/v8 POR function.

Stage channels/depth/attention/decoder stay fixed. This is internal FFN width
scaling, not uniform widening of every network dimension. New output columns
are zero; split evaluation preserves the old arithmetic at initialization.
"""
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from independent_channels_v8 import Network as PrivateNetwork
from blur_ungated_sequence import UngatedBlock

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/champion_scaling_v21_20260928'
REFERENCE=ROOT/'outputs/pure_dl_independent_v8_selected_fold5'
PRIVATE=ROOT/'outputs/pure_dl_independent_v8_POR_dual80_fold5/model_best.pt'
HEADS=ROOT/'outputs/pure_dl_dual8222_v5_location_point3280_fold5/readout_checkpoint.pt'
MODES={'base1':1,'ffn2':2,'ffn4':4}

class ExpandedBlock(nn.Module):
    def __init__(self,old,factor):
        super().__init__()
        self.norm=old.norm;self.depthwise=old.depthwise
        self.expand=old.expand;self.project=old.project
        self.dropout=old.dropout;self.gain=old.gain
        c=old.expand.in_channels;h=old.expand.out_channels
        assert h==c and factor>1
        self.extra_expand=nn.Conv1d(c,(factor-1)*h,1)
        self.extra_project=nn.Conv1d((factor-1)*h,c,1,bias=False)
        nn.init.zeros_(self.extra_project.weight)
        self.factor=factor
    def forward(self,x):
        z=self.depthwise(self.norm(x))
        original=self.project(self.dropout(.5*F.gelu(self.expand(z))))
        additional=self.extra_project(self.dropout(.5*F.gelu(self.extra_expand(z))))
        return x+self.gain*(original+additional)

class ExpandedLocation(nn.Module):
    def __init__(self,old,factor):
        super().__init__()
        self.reduce=old.reduce;self.output=old.output
        self.context=old.context;self.bound_fraction=old.bound_fraction
        assert self.context is None
        old_width=self.reduce.out_channels
        self.extra_reduce=nn.Conv1d(self.reduce.in_channels,(factor-1)*old_width,1)
        self.extra_output=nn.Conv1d((factor-1)*old_width,self.output.out_channels,1,bias=False)
        nn.init.zeros_(self.extra_output.weight)
    def forward(self,h,valid,raw_scale):
        original=F.gelu(self.reduce(h)).masked_fill(~valid[:,None],0)
        extra=F.gelu(self.extra_reduce(h)).masked_fill(~valid[:,None],0)
        proposal=self.output(original).float()+self.extra_output(extra).float()
        bound=self.bound_fraction*(F.softplus(raw_scale.detach().float())+.025)
        return (bound*torch.tanh(proposal/bound)).masked_fill(~valid[:,None],0)

class Network(PrivateNetwork):
    def __init__(self,master,previous,champion,mode):
        if mode not in MODES:raise ValueError(mode)
        super().__init__(master,previous,'POR','dual')
        self.load_state_dict(champion['state_dict'],strict=True)
        self.mode=mode;self.factor=MODES[mode]
        initial_count=sum(p.numel() for p in self.parameters())
        changes=[]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261031)
            for prefix,parent in list(self.named_modules()):
                for name,child in list(parent.named_children()):
                    if isinstance(child,UngatedBlock):
                        path=prefix+'.'+name
                        changes.append(dict(path=path,stage_width=child.expand.in_channels,
                            old_internal=child.expand.out_channels,new_internal=child.expand.out_channels*self.factor))
                        if self.factor>1:setattr(parent,name,ExpandedBlock(child,self.factor))
            assert len(changes)==18, len(changes)
            for member in self.members:
                if self.factor>1:member.main_head.residual=ExpandedLocation(member.main_head.residual,self.factor)
        self.scaling_manifest=dict(mode=mode,internal_width_factor=self.factor,stage_widths_unchanged=True,
            attention_unchanged=True,blocks=changes,location_hidden_width=32*self.factor,
            base_parameters=initial_count,parameters=sum(p.numel() for p in self.parameters()),
            extra_parameters=sum(p.numel() for n,p in self.named_parameters() if '.extra_' in n),
            dropout_modules=[type(m.dropout).__name__ for m in self.modules() if isinstance(m,(UngatedBlock,ExpandedBlock))])


def make_model(mode,master=None):
    from train_dual8222_heads80 import SOURCE
    master=master if master is not None else torch.load(SOURCE,map_location='cpu',weights_only=False)
    previous=torch.load(HEADS,map_location='cpu',weights_only=False)
    champion=torch.load(PRIVATE,map_location='cpu',weights_only=False)
    return Network(master,previous,champion,mode)


def load_data():
    from train_dual8222_heads80 import SOURCE
    data=torch.load(STUDY/'prepared_data.pt',map_location='cpu',weights_only=False)
    for k,v in list(data.items()):
        if torch.is_tensor(v):data[k]=v.cuda()
    master=torch.load(SOURCE,map_location='cpu',weights_only=False)
    return data,master
