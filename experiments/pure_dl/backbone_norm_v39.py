"""Supervised SW normalization-affine adaptation of the retained private backbone.

Inference reproduces v29 exactly at initialization. Existing capture hooks are
replaced with differentiable hooks; no detached/no_grad backbone path is used.
"""
import json
import numpy as np
import torch
from torch import nn
from crosscurve_v32 import Network as ParentBuilder,ROOT,PREVIOUS,reference_prediction
from champion_all_heads_v26 import Evaluator,load_data,TARGETS
from champion_split_route_v29 import shift

STUDY=ROOT/'results/deep_learning/backbone_norm_v39_20260929'
MODES=('decoder_norm','all_norm')

class Network(nn.Module):
    def __init__(self,target,master,mode):
        super().__init__()
        if target!='SW' or mode not in MODES:raise ValueError((target,mode))
        self.target=target;self.c=2;self.mode=mode
        loader=ParentBuilder(target,master,modes=())
        self.current=loader.parent;self.parent_mode=loader.parent_mode;self.choice=loader.choice
        self.current.requires_grad_(False)
        self.current.close();self.current.handles=[]
        self.captured={};self.handles=[]
        for e,member in enumerate(self.current.source.members):
            def capture(_module,args,output,e=e):
                self.captured['h'+str(e)]=args[0]
                self.captured['o'+str(e)]=output
            self.handles.append(member.main_head.register_forward_hook(capture))
        self.handles.append(self.current.source.router[2].register_forward_hook(
            lambda _m,_a,v:self.captured.update(logits=v)))
        selected=[]
        for e,member in enumerate(self.current.source.members):
            for name,layer in member.named_modules():
                if not isinstance(layer,(nn.GroupNorm,nn.LayerNorm)) or name.startswith('main_head.'):continue
                if mode=='decoder_norm' and not name.startswith(('dec0.','dec1.','dec2.')):continue
                for field,p in layer.named_parameters(recurse=False):
                    p.requires_grad_(True);selected.append(f'current.source.members.{e}.{name}.{field}')
        self.trainable_names=set(selected)
        assert self.trainable_names and self.trainable_names=={n for n,p in self.named_parameters() if p.requires_grad}
        self.train(False)

    def train(self,mode=True):
        # Dropout and parent operating mode remain unchanged; only gradients differ.
        super().train(False);self.training=mode;return self

    def forward(self,x,relative,valid,absolute_depth=None,factors=None):
        if factors is not None and tuple(factors)!=(self.mode,):raise ValueError(factors)
        c=self.current;self.captured.clear()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            c.source(x[...,:206].contiguous(),relative,valid,absolute_depth=absolute_depth)
        h=[self.captured['h'+str(e)] for e in range(2)]
        outputs=[self.captured['o'+str(e)] for e in range(2)]
        logits=self.captured['logits'].reshape(*x.shape[:2],3,2)[:,:,2:3]
        self.captured.clear();side=x[...,206:]
        if c.retained is not None:outputs=c.retained.corrected_members(h,outputs,side,valid)
        delta=c.retained_route.correction(h,side,valid)
        density=shift(logits,delta)
        atom=density if c.parent_mode=='joint' else logits
        return {self.mode:c.heads[self.parent_mode](h,outputs,(atom,density),side,valid)}

    def trainable_state(self):
        return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if p.requires_grad}

    def load_trainable_state(self,state):
        assert set(state)==self.trainable_names
        params=dict(self.named_parameters())
        with torch.no_grad():
            for n,v in state.items():params[n].copy_(v.to(params[n].device))

    def frozen_state(self):return {n:v for n,v in self.state_dict().items() if n not in self.trainable_names}

    def close(self):
        for h in self.handles:h.remove()
        self.handles=[];self.captured.clear()
