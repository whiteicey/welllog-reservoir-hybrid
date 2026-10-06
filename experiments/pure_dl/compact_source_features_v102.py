"""Frozen source hidden features for a randomly initialized private SW network.

Matched random-encoder control keeps all source preprocessing buffers. No
source predictions are consumed. Source preparation counts as stage one.
"""
import torch
from torch import nn
from torch.nn import functional as F
from compact_five_stage_v100 import ROOT,SOURCE,CompactSW,load_data as original_data
from paired_local_context_router_sequence import Network as Original,retained_indices,calibration_indices

STUDY=ROOT/'results/deep_learning/compact_source_features_v102_20261001'
OUT=ROOT/'outputs/pure_dl_compact_source_features_v102_fold5'
STAGES={name:('SW',1024,True,1.,3e-4) for name in ('random_block256','pretrained_block256')}

class Network(CompactSW):
    def __init__(self,master,metadata,mode):
        super().__init__(metadata)
        # Both candidates construct the same private body and added stem.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261102)
            self.body.stem=nn.Linear(463,96)
        numeric,condition=calibration_indices(master['selected_feature_names'])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261103)
            self.encoder=Original(retain_indices=retained_indices(master['selected_feature_names']),numeric_indices=numeric,conditioning_indices=condition)
        if mode=='pretrained_block256':self.encoder.load_state_dict(master['state_dict'],strict=True)
        elif mode=='random_block256':
            with torch.no_grad():
                for name,buffer in self.encoder.named_buffers():
                    if name in master['state_dict']:buffer.copy_(master['state_dict'][name])
        else:raise ValueError(mode)
        self.encoder.requires_grad_(False).eval();self.feature_mode=mode
        self.captured={};self.feature_cache=None
        self.handles=[member.main_head.register_forward_pre_hook(self.capture(i)) for i,member in enumerate(self.encoder.members)]

    def capture(self,i):
        def hook(module,args):self.captured[i]=args[0].detach()
        return hook

    def train(self,mode=True):
        super().train(mode);self.encoder.eval()
        if not mode:self.feature_cache=None
        return self

    @torch.no_grad()
    def features(self,x,relative,valid,absolute_depth):
        parts=[]
        for lo in range(0,x.shape[1],256):
            hi=lo+256;self.captured.clear()
            for member in self.encoder.members:member(x[:,lo:hi],relative[:,lo:hi],valid[:,lo:hi],absolute_depth=absolute_depth[:,lo:hi])
            hidden=[F.layer_norm(self.captured[i].transpose(1,2).float(),(128,)).to(x.dtype) for i in range(2)]
            parts.append(torch.cat(hidden,-1))
        self.captured.clear();return torch.cat(parts,1).masked_fill(~valid[...,None],0)

    @torch.no_grad()
    def prepare_batch(self,x,relative,valid,absolute_depth):
        # Deterministic frozen features reused across both SAM passes.
        self.feature_cache=self.features(x,relative,valid,absolute_depth)

    def clear_batch(self):self.feature_cache=None

    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None,**kwargs):
        features=self.feature_cache if self.training and self.feature_cache is not None else self.features(x,relative,valid,absolute_depth)
        return super().forward(torch.cat((x,features),-1),relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs,**kwargs)

def load_data(stage):return original_data('sw_joint')
def make_model(stage,master,metadata):return Network(master,metadata,stage)
