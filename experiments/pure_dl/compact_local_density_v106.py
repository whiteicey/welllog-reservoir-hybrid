"""Fifth training stage for the selected compact local-filter SW architecture."""
import hashlib
import torch
from torch import nn
from compact_five_stage_v100 import ROOT,SOURCE,load_data as original_data
from compact_frequency_v104 import Network as LocalNetwork

STUDY=ROOT/'results/deep_learning/compact_local_density_v106_20261001'
OUT=ROOT/'outputs/pure_dl_compact_local_density_v106_fold5'
PARENT=ROOT/'outputs/pure_dl_compact_frequency_v104_fold5/local/model_best.pt'
STAGES={'density':('SW',1024,True,0.,1e-4)}

class Network(nn.Module):
    def __init__(self,metadata,saved):
        super().__init__();self.c=2;self.length=1024
        self.atom=LocalNetwork(metadata,'local');self.atom.load_state_dict(saved['state_dict'],strict=True);self.atom.requires_grad_(False).eval()
        self.density=LocalNetwork(metadata,'local');self.density.load_state_dict(saved['state_dict'],strict=True)
        self.cached_gate=None
        self.parent_info=dict(checkpoint=str(PARENT.relative_to(ROOT)),sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),best_epoch=saved['best_epoch'],target_score=saved['target_score'],
            scope='selected v104 local own-training-fold model; actual stage4; frozen atom/private full conditional density',
            cached_frozen_atom='once per train batch across two SAM passes; deterministic eval only; no persistent cached input')

    def train(self,mode=True):
        super().train(mode);self.atom.eval()
        if not mode:self.cached_gate=None
        return self

    @torch.no_grad()
    def prepare_batch(self,x,relative,valid,absolute_depth,well_inputs):
        self.cached_gate=self.atom(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)[0]

    def clear_batch(self):self.cached_gate=None

    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        if self.training and self.cached_gate is not None:gate=self.cached_gate
        else:
            with torch.no_grad():gate=self.atom(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)[0]
        _,raw=self.density(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
        return gate,raw

def load_data(stage):return original_data('sw_joint')
def make_model(stage,master,metadata):return Network(metadata,torch.load(PARENT,map_location='cpu',weights_only=False))
