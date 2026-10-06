"""Compact POR update-scope ablation: source mixture/scale rows stay fixed."""
import torch
from torch import nn
from compact_five_stage_v100 import CompactReadout,PackedHead

class LocationHead(PackedHead):
    def __init__(self,original,c):
        super().__init__(original,c)
        assert c==0
        self.density_weight.requires_grad_(False);self.density_bias.requires_grad_(False)
        self.mean_weight=nn.Parameter(self.density_weight[8:16].detach().clone())
        self.mean_bias=nn.Parameter(self.density_bias[8:16].detach().clone())
    def packed(self,stem):
        if stem.startswith('density'):
            fixed=getattr(self,stem);mean=self.mean_weight if stem.endswith('weight') else self.mean_bias
            own=torch.cat((fixed[:8],mean,fixed[16:]),0)
            return torch.cat((getattr(self,stem+'_before'),own,getattr(self,stem+'_after')),0)
        return super().packed(stem)

class Network(CompactReadout):
    def __init__(self,master,c=0):
        super().__init__(master,c)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010)
            self.heads=nn.ModuleList([LocationHead(m.main_head,c) for m in self.backbone.members])
    def groups(self):
        linear=[p for h in self.heads for p in (h.mean_weight,h.mean_bias)]
        nonlinear=[p for h in self.heads for n,p in h.named_parameters() if p.requires_grad and n not in ('mean_weight','mean_bias')]+list(self.routes.parameters())
        return [dict(params=linear,lr=3e-5,name='original_target_locations'),dict(params=nonlinear,lr=3e-4,name='combined_new_heads_routes')]

def make_model(stage,master,metadata):
    if stage!='por':raise ValueError('v101 is a single predefined POR scope ablation')
    return Network(master)
