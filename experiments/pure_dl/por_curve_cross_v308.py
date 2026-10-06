"""Preserve fresh full-curve encoder; add a zero-initialized low-rank interaction."""
import torch
from torch import nn
from por_curve_scope_v302 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent
STUDY=ROOT/'results/deep_learning/por_curve_cross_v308_20261002'
OUT=ROOT/'outputs/pure_dl_por_curve_cross_v308_fold5'
STAGES={a:('POR',256,False,0.,3e-4) for a in ('cross','additive')}

class ConditionalCross(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode;self.norm=nn.LayerNorm(48)
        self.down=nn.ModuleList([nn.Linear(48,8,bias=False) for _ in range(2)])
        self.up=nn.ModuleList([nn.Linear(8,48,bias=False) for _ in range(2)])
        self.bias=nn.ParameterList([nn.Parameter(torch.zeros(48)) for _ in range(2)])
        for up in self.up:nn.init.zeros_(up.weight)
    def forward(self,h):
        anchor=self.norm(h)
        for down,up,bias in zip(self.down,self.up,self.bias):
            delta=up(down(self.norm(h)))+bias
            h=h+.1*(anchor*delta if self.mode=='cross' else delta)
        return h

class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,'full');self.cross_mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261308);self.conditional_cross=ConditionalCross(mode)
        # Parent's first hook adds curve correction. This second hook operates on it.
        self.stem.register_forward_hook(self.add_cross)
    def add_cross(self,module,args,result):return self.conditional_cross(result)
    def groups(self):
        return super().groups()+[dict(params=list(self.conditional_cross.parameters()),lr=3e-4,name='POR_conditioned_lowrank_backbone')]
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(raw_encoder_preserved_from302=True,new_cross_parameters=1728,new_active_parameters=78320,new_interaction_parameters=19024,lowrank_mode=self.cross_mode,lowrank_rank=8,lowrank_blocks=2,lowrank_scale=.1,anchor_and_update_inputs_shared_LN=True,normalization_active_both_arms=True,zero_initial_cross_preserves_full302_hidden_function=True,no302_or306_trained_weights=True,only_original151_checkpoint=True,only_scope_differs_in_matched_pair=False,registered_and_active_capacity_same_pair=True,comparison='Same fullcurve encoder; only anchor multiply versusadditive differs')
        return info

def make_model(stage,master,metadata):return Network(master,metadata,stage)
