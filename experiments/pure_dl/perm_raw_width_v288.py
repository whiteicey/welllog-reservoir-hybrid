"""Larger joint raw feature encoder under proven direct component supervision."""
import torch
from torch import nn
from perm_direct_share_v283 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent
STUDY=ROOT/'results/deep_learning/perm_raw_width_v288_20261002'
OUT=ROOT/'outputs/pure_dl_perm_raw_width_v288_fold5'
STAGES={'width96':('PERM',256,False,0.,3e-4)}
class ScaleMLP(nn.Module):
    def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(96,192),nn.GELU(),nn.Linear(192,96))
    def forward(self,h):return h+self.net(h)
class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,mode)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261271);self.stem=nn.Sequential(nn.Linear(207,96),nn.GELU());self.branches=nn.ModuleList([ScaleMLP() for _ in range(3)]);self.fuse=nn.Sequential(nn.Linear(288,96),nn.GELU());self.readout=nn.Linear(96,256)
    def initialize_priors(self,data):
        info=super().initialize_priors(data);info.update(new_active_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),raw_encoder_width=96,reference_width=48,reference='283alpha30 width48',no283_or281trained_weights_loaded=True,same_seed_but_different_tensor_shapes_initialization=True,initial_final_prediction_same_prior_alpha0=True,new_encoder_readout_both_scaled=True)
        return info
def make_model(stage,master,metadata):return Network(master,metadata,stage)
