"""One compact full fresh SW encoder/readout width96 scaling candidate."""
import torch
from torch import nn
from sw_latent_morphology_v326 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,DecisionObjective,Network as Parent,Morphology
from sw_raw_well_condition_v297 import BankCondition,Network as WellParent
STUDY=ROOT/'results/deep_learning/sw_morphology_width_v329_20261002'
OUT=ROOT/'outputs/pure_dl_sw_morphology_width_v329_fold5'
STAGES={'wide':('SW',1024,False,0.,3e-4)}
class ScaleMLP(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(96,192),nn.GELU(),nn.Linear(192,96))
 def forward(self,h):return h+self.net(h)
class WideCondition(BankCondition):
 def __init__(self):
  nn.Module.__init__(self)
  self.point=nn.Sequential(nn.Linear(9,32),nn.GELU(),nn.Linear(32,32),nn.GELU())
  self.code=nn.Sequential(nn.Linear(96,32),nn.GELU(),nn.Linear(32,192))
  nn.init.zeros_(self.code[-1].weight);nn.init.zeros_(self.code[-1].bias)
class WideMorphology(Morphology):
 def __init__(self,mode):
  nn.Module.__init__(self);self.mode=mode;self.norm=nn.LayerNorm(96)
  self.dilation=nn.Parameter(torch.randn(96,7)*.01);self.erosion=nn.Parameter(torch.randn(96,7)*.01)
  self.projection=nn.Linear(192,96);nn.init.zeros_(self.projection.weight);nn.init.zeros_(self.projection.bias)
class Network(Parent):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,'depth')
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261271)
   self.stem=nn.Sequential(nn.Linear(207,96),nn.GELU());self.branches=nn.ModuleList([ScaleMLP() for _ in range(3)])
   self.fuse=nn.Sequential(nn.Linear(288,96),nn.GELU());self.readout=nn.Linear(96,256)
   self.gate_patch=nn.Linear(96,1);nn.init.zeros_(self.gate_patch.weight);nn.init.zeros_(self.gate_patch.bias)
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261297);self.condition=WideCondition()
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261321);self.latent_fuse=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,96),nn.GELU(),nn.Linear(96,96))
   nn.init.zeros_(self.latent_fuse[-1].weight);nn.init.zeros_(self.latent_fuse[-1].bias)
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261326);self.morphology=WideMorphology('depth')
  # Replaced modules lose their old hooks: restore the exact single-branch early-fusion order.
  self.stem.register_forward_hook(self.modulate_stem);self.stem.register_forward_hook(self.add_early_latent)
  self.branches[0].register_forward_hook(self.morph_branch);self.fuse.register_forward_hook(self.capture)
 def initialize_priors(self,data):
  info=WellParent.initialize_priors(self,data)
  count=sum(p.numel() for p in self.parameters() if p.requires_grad);assert count==262017
  info.update(active_parameters=count,new_active_parameters=count,new_latent_fusion_parameters=47040,
   new_morphology_parameters=20064,new_condition_parameters=10816,new_gate_readout_parameters=97,
   fresh_hidden_width=96,MLP_inner_width=192,bank_point_width=32,prior_norm_input_width=128,
   source_only104=True,no326_or327_trained_weights_loaded=True,original104_frozen_parameters=845913,
   early_own_latent_fusion=True,only_first_branch_morphology=True,depth_kernel7_unchanged=True,
   all_fresh_encoder_state_condition_readout_fusion_morphology_joint=True,
   historical326_capacity_not_matched=True,initial_predictor_prior_only_alpha_zero=True,
   new_encoder_initial_function_not_48wide_match=True,
   primary_control='Historical326width48 natural27; predefinedbestthrough27, laterfullbest not equalworkgain',
   one_SW_added_phase=True,width_grid=False)
  return info
def make_model(stage,master,metadata):return Network(master,metadata,stage)
