"""Fresh POR fullcurve interaction backbone/readout with latent morphology."""
import torch
from por_curve_scope_v302 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent
from sw_latent_morphology_v326 import Morphology
STUDY=ROOT/'results/deep_learning/por_latent_morphology_v328_20261002'
OUT=ROOT/'outputs/pure_dl_por_latent_morphology_v328_fold5'
STAGES={'depth':('POR',256,False,0.,3e-4)}
class Network(Parent):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,'full');self.morphology_mode=mode;self.current_valid=None
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261326);self.morphology=Morphology(mode)
  self.branches[0].register_forward_hook(self.morph_branch)
 def morph_branch(self,module,args,h):return self.morphology(h,self.current_valid)
 def groups(self):return super().groups()+[dict(params=list(self.morphology.parameters()),lr=3e-4,name='POR_joint_latent_morphology_backbone')]
 def initialize_priors(self,data):
  info=super().initialize_priors(data);n=sum(p.numel() for p in self.parameters() if p.requires_grad);assert n==82016
  info.update(active_parameters=n,new_active_parameters=n,new_morphology_parameters=5424,
   morphology_mode=self.morphology_mode,neighborhood_taps=7,only_first_branch_morphology=True,
   initial_morphology_projection_zero=True,original151_only=True,no302_or326_trained_weights_loaded=True,
   mechanism_code_transfer_not_SW_weight_or_feature_transfer=True,
   joint_raw_curve_attention_morphology_readout_one_POR_phase=True,
   historical302_comparison_not_capacity_matched=True,original206_already_contains_range=True,
   hard_winner_gradients_sparse=True,no_sampling_gap_compensation=True)
  return info
 def forward(self,x,relative,valid,**kwargs):
  self.current_valid=valid
  try:return super().forward(x,relative,valid,**kwargs)
  finally:self.current_valid=None
def make_model(stage,master,metadata):return Network(master,metadata,stage)
