"""SW fresh joint raw encoder/readout plus central nine-curve interaction."""
import torch
from torch.nn import functional as F
from sw_latent_morphology_v326 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,DecisionObjective,Network as MorphParent
from sw_raw_well_condition_v297 import Network as FilmParent
from por_curve_scope_v302 import CurveInteraction
STUDY=ROOT/'results/deep_learning/sw_curve_interaction_v344_20261002'
OUT=ROOT/'outputs/pure_dl_sw_curve_interaction_v344_fold5'
STAGES={s:('SW',1024,False,0.,3e-4) for s in ('full','grouped')}
class Network(MorphParent):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,'depth');self.interaction_mode=mode
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261302);self.curve_interaction=CurveInteraction(mode)
  self.current_curve_correction=None
  # Registered after parent's FiLM and own-latent hooks; reused by three stems.
  self.stem.register_forward_hook(self.add_curve_correction)
 def add_curve_correction(self,module,args,h):
  assert self.current_curve_correction is not None
  return h+self.current_curve_correction.to(h.dtype)
 def groups(self):return super().groups()+[dict(params=list(self.curve_interaction.parameters()),lr=3e-4,name='SW_curve_interaction_backbone')]
 def initialize_priors(self,data):
  info=FilmParent.initialize_priors(self,data)
  n=sum(p.numel() for p in self.parameters() if p.requires_grad);extra=sum(p.numel() for p in self.curve_interaction.parameters())
  assert n==111313 and extra==19024
  info.update(active_parameters=n,new_active_parameters=n,new_curve_parameters=extra,new_latent_fusion_parameters=21600,new_morphology_parameters=5424,morphology_mode='depth',neighborhood_taps=7,own_norm128_repeat3_input=True,zero_morphology_projection=True,original104_only=True,no326_or302_trained_weights=True,curve_interaction_scope=self.interaction_mode,curve_tokens=9,token_width=32,attention_blocks=2,heads=4,FFN=64,zero_curve_bridge=True,central256_curve_in_original1024=True,curve_indices=self.curve_indices.tolist(),all_fresh_encoder_state_density_film_latent_morph_curve_joint=True,one_SW_added_phase=True,physical_parameter_matched_not_effective_function_capacity=True,fusion_position='stem after FiLM and ownlatent, before three branches')
  return info
 def forward(self,x,relative,valid,**kwargs):
  lo=x.shape[1]//2-128;hi=lo+256
  z=self.curve_interaction(x[:,lo:hi].index_select(-1,self.curve_indices))*valid[:,lo:hi,None]
  self.current_curve_correction=F.pad(z,(0,0,lo,x.shape[1]-hi))
  try:return super().forward(x,relative,valid,**kwargs)
  finally:self.current_curve_correction=None

def make_model(stage,master,metadata):return Network(master,metadata,stage)
