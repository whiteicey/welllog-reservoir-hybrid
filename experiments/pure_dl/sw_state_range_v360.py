"""One finite positive-mechanism state residual range expansion, full fresh joint training."""
import torch
from torch.nn import functional as F
from sw_dynamic_cross_spectrum_v351 import ROOT,SOURCE,PARENT,load_data,Evaluator,DecisionObjective,channel_objective,Network as Parent
from sw_direct_component_v275 import Network as RawNetwork
STUDY=ROOT/'results/deep_learning/sw_state_range_v360_20261002'
OUT=ROOT/'outputs/pure_dl_sw_state_range_v360_fold5'
STAGES={'bound8':('SW',1024,False,0.,3e-4)}
class Network(Parent):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,'aligned');self.state_bound=8.;self.last_patch_abs_max=0.;self.last_patch_abs_gt4_fraction=0.
 def initialize_priors(self,data):
  info=super().initialize_priors(data)
  info.update(new_gate_bound=8.,reference_gate_bound=4.,positive289_bound2_to4_whole_gain=.001349683147992664,
   positive289_bound2_to4_CONT_gain=.00756562554814,no351_trained_weights=True,original104_only=True,
   all_fresh_encoder_density_state_film_latent_morph_spectrum_joint=True,one_SW_added_phase=True,
   hypothesis='larger state residual reach may help native104 confidenterrors; restriction not diagnosed',
   last_batch_input_only_patch_diagnostic=True,no_gate_width_or_LR_grid=True)
  return info
 def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
  if well_inputs is None:raise ValueError('Same-well input bank required')
  lo=x.shape[1]//2-128;hi=lo+256
  z=self.cross_spectrum(x[:,lo:hi].index_select(-1,self.curve_indices),valid[:,lo:hi])
  self.current_spectrum_correction=F.pad(z,(0,0,lo,x.shape[1]-hi))
  self.own_latent=None;self.current_valid=valid;self.state_feature=None
  context=well_inputs if self.conditioning_mode=='well' else self.raw_reference_inputs[None].expand(x.shape[0],-1,-1)
  self.current_modulation=self.condition(context.index_select(-1,self.curve_indices))
  try:
   # Same275 forward and registered297/326/351 hooks as parent, explicit289 gate with bound8.
   gate,mass=RawNetwork.forward(self,x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
   assert self.state_feature is not None
   patch=self.gate_patch(self.state_feature).float()
   if self.training:
    self.last_patch_abs_max=float(patch[valid].detach().abs().max())
    self.last_patch_abs_gt4_fraction=float((patch[valid].detach().abs()>4).float().mean())
   return gate.float()+self.state_bound*torch.tanh(patch/self.state_bound),mass
  finally:
   self.current_spectrum_correction=None;self.own_latent=None;self.current_valid=None;self.state_feature=None;self.current_modulation=None

def make_model(stage,master,metadata):return Network(master,metadata,stage)
