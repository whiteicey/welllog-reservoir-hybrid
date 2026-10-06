"""Fresh SW body/readout with own-task latent injected before final readout."""
import torch
from torch import nn
from torch.nn import functional as F
from sw_raw_well_condition_v297 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,DecisionObjective,Network as Parent
STUDY=ROOT/'results/deep_learning/sw_late_latent_v323_20261002'
OUT=ROOT/'outputs/pure_dl_sw_late_latent_v323_fold5'
STAGES={'late':('SW',1024,False,0.,3e-4)}
CONTROL=ROOT/'outputs/pure_dl_sw_private_latent_v321_fold5_r1/own'
class Network(Parent):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,'well');self.fusion_position=mode
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261321);self.latent_fuse=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,48),nn.GELU(),nn.Linear(48,48))
   nn.init.zeros_(self.latent_fuse[-1].weight);nn.init.zeros_(self.latent_fuse[-1].bias)
  self.own_latent=None
  self.base.output.register_forward_pre_hook(self.capture_own_latent)
  # Correct before inherited fused feature hook, so gate and density both see correction.
  self.fuse.register_forward_hook(self.add_late_latent,prepend=True)
 def capture_own_latent(self,module,args):self.own_latent=args[0].detach()
 def add_late_latent(self,module,args,h):
  lo=h.shape[1]//2-128;hi=lo+256
  z=self.own_latent[:,lo:hi].repeat(1,1,3)
  delta=.25*torch.tanh(self.latent_fuse(z).float())
  return h+F.pad(delta.to(h.dtype),(0,0,lo,h.shape[1]-hi))
 def groups(self):return super().groups()+[dict(params=list(self.latent_fuse.parameters()),lr=3e-4,name='SW_joint_own_latent_late_fusion')]
 def initialize_priors(self,data):
  info=super().initialize_priors(data);n=sum(p.numel() for p in self.parameters() if p.requires_grad);assert n==86865
  info.update(active_parameters=n,new_latent_fusion_parameters=21600,source_only104=True,loaded321_trained_weights=False,own_norm128_repeat3_input=True,central256_in_SW1024=True,fusion_position='final_fuse48_pre_state_capture',historical_control=str(CONTROL),control_original321_own_early_positive=True,only_nonfunctional_unused_donors_and63_side_removed=True,baseline_control_full_state_replay_verified=True,no_equal_trajectory_claim_for_removed_donor_computation=True,auxiliary_count4_same_but_gradient_paths_differ=True,all_fresh_encoder_state_condition_readout_fusion_joint=True)
  return info
 def forward(self,*args,**kwargs):
  self.own_latent=None
  try:return super().forward(*args,**kwargs)
  finally:self.own_latent=None
def make_model(stage,master,metadata):return Network(master,metadata,stage)
