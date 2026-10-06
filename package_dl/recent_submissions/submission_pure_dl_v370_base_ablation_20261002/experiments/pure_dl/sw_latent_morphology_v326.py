"""Joint SW encoder/readout with learned max/min latent neighborhood residual."""
import torch
from torch import nn
from torch.nn import functional as F
from sw_raw_well_condition_v297 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,DecisionObjective,Network as Parent
STUDY=ROOT/'results/deep_learning/sw_latent_morphology_v326_20261002'
OUT=ROOT/'outputs/pure_dl_sw_latent_morphology_v326_fold5'
STAGES={a:('SW',1024,False,0.,3e-4) for a in ('depth','channel')}
class Morphology(nn.Module):
 def __init__(self,mode):
  super().__init__();self.mode=mode;self.norm=nn.LayerNorm(48)
  self.dilation=nn.Parameter(torch.randn(48,7)*.01);self.erosion=nn.Parameter(torch.randn(48,7)*.01)
  self.projection=nn.Linear(96,48);nn.init.zeros_(self.projection.weight);nn.init.zeros_(self.projection.bias)
 def forward(self,h,valid):
  with torch.autocast(device_type=h.device.type,enabled=False):
   z=self.norm(h.float());maximum=minimum=None;rows=torch.arange(h.shape[1],device=h.device)
   for tap,offset in enumerate(range(-3,4)):
    if self.mode=='depth':
     ix=rows+offset;in_range=(ix>=0)&(ix<h.shape[1]);ix=ix.clamp(0,h.shape[1]-1)
     v=z.index_select(1,ix);m=valid.index_select(1,ix)&in_range[None]
    else:v=z.roll(-offset,dims=-1);m=valid
    d=(v+self.dilation[:,tap]).masked_fill(~m.unsqueeze(-1),-float('inf'))
    a=(v-self.erosion[:,tap]).masked_fill(~m.unsqueeze(-1),float('inf'))
    maximum=d if maximum is None else torch.maximum(maximum,d)
    minimum=a if minimum is None else torch.minimum(minimum,a)
   maximum=torch.where(valid.unsqueeze(-1),maximum,torch.zeros_like(maximum))
   minimum=torch.where(valid.unsqueeze(-1),minimum,torch.zeros_like(minimum))
   correction=.25*torch.tanh(self.projection(torch.cat((maximum,minimum),-1)))
   return ((h.float()+correction)*valid.unsqueeze(-1)).to(h.dtype)
class Network(Parent):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,'well');self.morphology_mode=mode;self.current_valid=None
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261321);self.latent_fuse=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,48),nn.GELU(),nn.Linear(48,48))
   nn.init.zeros_(self.latent_fuse[-1].weight);nn.init.zeros_(self.latent_fuse[-1].bias)
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261326);self.morphology=Morphology(mode)
  self.own_latent=None;self.base.output.register_forward_pre_hook(self.capture_own_latent)
  # Existing train-only well FiLM stem hook is already registered; own fusion follows it.
  self.stem.register_forward_hook(self.add_early_latent)
  self.branches[0].register_forward_hook(self.morph_branch)
 def capture_own_latent(self,module,args):self.own_latent=args[0].detach()
 def add_early_latent(self,module,args,h):
  lo=h.shape[1]//2-128;hi=lo+256;z=self.own_latent[:,lo:hi].repeat(1,1,3)
  delta=.25*torch.tanh(self.latent_fuse(z).float())
  return h+F.pad(delta.to(h.dtype),(0,0,lo,h.shape[1]-hi))
 def morph_branch(self,module,args,h):return self.morphology(h,self.current_valid)
 def groups(self):return super().groups()+[dict(params=list(self.latent_fuse.parameters()),lr=3e-4,name='SW_joint_own_latent_early_fusion'),dict(params=list(self.morphology.parameters()),lr=3e-4,name='SW_joint_latent_morphology_backbone')]
 def initialize_priors(self,data):
  info=super().initialize_priors(data);n=sum(p.numel() for p in self.parameters() if p.requires_grad);assert n==92289
  info.update(active_parameters=n,new_active_parameters=n,new_latent_fusion_parameters=21600,new_morphology_parameters=5424,
   morphology_mode=self.morphology_mode,neighborhood_taps=7,only_first_branch_morphology=True,
   zero_output_projection=True,SE_initial_std=.01,morphology_seed=20261326,
   source_only104=True,no297_or321_or323_trained_weights_loaded=True,own_norm128_repeat3_input=True,
   central256_in_SW1024=True,fusion_position='stem_after_well_FiLM_before_three_branches',
   all_fresh_encoder_state_condition_readout_fusion_morphology_joint=True,
   primary_control='same max/min parameters/init, real depth neighbors versus samepoint cyclic latent channel neighbors',
   original206_already_contains_range=True,no_equal_effective_function_capacity_claim=True,
   hard_winner_gradients_sparse=True,no_sampling_gap_compensation=True)
  return info
 def forward(self,x,relative,valid,**kwargs):
  self.own_latent=None;self.current_valid=valid
  try:return super().forward(x,relative,valid,**kwargs)
  finally:self.own_latent=None;self.current_valid=None
def make_model(stage,master,metadata):return Network(master,metadata,stage)
