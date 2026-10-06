"""Fresh joint SW encoder/readout, own versus private POR/PERM input latents."""
import hashlib,json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sw_raw_well_condition_v297 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data as original_data,Evaluator,channel_objective,DecisionObjective,Network as Parent
from por_ranked_probability_v151 import Network as PorSource
from perm_ranked_probability_v154 import Network as PermSource
STUDY=ROOT/'results/deep_learning/sw_private_latent_v321_20261002'
OUT=ROOT/'outputs/pure_dl_sw_private_latent_v321_fold5_r1'
STAGES={a:('SW',1024,False,0.,3e-4) for a in ('own','cross')}
POR=ROOT/'outputs/pure_dl_por_ranked_probability_v151_fold5/joint_rps/model_best.pt'
PERM=ROOT/'outputs/pure_dl_perm_ranked_probability_v154_fold5/perm_conditional/model_best.pt'
POR_SHA='39f2255b8fe61fa74607af027ece1dd352e5e2d0729b9bce95d1424c1b227603'
PERM_SHA='2b355bb8c3736990b05e2726373a4343c574a93ab6ab1a3eda0ef796ac70cf11'
SIDE=ROOT/'results/deep_learning/compact_five_stage_v100_20260930/amplitude.npz'
SIDE_SHA='4935b88a194e825f8794ff3e0125ecad8d5a39a46a9abbf8c69d56e7555df905'
def load_data(stage):
 data,master,metadata=original_data(stage)
 assert hashlib.sha256(SIDE.read_bytes()).hexdigest()==SIDE_SHA
 with np.load(SIDE) as z:side=torch.as_tensor(z['amplitude'],device='cuda')
 assert side.shape==(data['x'].shape[0],63)
 data['x']=torch.cat((data['x'][...,:206],side),-1)
 return data,master,metadata
class Network(Parent):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,'well');self.latent_mode=mode
  for p,h in ((POR,POR_SHA),(PERM,PERM_SHA)):assert hashlib.sha256(p.read_bytes()).hexdigest()==h
  self.por=PorSource(master,'joint_rps');self.por.load_state_dict(torch.load(POR,map_location='cpu',weights_only=False)['state_dict'],strict=True);self.por.requires_grad_(False).eval()
  self.perm=PermSource(master,'perm_conditional');self.perm.load_state_dict(torch.load(PERM,map_location='cpu',weights_only=False)['state_dict'],strict=True);self.perm.requires_grad_(False).eval()
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261321);self.latent_fuse=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,48),nn.GELU(),nn.Linear(48,48))
   nn.init.zeros_(self.latent_fuse[-1].weight);nn.init.zeros_(self.latent_fuse[-1].bias)
  self.captured={};self.current_latent=None;self.current_slice=None
  self.por.context_head.expand.register_forward_pre_hook(self.capture_private('por',True))
  self.perm.context_head.expand.register_forward_pre_hook(self.capture_private('perm',True))
  self.base.output.register_forward_pre_hook(self.capture_private('sw',False))
  native=self.base.forward
  self.base.forward=lambda x,*args,**kwargs:native(x[...,:206],*args,**kwargs)
  self.stem.register_forward_hook(self.add_latent)
 def capture_private(self,name,transpose):
  def hook(module,args):self.captured[name]=args[0].detach().transpose(1,2) if transpose else args[0].detach()
  return hook
 def train(self,mode=True):
  super().train(mode);self.por.eval();self.perm.eval();return self
 def groups(self):return super().groups()+[dict(params=list(self.latent_fuse.parameters()),lr=3e-4,name='SW_joint_private_latent_fusion')]
 def initialize_priors(self,data):
  info=super().initialize_priors(data);n=sum(p.numel() for p in self.parameters() if p.requires_grad);assert n==86865
  info.update(latent_mode=self.latent_mode,active_parameters=n,new_latent_fusion_parameters=21600,private_latent_channels=dict(POR=256,PERM=128,SW=128),POR_source_sha=POR_SHA,PERM_source_sha=PERM_SHA,side_sha=SIDE_SHA,donor_window=256,SW_window=1024,aligned_central256=True,own128_repeat3_control=True,all_source_encoders_frozen=True,donor_predictions_not_used=True,no_trained297_or_other_repairs=True,one_SW_added_phase=True,all_fresh_body_state_condition_readout_fusion_joint=True,own_and_cross_same_active_capacity_but_different_feature_rank=True)
  return info
 def add_latent(self,module,args,h):
  lo,hi=self.current_slice
  latent=self.current_latent if self.latent_mode=='cross' else self.captured['sw'][:,lo:hi].repeat(1,1,3)
  assert latent.shape==(*h.shape[:1],256,384)
  delta=.25*torch.tanh(self.latent_fuse(latent).float())
  return h+F.pad(delta.to(h.dtype),(0,0,lo,h.shape[1]-hi))
 def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
  lo=x.shape[1]//2-128;hi=lo+256;self.current_slice=(lo,hi);self.captured.clear()
  # Both arms compute donors; only hidden inputs to fusion differ. Original256 source contract.
  with torch.no_grad():
   kw=dict(absolute_depth=absolute_depth[:,lo:hi] if absolute_depth is not None else None,well_inputs=well_inputs)
   self.por(x[:,lo:hi],relative[:,lo:hi],valid[:,lo:hi],**kw)
   self.perm(x[:,lo:hi],relative[:,lo:hi],valid[:,lo:hi],**kw)
  self.current_latent=torch.cat((self.captured.pop('por'),self.captured.pop('perm')),-1)
  try:return super().forward(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
  finally:self.current_latent=None;self.current_slice=None;self.captured.clear()
def make_model(stage,master,metadata):return Network(master,metadata,stage)
