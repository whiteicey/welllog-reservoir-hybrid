"""Positive336 body plus zero-initial local cross-curve path residual."""
import torch
from torch import nn
from torch.nn import functional as F
from perm_low_rank_cross_v336 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent
STUDY=ROOT/'results/deep_learning/perm_path_residual_v358_20261002'
OUT=ROOT/'outputs/pure_dl_perm_path_residual_v358_fold5'
STAGES={a:('PERM',256,False,0.,3e-4) for a in ('area','variation')}
class PathResidual(nn.Module):
 def __init__(self,mode):
  super().__init__();self.mode=mode;self.bridge=nn.Linear(36,48);nn.init.zeros_(self.bridge.weight);nn.init.zeros_(self.bridge.bias)
  self.register_buffer('pairs',torch.triu_indices(9,9,1))
 def forward(self,curves,valid):
  with torch.autocast(device_type=curves.device.type,enabled=False):
   bounded=torch.tanh(curves.float()/3)
   path=F.pad(bounded.transpose(1,2),(8,8)).transpose(1,2).unfold(1,17,1).permute(0,1,3,2)
   v=F.pad(valid,(8,8),value=False).unfold(1,17,1);complete=v.all(-1)
   dx=path[:,:,1:]-path[:,:,:-1]
   if self.mode=='area':
    prefix=dx.cumsum(2)-dx;cross=torch.einsum('blti,bltj->blij',prefix,dx);value=.5*(cross-cross.transpose(-1,-2))
   else:value=torch.einsum('blti,bltj->blij',dx,dx)
   features=(value[...,self.pairs[0],self.pairs[1]]/16)*complete.unsqueeze(-1)
   return .25*torch.tanh(self.bridge(features))*complete.unsqueeze(-1)
class Network(Parent):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,'cross');self.path_mode=mode;self.path_correction=None
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261358);self.path_residual=PathResidual(mode)
  self.stem.register_forward_hook(self.add_path)
 def add_path(self,module,args,h):
  assert self.path_correction is not None
  return h+self.path_correction.to(h.dtype)
 def groups(self):return super().groups()+[dict(params=list(self.path_residual.parameters()),lr=3e-4,name='PERM_crosscurve_path_residual_backbone')]
 def initialize_priors(self,data):
  patch=self.path_residual;del self.path_residual
  try:info=super().initialize_priors(data)
  finally:self.path_residual=patch
  n=sum(p.numel() for p in self.parameters() if p.requires_grad);assert n==39162
  info.update(new_active_parameters=n,active_parameters=n,new_path_residual_parameters=1776,path_mode=self.path_mode,
   original154_only=True,no336356357_trained_weights=True,positive336_architecture_retained=True,
   all_fresh_encoder_readout_adapter_path_joint=True,zero_path_bridge=True,residual_bound=.25,
   path17points16edges=True,full17valid_required=True,curve_bounding='tanh original train-scaled x/3',feature_divisor=16,
   pair_features=36,area_formula='0.5 sum(prefix_before_i dx_j - dx_i prefix_before_j)',variation_formula='sum(dx_i dx_j)',
   primary_control='same1776bridge/init, signed order-sensitive area versus symmetric crossvariation; functionfamilies differ',
   no_signatory_NeuralRDE_full_reproduction=True,not_reverse_sign_pseudocontrol=True,one_PERM_added_phase=True)
  return info
 def forward(self,x,relative,valid,**kwargs):
  self.path_correction=self.path_residual(x.index_select(-1,self.curve_indices),valid)
  try:return super().forward(x,relative,valid,**kwargs)
  finally:self.path_correction=None

def make_model(stage,master,metadata):return Network(master,metadata,stage)
