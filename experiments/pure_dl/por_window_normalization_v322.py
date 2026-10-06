"""Adaptive window versus training-reference normalization of fresh POR curve tokens."""
import torch
from torch import nn
from torch.nn import functional as F
from por_curve_scope_v302 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent
STUDY=ROOT/'results/deep_learning/por_window_normalization_v322_20261002'
OUT=ROOT/'outputs/pure_dl_por_window_normalization_v322_fold5'
STAGES={a:('POR',256,False,0.,3e-4) for a in ('adaptive','reference')}
class WindowNorm(nn.Module):
 def __init__(self,mode,reference):
  super().__init__();self.mode=mode
  self.shift=nn.Parameter(torch.zeros(9,9));self.scale=nn.Parameter(torch.zeros(9,9))
  self.register_buffer('reference',reference.detach().clone().cpu())
 def forward(self,x,valid):
  if self.mode=='adaptive':z=x.float();v=valid.unsqueeze(-1).float()
  else:z=self.reference[None].expand(x.shape[0],-1,-1).float();v=torch.ones_like(z[...,:1])
  den=v.sum(1).clamp_min(1)
  mu=(z*v).sum(1)/den
  shift=.25*torch.tanh(F.linear(mu,self.shift))
  rms=(((z-shift[:,None]).square()*v).sum(1)/den+1e-6).sqrt()
  logscale=.25*torch.tanh(F.linear(rms.log().clamp(-3,3),self.scale))
  return ((x.float()-shift[:,None])/logscale.exp()[:,None]).to(x.dtype)
class Network(Parent):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,'full');self.normalization_mode=mode
  reference=metadata['reference_inputs'].index_select(-1,self.curve_indices.to(metadata['reference_inputs'].device))
  self.window_norm=WindowNorm(mode,reference);self.current_valid=None
  native=self.curve_interaction.forward
  self.curve_interaction.forward=lambda curves:native(self.window_norm(curves,self.current_valid))
 def groups(self):return super().groups()+[dict(params=list(self.window_norm.parameters()),lr=3e-4,name='POR_learned_window_shift_scale')]
 def initialize_priors(self,data):
  info=super().initialize_priors(data);n=sum(p.numel() for p in self.parameters() if p.requires_grad);assert n==76754
  info.update(normalization_mode=self.normalization_mode,active_parameters=n,new_normalizer_parameters=162,shift_bound=.25,positive_scale_bounds=[.7788007830714049,1.2840254166877414],current_valid_window_stats_vs64train_reference=True,normalization_only_curve_token_side=True,original206_plus63_dense_branch_unchanged=True,original151prior_unchanged=True,no302trained_weights_loaded=True,DAIN_inspired_not_official_reproduction=True,no_target_denormalization=True,initial_curve_function_identity=True)
  return info
 def forward(self,x,relative,valid,**kwargs):
  self.current_valid=valid
  try:return super().forward(x,relative,valid,**kwargs)
  finally:self.current_valid=None
def make_model(stage,master,metadata):return Network(master,metadata,stage)
