"""Fresh joint curve/depth encoder: one local low-rank temporal residual."""
import torch
from torch import nn
from por_curve_token_width_v335 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as WideNetwork,WideCurveInteraction
from por_curve_scope_v302 import Network as OriginalParent
STUDY=ROOT/'results/deep_learning/por_depth_curve_coupling_v338_20261002'
OUT=ROOT/'outputs/pure_dl_por_depth_curve_coupling_v338_fold5'
STAGES={a:('POR',256,False,0.,3e-4) for a in ('depth','control')}
class TemporalResidual(nn.Module):
 def __init__(self,mode):
  super().__init__();self.mode=mode;self.norm=nn.LayerNorm(64);self.down=nn.Linear(64,8)
  self.middle=nn.Sequential(nn.Linear(153,32),nn.GELU(),nn.Linear(32,8));self.up=nn.Linear(8,64)
  nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
 def forward(self,h,valid):
  # B,L,9,64; offsets never wrap and all windows are from one well.
  z=self.down(self.norm(h));length=h.shape[1];pos=torch.arange(length,device=h.device);values=[];masks=[]
  for k in range(-8,9):
   present=valid & torch.roll(valid,-k,1) & ((pos+k>=0)&(pos+k<length))[None]
   v=torch.roll(z,-k,1) if self.mode=='depth' else torch.roll(z,k,-1)
   values.append(v*present[:,:,None,None]);masks.append(present)
  features=torch.cat(values,-1);availability=torch.stack(masks,-1)[:,:,None,:].expand(-1,-1,9,-1).to(z.dtype)
  correction=.25*torch.tanh(self.up(self.middle(torch.cat((features,availability),-1))))
  return h+correction*valid[:,:,None,None]
class CoupledInteraction(WideCurveInteraction):
 def __init__(self,mode):
  super().__init__()
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261338);self.temporal=TemporalResidual(mode)
  self.current_valid=None
 def forward(self,curves):
  b,l=curves.shape[:2];h=self.numeric(curves.reshape(-1,9,1))+self.identity[None]
  h=self.blocks[0](h,None).reshape(b,l,9,64);assert self.current_valid is not None
  h=self.temporal(h,self.current_valid).reshape(-1,9,64);h=self.blocks[1](h,None)
  return self.bridge(h.mean(1)).reshape(b,l,48)
class Network(WideNetwork):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,'wide');self.coupling_mode=mode
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261302);self.curve_interaction=CoupledInteraction(mode)
 def initialize_priors(self,data):
  info=OriginalParent.initialize_priors(self,data);n=sum(p.numel() for p in self.parameters() if p.requires_grad);extra=sum(p.numel() for p in self.curve_interaction.temporal.parameters());assert n==134752 and extra==6416
  info.update(token_width=64,FFN=128,new_interaction_parameters=77184,new_active_parameters=n,active_parameters=n,temporal_parameters=extra,coupling_mode=self.coupling_mode,taps=17,latent=8,middle=32,raw_encoder_readout_width_unchanged=48,original151_only=True,no335_or_other_trained_repair_weights=True,all_fresh_encoder_attention_readout_joint=True,only_scope_differs_in_matched_pair=False,physical_parameter_matched_not_effective_function_capacity=True,one_POR_added_phase=True,zero_bridge_and_zero_temporal_up_delay_gradients=True)
  return info
 def forward(self,x,relative,valid,**kwargs):
  self.curve_interaction.current_valid=valid
  try:return super().forward(x,relative,valid,**kwargs)
  finally:self.curve_interaction.current_valid=None

def make_model(stage,master,metadata):return Network(master,metadata,stage)
