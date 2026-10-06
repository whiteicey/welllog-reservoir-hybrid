"""Fresh positive336+358 code with one learned local feature residual branch."""
import torch
from torch import nn
from perm_path_residual_v358 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent
STUDY=ROOT/'results/deep_learning/perm_depth_path_residual_v359_20261002'
OUT=ROOT/'outputs/pure_dl_perm_depth_path_residual_v359_fold5'
STAGES={a:('PERM',256,False,0.,3e-4) for a in ('depth','control')}
class TemporalResidual(nn.Module):
 def __init__(self,mode):
  super().__init__();self.mode=mode;self.norm=nn.LayerNorm(48);self.down=nn.Linear(48,8)
  self.middle=nn.Sequential(nn.Linear(153,32),nn.GELU(),nn.Linear(32,8));self.up=nn.Linear(8,48)
  nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
 def forward(self,h,valid):
  z=self.down(self.norm(h));length=h.shape[1];pos=torch.arange(length,device=h.device);values=[];masks=[]
  for k in range(-8,9):
   present=valid & torch.roll(valid,-k,1) & ((pos+k>=0)&(pos+k<length))[None]
   v=torch.roll(z,-k,1) if self.mode=='depth' else torch.roll(z,k,-1)
   values.append(v*present.unsqueeze(-1));masks.append(present)
  features=torch.cat(values,-1);availability=torch.stack(masks,-1).to(z.dtype)
  correction=.25*torch.tanh(self.up(self.middle(torch.cat((features,availability),-1))))
  return h+correction*valid.unsqueeze(-1)
class Network(Parent):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,'area');self.temporal_mode=mode
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261359);self.temporal_residual=TemporalResidual(mode)
  self.stem.register_forward_hook(self.add_temporal)
 def add_temporal(self,module,args,h):return self.temporal_residual(h,self.current_valid)
 def groups(self):return super().groups()+[dict(params=list(self.temporal_residual.parameters()),lr=3e-4,name='PERM_learned_depth_residual_backbone')]
 def initialize_priors(self,data):
  branch=self.temporal_residual;del self.temporal_residual
  try:info=super().initialize_priors(data)
  finally:self.temporal_residual=branch
  n=sum(p.numel() for p in self.parameters() if p.requires_grad);extra=sum(p.numel() for p in branch.parameters());assert n==45274 and extra==6112
  info.update(active_parameters=n,new_active_parameters=n,new_temporal_residual_parameters=extra,temporal_mode=self.temporal_mode,
   positive336358_code_retained=True,no336358_trained_weights=True,original154_only=True,all_fresh_encoder_readout_path_temporal_joint=True,
   temporal_taps=17,temporal_latent=8,temporal_middle=32,zero_temporal_exit=True,temporal_residual_bound=.25,
   temporal_placement='stem after336cross and358area, before originalselector',
   primary_control='17depth offsets vs samepoint cyclic8latent, same45274params/init/availability but function-information differs',
   positivePOR338_temporal_code_transfer=True,no338_trained_weights=True,not_firstPERMTCN=True,
   old273fullreplacement_negative=True,old293selector_pooling_negative=True,old356357_native_residual_negative=True,
   one_PERM_added_phase=True)
  return info

def make_model(stage,master,metadata):return Network(master,metadata,stage)
