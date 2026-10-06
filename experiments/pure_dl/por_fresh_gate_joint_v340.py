"""Fresh positive curve encoder jointly learns a bounded gate residual and density."""
import torch
from torch import nn
from por_curve_token_width_v335 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as WideParent
from por_curve_scope_v302 import Network as OriginalParent
STUDY=ROOT/'results/deep_learning/por_fresh_gate_joint_v340_20261002'
OUT=ROOT/'outputs/pure_dl_por_fresh_gate_joint_v340_fold5'
STAGES={'gate':('POR',256,False,0.,3e-4)}
class Network(WideParent):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,'wide')
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261340);self.gate_residual=nn.Linear(48,1)
  nn.init.zeros_(self.gate_residual.weight);nn.init.zeros_(self.gate_residual.bias)
  self.gate_feature=None;self.fuse.register_forward_hook(self.capture_gate_feature)
 def capture_gate_feature(self,module,args,result):self.gate_feature=result
 def groups(self):
  return super().groups()+[dict(params=list(self.gate_residual.parameters()),lr=3e-4,name='POR_fresh_joint_gate')]
 def initialize_priors(self,data):
  info=OriginalParent.initialize_priors(self,data);n=sum(p.numel() for p in self.parameters() if p.requires_grad);assert n==128385
  info.update(token_width=64,FFN=128,new_interaction_parameters=70768,new_active_parameters=n,active_parameters=n,gate_residual_parameters=49,gate_residual_bound=.5,gate_formula='g151+.5*tanh(u/.5)',zero_gate_init_preserves_original_function=True,final_gate_trainable_but_original151_gate_frozen=True,all_fresh_raw_encoder_attention_gate_density_joint=True,original151_only=True,no335_or_other_repair_trained_weights=True,only_scope_differs_in_matched_pair=False,not_capacity_matched_control=True)
  return info
 def forward(self,x,*args,**kwargs):
  self.gate_feature=None
  try:
   gate,mass=super().forward(x,*args,**kwargs);assert self.gate_feature is not None
   delta=.5*torch.tanh(self.gate_residual(self.gate_feature).float()/.5)
   return torch.cat((gate[:,:,:1]+delta,gate[:,:,1:]),-1),mass
  finally:self.gate_feature=None

def make_model(stage,master,metadata):return Network(master,metadata,stage)
