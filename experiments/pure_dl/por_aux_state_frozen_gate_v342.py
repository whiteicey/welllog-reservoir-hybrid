"""Fresh encoder state auxiliary BCE; main and inference gate remain original151."""
import torch
from torch.nn import functional as F
from por_fresh_gate_joint_v340 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,Network as Parent
from por_curve_token_width_v335 import Network as WideParent,channel_objective as original_objective
STUDY=ROOT/'results/deep_learning/por_aux_state_frozen_gate_v342_20261002'
OUT=ROOT/'outputs/pure_dl_por_aux_state_frozen_gate_v342_fold5'
STAGES={'aux':('POR',256,False,0.,3e-4)}
class Network(Parent):
 def __init__(self,master,metadata,mode):super().__init__(master,metadata,mode);self.aux_gate=None
 def initialize_priors(self,data):
  info=super().initialize_priors(data)
  info.update(final_gate_trainable_but_original151_gate_frozen=False,final_gate_always_original151=True,main_RPS_gate_always_original151=True,auxiliary_state_true_labels_only=True,ordinary_BCE_constant_replaced_not_doubled=True,all_fresh_encoder_density_auxstate_joint=True,no340_trained_weights=True,primary_control='340same128385structure but changedmainRPS andinferencegate role; notpurecausal',one_POR_added_phase=True)
  return info
 def forward(self,x,*args,**kwargs):
  self.gate_feature=None;self.aux_gate=None
  try:
   gate,mass=WideParent.forward(self,x,*args,**kwargs);assert self.gate_feature is not None
   if self.training:self.aux_gate=gate[:,:,0].float()+(.5*torch.tanh(self.gate_residual(self.gate_feature).float()/.5)).squeeze(-1)
   return gate,mass
  finally:self.gate_feature=None

def make_model(stage,master,metadata):return Network(master,metadata,stage)

def channel_objective(output,labels,default,mask,quality,target_index,mode,model,well_ids=None):
 original=original_objective(output,labels,default,mask,quality,target_index,mode,model,well_ids=well_ids)
 assert model.aux_gate is not None;d=default[mask][:,0].float();q=quality[mask].detach().float()
 base=F.binary_cross_entropy_with_logits(output[0][mask][:,0].float(),d,reduction='none')
 aux=F.binary_cross_entropy_with_logits(model.aux_gate[mask].float(),d,reduction='none')
 change=(q*(aux-base)).mean()/3;model.aux_gate=None
 return original[0]+change,original[1]+change,original[2]
