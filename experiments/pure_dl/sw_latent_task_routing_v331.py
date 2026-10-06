"""Joint SW encoder/readout with tiny task-specific latent expert routing."""
import torch
from torch import nn
from torch.nn import functional as F
from sw_latent_morphology_v326 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,DecisionObjective,Network as Parent
from sw_raw_well_condition_v297 import Network as RawParent
STUDY=ROOT/'results/deep_learning/sw_latent_task_routing_v331_20261002'
OUT=ROOT/'outputs/pure_dl_sw_latent_task_routing_v331_fold5'
STAGES={s:('SW',1024,False,0.,3e-4) for s in ('shared','task')}
class Network(Parent):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,'depth');self.task_routing_mode=mode
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20261331);self.task_routers=nn.ModuleList([nn.Linear(144,3) for _ in range(2)])
   for router in self.task_routers:nn.init.zeros_(router.weight);nn.init.zeros_(router.bias)
  self.routed_state_input=None
  self.fuse.register_forward_pre_hook(self.route_experts)
  # Parent captures density fused output first; override state feature afterwards.
  self.fuse.register_forward_hook(self.route_state_feature)
 def groups(self):return super().groups()+[dict(params=list(self.task_routers.parameters()),lr=3e-4,name='SW_joint_latent_task_routing')]
 def route_experts(self,module,args):
  concat=args[0];logits=[router(concat).float() for router in self.task_routers]
  if self.task_routing_mode=='shared':logits=[(logits[0]+logits[1])*.5]*2
  experts=concat.reshape(*concat.shape[:-1],3,48)
  routed=[(experts*(3*value.softmax(-1)).unsqueeze(-1).to(experts.dtype)).flatten(-2) for value in logits]
  self.routed_state_input=routed[1]
  return (routed[0],)
 def route_state_feature(self,module,args,result):
  # Reuse the SAME fusion layer; bypass module hooks to prevent recursion.
  self.state_feature=F.gelu(F.linear(self.routed_state_input,self.fuse[0].weight,self.fuse[0].bias))
  self.routed_state_input=None
 def initialize_priors(self,data):
  info=RawParent.initialize_priors(self,data);n=sum(p.numel() for p in self.parameters() if p.requires_grad);assert n==93159
  info.update(active_parameters=n,new_active_parameters=n,new_latent_fusion_parameters=21600,new_morphology_parameters=5424,new_task_router_parameters=870,task_routing_mode=self.task_routing_mode,
   routing_level='three latent experts before shared fuse, not prediction allocation',router_zero_initial_uniform_times3=True,original_fresh326_function_preserved_before_updates=True,
   density_and_state_use_same_fuse_parameters=True,shared_control_averages_both_router_logits=True,effective_control_router_capacity_lower_than_task=True,
   source_only104=True,no326_or_other_trained_repair_weights_loaded=True,own_norm128_repeat3_early_latent=True,morphology_depth_first_branch_only=True,
   all_fresh_encoder_readout_state_fusion_condition_morphology_routers_joint=True,final_mix_alpha=.25,one_SW_additional_phase=True,
   initial_predictor_preserved_by_alpha0_and_zero_gate=True,no_label_or_prior_prediction_router_input=True,no_new_experts=True)
  return info
 def forward(self,*args,**kwargs):
  try:return super().forward(*args,**kwargs)
  finally:self.routed_state_input=None

def make_model(stage,master,metadata):return Network(master,metadata,stage)
