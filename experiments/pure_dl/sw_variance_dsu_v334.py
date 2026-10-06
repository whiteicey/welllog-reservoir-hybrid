"""Joint SW encoder/readout with mean-preserving bounded variance-DSU training."""
import torch
from sw_latent_morphology_v326 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,DecisionObjective,Network as Parent
STUDY=ROOT/'results/deep_learning/sw_variance_dsu_v334_20261002'
OUT=ROOT/'outputs/pure_dl_sw_variance_dsu_v334_fold5'
STAGES={a:('SW',1024,False,0.,3e-4) for a in ('identity','noise')}
class Network(Parent):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,'depth');self.dsu_mode=mode
  # AFTER parent train-only bankFiLM and own104 latent fusion hooks.
  self.stem.register_forward_hook(self.variance_dsu)
 def variance_dsu(self,module,args,h):
  if not self.training:return None
  with torch.autocast(device_type=h.device.type,enabled=False):
   valid=self.current_valid.unsqueeze(-1).float();n=valid.sum(1,keepdim=True).clamp_min(1)
   mean=((h.float()*valid).sum(1,keepdim=True)/n).detach()
   std=(((h.float()-mean).square()*valid).sum(1,keepdim=True)/n+1e-6).sqrt().detach()
   uncertainty=std.var(0,unbiased=False,keepdim=True).clamp_min(0).sqrt().detach()
   noise=torch.randn_like(std);apply=(torch.rand(h.shape[0],1,1,device=h.device)<.5).float()
   scale_delta=.1*torch.tanh(noise*uncertainty/std.clamp_min(1e-3))*apply
   # Identity draws identical randomness and computes statistics, preserves h exactly.
   if self.dsu_mode=='identity':return h
   return ((h.float()+(h.float()-mean)*scale_delta)*valid).to(h.dtype)
 def initialize_priors(self,data):
  info=super().initialize_priors(data)
  info.update(dsu_mode=self.dsu_mode,training_only_variance_DSU=True,p_window=.5,max_relative_std_perturbation=.1,
   statistics='validrow masked perwindow/channelstd, crossbatchstd variability; detached input-only moments',
   mean_preserved=True,latent_mean_or_prior_not_perturbed=True,new_DSU_parameters=0,new_active_parameters=92289,
   hook_position='stem after wellFiLM and ownlatentearly before three branches',identity_draws_same_randomness=True,
   all_fresh_encoder_morph_fusion_condition_state_readout_joint=True,no326_trained_weights_or_labels_in_perturbation=True,
   exact_evaluation_function_unchanged=True,not_full_DSU_or_first_MixStyle=True,
   original104_only=True,one_SW_added_phase=True,training_batch_uncertainty_not_verified_domain_uncertainty=True)
  return info

def make_model(stage,master,metadata):return Network(master,metadata,stage)
