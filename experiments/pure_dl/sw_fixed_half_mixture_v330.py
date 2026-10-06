"""Fresh positive SW encoder/readout jointly adapts to fixed half mixture."""
from sw_latent_morphology_v326 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,DecisionObjective,Network as Parent
STUDY=ROOT/'results/deep_learning/sw_fixed_half_mixture_v330_20261002'
OUT=ROOT/'outputs/pure_dl_sw_fixed_half_mixture_v330_fold5'
STAGES={'half':('SW',1024,False,0.,3e-4)}
class Network(Parent):
 def __init__(self,master,metadata,mode):super().__init__(master,metadata,'depth')
 def set_epoch(self,epoch):self.mix_alpha.fill_(.5*min(epoch/5,1.))
 def initialize_priors(self,data):
  info=super().initialize_priors(data)
  info.update(final_mix_alpha=.5,reference_final_mix_alpha=.25,alpha_ramp_epochs=5,
   active_parameters=92289,new_active_parameters=92289,new_backbone_width=48,
   source_only104=True,no326_or329_trained_weights_loaded=True,
   unchanged_density_CE_RPS_and_direct_aux_coefficients=True,all_fresh_encoder_readout_morphology_joint=True,
   fixed_half_used_for_training_and_inference=True,no_posthoc_mixture_sweep=True,
   not_new_backbone_architecture_claim=True,
   primary_control='Historical326alpha.25 natural27; compare predeclared bestthrough27 thenfull80 separately')
  return info
def make_model(stage,master,metadata):return Network(master,metadata,stage)
