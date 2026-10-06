"""Same fresh learned morphology shared by all three SW encoding branches."""
from sw_latent_morphology_v326 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,DecisionObjective,Network as Parent
STUDY=ROOT/'results/deep_learning/sw_shared_morphology_v327_20261002'
OUT=ROOT/'outputs/pure_dl_sw_shared_morphology_v327_fold5'
STAGES={'all':('SW',1024,False,0.,3e-4)}
class Network(Parent):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,'depth');self.morphology_coverage=mode
  for branch in self.branches[1:]:branch.register_forward_hook(self.morph_branch)
 def initialize_priors(self,data):
  info=super().initialize_priors(data)
  info.update(morphology_coverage='shared_all_three_parallel_branches',only_first_branch_morphology=False,
   new_morphology_parameters=5424,active_parameters=92289,shared_morphology_registered_once=True,
   no326_trained_weights_loaded=True,no_additional_SW_training_phase=True,
   unchanged_depth_receptive_field=True,shared_gradient_sum_no_LR_rescaling=True,
   auxiliary_count_four_same_but_module_gradient_paths_one_to_three=True,
   primary_control='Historical326depth firstbranch natural27; predefined bestthrough27, finalafter27 not equalwork gain')
  return info
def make_model(stage,master,metadata):return Network(master,metadata,stage)
