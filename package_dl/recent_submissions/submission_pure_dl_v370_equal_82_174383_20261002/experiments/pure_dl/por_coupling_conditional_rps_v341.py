"""One fresh POR joint phase combining proven coupling code and conditionalRPS."""
from por_depth_curve_coupling_v338 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,Network as Parent
from por_conditional_rps_v339 import channel_objective
STUDY=ROOT/'results/deep_learning/por_coupling_conditional_rps_v341_20261002'
OUT=ROOT/'outputs/pure_dl_por_coupling_conditional_rps_v341_fold5'
STAGES={'conditional':('POR',256,False,0.,3e-4)}
class Network(Parent):
 def __init__(self,master,metadata,mode):super().__init__(master,metadata,'depth')
 def initialize_priors(self,data):
  info=super().initialize_priors(data)
  info.update(joint_ranked_probability_loss=False,conditional_only_RPS=True,conditional_RPS_all_rows_mean_normalization=True,default_RPS_gradient_removed=True,only_original151_checkpoint=True,no338_or339_trained_repair_weights=True,positive_code_and_init_only=True,one_POR_added_phase=True,primary_control='338depthjointRPS same134752params/init historical42',secondary_control='339conditional no6416temporal; capacitydiffers',not_same_effective_gradient_weight_as_joint=True)
  return info

def make_model(stage,master,metadata):return Network(master,metadata,stage)
