"""Joint new raw state/density encoder with larger supervised component share."""
from sw_joint_state_encoder_v282 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,DecisionObjective,Network as Parent
STUDY=ROOT/'results/deep_learning/sw_joint_share_v285_20261002'
OUT=ROOT/'outputs/pure_dl_sw_joint_share_v285_fold5'
STAGES={'balanced':('SW',1024,False,0.,3e-4)}
class Network(Parent):
    def set_epoch(self,epoch):self.mix_alpha.fill_(.25*min(epoch/5,1.))
    def initialize_priors(self,data):
        info=super().initialize_priors(data);info.update(final_mix_alpha=.25,reference_final_mix_alpha=.1,reference='v282 balanced matched18',no282trained_weights_loaded=True,share_schedule_only_changed=True)
        return info
def make_model(stage,master,metadata):return Network(master,metadata,stage)
