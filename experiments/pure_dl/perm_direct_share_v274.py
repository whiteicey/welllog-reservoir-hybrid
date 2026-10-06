"""Joint raw encoder/direct distribution training with stronger fixed component share."""
from perm_direct_multiscale_v271 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent
STUDY=ROOT/'results/deep_learning/perm_direct_share_v274_20261001'
OUT=ROOT/'outputs/pure_dl_perm_direct_share_v274_fold5'
STAGES={'alpha25':('PERM',256,False,0.,3e-4)}
class Network(Parent):
    def __init__(self,master,metadata,mode):super().__init__(master,metadata,'repeat');self.mode=mode
    def set_epoch(self,epoch):self.mix_alpha.fill_(.25*min(epoch/5,1.))
    def initialize_priors(self,data):
        info=super().initialize_priors(data);info.update(final_alpha=.25,reference_final_alpha=.1,reference='v271 repeat',new_encoder_and_readout_jointly_trained=True,only_schedule_amplitude_changed=True,no271_weights_loaded=True)
        return info
def make_model(stage,master,metadata):return Network(master,metadata,stage)
