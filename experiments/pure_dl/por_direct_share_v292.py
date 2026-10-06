"""Joint new POR raw encoder/readout with stronger supervised component share."""
from por_direct_encoder_v287 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent
STUDY=ROOT/'results/deep_learning/por_direct_share_v292_20261002'
OUT=ROOT/'outputs/pure_dl_por_direct_share_v292_fold5'
STAGES={'raw':('POR',256,False,0.,3e-4)}
class Network(Parent):
    def set_epoch(self,epoch):self.mix_alpha.fill_(.5*min(epoch/5,1.))
    def initialize_priors(self,data):
        info=super().initialize_priors(data);info.update(final_mix_alpha=.5,reference_final_mix_alpha=.25,reference='v290raw natural80-14',no287_or290trained_weights_loaded=True,only_schedule_amplitude_changed=True)
        return info
def make_model(stage,master,metadata):return Network(master,metadata,stage)
