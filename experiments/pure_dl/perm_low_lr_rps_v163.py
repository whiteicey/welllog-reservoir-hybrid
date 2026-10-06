"""Evidence-driven lower LR for positive PERM RPS under userpatience14."""
from perm_ranked_probability_v154 import ROOT,SOURCE,PARENT,Network as Previous,load_data,Evaluator,channel_objective
STUDY=ROOT/'results/deep_learning/perm_low_lr_rps_v163_20261001'
OUT=ROOT/'outputs/pure_dl_perm_low_lr_rps_v163_fold5'
STAGES={'perm_conditional':('PERM',256,False,0.,1e-4)}
class Network(Previous):
    def groups(self):return [dict(params=list(self.context_head.parameters()),lr=1e-4,name='PERM_lowLR_location_weight_head')]
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(head_lr=1e-4,original_gate_scale_body_frozen=True,trained_v154_head_not_loaded=True,
                    optimization_under_userpatience14=True,same_architecture_as_v154=True)
        return info

def make_model(stage,master,metadata):return Network(master,stage)
