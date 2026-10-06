"""Own PERM SOURCE body coadaptation with proven SAM/conditionalCDF/utility loss."""
import torch
from perm_private_backbone_rps_v166 import ROOT,SOURCE,PARENT,Network as Parent,load_data,Evaluator
from perm_conditional_cdf_v174 import channel_objective
STUDY=ROOT/'results/deep_learning/perm_source_body_sam_v184_20261001'
OUT=ROOT/'outputs/pure_dl_perm_source_body_sam_v184_fold5'
STAGES={'perm_conditional':('PERM',256,True,0.,3e-4)}
class Network(Parent):
    def __init__(self,master,mode):
        super().__init__(master,mode)
        self.register_buffer('physical_centers',torch.as_tensor(master['centers'][1],dtype=torch.float32))
        self.register_buffer('target_median',torch.tensor(float(master['regression_median'][1])))
        self.register_buffer('target_scale',torch.tensor(float(master['regression_scale'][1])))
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(joint_ranked_probability_loss=False,conditional_continuous_RPS=True,RPS_default_rows_zero=True,
                    RPS_global_row_normalization_unchanged=True,train_only_expected_logratio_utility_weight=.1,
                    original_decoder_unchanged=True,no_trained_v174_weights_loaded=True,
                    old_predictor_fully_frozen=False,body_eval_dropout_disabled=True,
                    inherited_v166_update_scope_not_weights=True,main_loss_own_PERM_only=True)
        return info
def make_model(stage,master,metadata):return Network(master,stage)
