"""Retain positive151 and tune only its already learned context head."""
import hashlib
import torch
from por_ranked_probability_v151 import ROOT,SOURCE,Network as Positive,load_data,Evaluator,channel_objective as own_objective
STUDY=ROOT/'results/deep_learning/por_native_coadapt_v226_20261001'
OUT=ROOT/'outputs/pure_dl_por_native_coadapt_v226_fold5'
PARENT=ROOT/'outputs/pure_dl_por_ranked_probability_v151_fold5/joint_rps/model_best.pt'
PARENT_SHA='39f2255b8fe61fa74607af027ece1dd352e5e2d0729b9bce95d1424c1b227603'
STAGES={'coadapt':('POR',256,False,0.,3e-5)}
class Network(Positive):
    def __init__(self,master):
        super().__init__(master,'joint_rps')
        assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
        self.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=False)['state_dict'],strict=True)
        self.requires_grad_(False);self.context_head.requires_grad_(True)
    def groups(self):return [dict(params=list(self.context_head.parameters()),lr=3e-5,name='POR_existing_positive_context_head_only')]
    def initialize_priors(self,data):
        return dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,same_initial_predictive_function=True,
                    only_existing151_context_head_trainable=True,added_parameters=0,
                    active_parameters=sum(p.numel() for p in self.context_head.parameters()),
                    SOURCE_113_all_weights_frozen=True,old_gate_scale_routes_retained_and_frozen=True,
                    native_mixed_likelihood_and_jointRPS1_retained=True,new_optimizer=True,
                    stored33tap_effective_pointwise_sum_retained=True,no_probability_floor_added=True,
                    no224_or_other_repair_weights_loaded=True,executed_training_stages_including_source=6,added_POR_stages=1)
def make_model(stage,master,metadata):return Network(master)
def channel_objective(output,truth,default,mask,quality,c,mode,model):
    return own_objective(output,truth,default,mask,quality,c,'joint_rps',model)
