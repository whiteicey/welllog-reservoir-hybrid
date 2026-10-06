"""Modify only one original QKV module, train-time DropKey for that module only."""
import hashlib
import torch
from torch import nn
from compact_frequency_v104 import ROOT,SOURCE,Network as Positive,load_data
from train_stage_fullcv_v97 import Evaluator as ParentEvaluator
STUDY=ROOT/'results/deep_learning/sw_first_attention_tune_v218_20261001'
OUT=ROOT/'outputs/pure_dl_sw_first_attention_tune_v218_fold5'
PARENT=ROOT/'outputs/pure_dl_compact_frequency_v104_fold5/local/model_best.pt'
PARENT_SHA='07e0b8df208fe93c8ac07dcf18924459c47afb12ec84b1b7306a0e6017f58670'
STAGES={'qkv':('SW',1024,True,1.,3e-5)}
class Network(nn.Module):
    def __init__(self,metadata):
        super().__init__();self.base=Positive(metadata,'local')
        assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
        self.base.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=False)['state_dict'],strict=True)
        self.base.requires_grad_(False).eval()
        self.base.body.global_blocks[0].qkv.requires_grad_(True)
    def train(self,mode=True):
        super().train(mode);self.base.eval()
        self.base.body.global_blocks[0].train(mode)
        return self
    def groups(self):return [dict(params=list(self.base.body.global_blocks[0].qkv.parameters()),lr=3e-5,name='SW_only_first_original_QKV')]
    def initialize_priors(self,data):
        return dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,same_initial_predictive_function=True,
                    only_first_QKV_original_parameters_trainable=True,existing_active_parameters=27936,
                    added_parameters=0,all_other_weights_frozen=True,old_readout_retained=True,
                    training_keydrop_first_block_only=.4,evaluation_keydrop=0.,
                    own_original_NLL_BCE_jointutility1_SAM01=True,only_one_small_module_modified=True,
                    no214_adapter_or_mass_weights_loaded=True,executed_training_stages_including_source=6,added_SW_stages=1)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        return self.base(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
class Evaluator(ParentEvaluator):
    def __init__(self,data,master,c,length,mode):super().__init__(data,master,c,length)
def make_model(stage,master,metadata):return Network(metadata)
def channel_objective(*args):raise RuntimeError('Original own SW DecisionObjective only')
