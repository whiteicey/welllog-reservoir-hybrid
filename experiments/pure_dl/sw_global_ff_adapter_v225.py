"""Modify only one original QKV module, train-time DropKey for that module only."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from compact_frequency_v104 import ROOT,SOURCE,Network as Positive,load_data
from train_stage_fullcv_v97 import Evaluator as ParentEvaluator
STUDY=ROOT/'results/deep_learning/sw_global_ff_adapter_v225_20261001'
OUT=ROOT/'outputs/pure_dl_sw_global_ff_adapter_v225_fold5'
PARENT=ROOT/'outputs/pure_dl_compact_frequency_v104_fold5/local/model_best.pt'
PARENT_SHA='07e0b8df208fe93c8ac07dcf18924459c47afb12ec84b1b7306a0e6017f58670'
STAGES={'adapter':('SW',1024,True,1.,3e-4)}
class ResidualFF(nn.Module):
    def __init__(self,old):
        super().__init__();self.old=old
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010);self.patch=nn.Sequential(nn.Linear(96,32),nn.GELU(),nn.Linear(32,96))
            nn.init.zeros_(self.patch[-1].weight);nn.init.zeros_(self.patch[-1].bias)
    def forward(self,x):
        return self.old(x)+(.1*torch.tanh(self.patch(x).float()/.1)).to(x.dtype)

class Network(nn.Module):
    def __init__(self,metadata):
        super().__init__();self.base=Positive(metadata,'local')
        assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
        self.base.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=False)['state_dict'],strict=True)
        self.base.requires_grad_(False).eval()
        self.base.body.global_blocks[-1].ff=ResidualFF(self.base.body.global_blocks[-1].ff)
    def train(self,mode=True):
        super().train(mode);self.base.eval()
        
        return self
    def groups(self):return [dict(params=list(self.base.body.global_blocks[-1].ff.patch.parameters()),lr=3e-4,name='SW_last_global_FF_zero_residual')]
    def initialize_priors(self,data):
        return dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,same_initial_predictive_function=True,
                    only_last_FF_additive_patch_trainable=True,added_parameters=6272,patch_width=32,
                    zero_output=True,residual_bound=.1,all_original_weights_frozen=True,
                    old_readout_retained=True,all_blocks_eval_no_keydrop=True,
                    own_original_NLL_BCE_jointutility1_SAM01=True,only_one_small_module_modified=True,
                    no_other_repair_weights_loaded=True,executed_training_stages_including_source=6,added_SW_stages=1)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        return self.base(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
class Evaluator(ParentEvaluator):
    def __init__(self,data,master,c,length,mode):super().__init__(data,master,c,length)
def make_model(stage,master,metadata):return Network(metadata)
def channel_objective(*args):raise RuntimeError('Original own SW DecisionObjective only')
