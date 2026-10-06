"""Preserve all104 weights; rank8 update to first global attention QKV only."""
import hashlib
import torch
from torch import nn
from compact_frequency_v104 import ROOT,SOURCE,Network as Positive,load_data
from train_stage_fullcv_v97 import Evaluator as ParentEvaluator
STUDY=ROOT/'results/deep_learning/sw_attention_adapter_v214_20261001'
OUT=ROOT/'outputs/pure_dl_sw_attention_adapter_v214_fold5'
PARENT=ROOT/'outputs/pure_dl_compact_frequency_v104_fold5/local/model_best.pt'
PARENT_SHA='07e0b8df208fe93c8ac07dcf18924459c47afb12ec84b1b7306a0e6017f58670'
STAGES={'rank8':('SW',1024,True,1.,1e-4)}

class AdaptedLinear(nn.Module):
    def __init__(self,old):
        super().__init__();self.old=old
        self.down=nn.Linear(old.in_features,8,bias=False)
        self.up=nn.Linear(8,old.out_features,bias=False)
        nn.init.zeros_(self.up.weight)
    def forward(self,x):return self.old(x)+self.up(self.down(x))

class Network(nn.Module):
    def __init__(self,metadata):
        super().__init__();self.base=Positive(metadata,'local')
        assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
        self.base.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=False)['state_dict'],strict=True)
        self.base.requires_grad_(False).eval()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010)
            self.base.body.global_blocks[0].qkv=AdaptedLinear(self.base.body.global_blocks[0].qkv)
    def train(self,mode=True):super().train(mode);self.base.eval();return self
    def groups(self):
        layer=self.base.body.global_blocks[0].qkv
        return [dict(params=list(layer.down.parameters())+list(layer.up.parameters()),lr=1e-4,name='SW_first_attention_rank8_only')]
    def initialize_priors(self,data):
        return dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,old104_all_weights_frozen=True,
                    same_initial_predictive_function=True,only_first_attention_QKV_lowrank_delta_trainable=True,
                    rank=8,update_scaling=1.,zero_up_projection=True,extra_active_parameters=3072,
                    original_readout_retained=True,original_continuous_NLL_BCE_jointutility1_SAM01=True,
                    keydrop_disabled_base_eval=True,own_SW_labels_only=True,
                    executed_training_stages_including_source=6,added_SW_stages=1,no_mass_repair_weights_loaded=True)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        return self.base(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)

class Evaluator(ParentEvaluator):
    def __init__(self,data,master,c,length,mode):super().__init__(data,master,c,length)
def make_model(stage,master,metadata):return Network(metadata)
def channel_objective(*args):raise RuntimeError('Original own SW DecisionObjective only')
