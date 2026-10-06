"""Modify only one original QKV module, train-time DropKey for that module only."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from compact_frequency_v104 import ROOT,SOURCE,Network as Positive,load_data
from train_stage_fullcv_v97 import Evaluator as ParentEvaluator
STUDY=ROOT/'results/deep_learning/sw_bidir_memory_v231_20261001'
OUT=ROOT/'outputs/pure_dl_sw_bidir_memory_v231_fold5'
PARENT=ROOT/'outputs/pure_dl_compact_frequency_v104_fold5/local/model_best.pt'
PARENT_SHA='07e0b8df208fe93c8ac07dcf18924459c47afb12ec84b1b7306a0e6017f58670'
STAGES={'bigru':('SW',1024,True,1.,1e-4)}
class RecurrentResidual(nn.Module):
    def __init__(self,old):
        super().__init__();self.old=old
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010)
            self.norm=nn.LayerNorm(96);self.recurrent=nn.GRU(96,16,batch_first=True,bidirectional=True)
            self.project=nn.Linear(32,96);nn.init.zeros_(self.project.weight);nn.init.zeros_(self.project.bias)
    def forward(self,x,valid):
        h=self.old(x,valid)
        with torch.autocast(device_type=x.device.type,enabled=False):
            z=self.norm(h.detach().float()).masked_fill(~valid[...,None],0)
            memory,_=self.recurrent(z)
            bound=.1*h.detach().float().square().mean(-1,keepdim=True).add(1e-6).sqrt()
            delta=bound*torch.tanh(self.project(memory)/bound)
        return (h+delta.to(h.dtype)).masked_fill(~valid[...,None],0)

class Network(nn.Module):
    def __init__(self,metadata):
        super().__init__();self.base=Positive(metadata,'local')
        assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
        self.base.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=False)['state_dict'],strict=True)
        self.base.requires_grad_(False).eval()
        self.base.body.global_blocks[-1]=RecurrentResidual(self.base.body.global_blocks[-1])
    def train(self,mode=True):
        super().train(mode);self.base.eval()
        
        return self
    def groups(self):return [dict(params=[p for p in self.base.body.global_blocks[-1].parameters() if p.requires_grad],lr=1e-4,name='SW_last_global_small_bidirectional_memory')]
    def initialize_priors(self,data):
        return dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,same_initial_predictive_function=True,
                    zero_memory_output_projection=True,new_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    recurrent_hidden16_per_direction=True,original64pooled_tokens_retained=True,
                    point_RMS_memory_residual_bound=.1,invalid_memory_inputs_zeroed=True,invalid_outputs_masked=True,
                    all_original_weights_frozen=True,no_original_attention_DropKey=True,new_GRU_training_mode_for_cudnn_backward=True,
                    explicit_float32_GRU=True,own_original_NLL_BCE_jointutility1_SAM01=True,
                    no_other_repair_weights_loaded=True,executed_training_stages_including_source=6,added_SW_stages=1)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        return self.base(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
class Evaluator(ParentEvaluator):
    def __init__(self,data,master,c,length,mode):super().__init__(data,master,c,length)
def make_model(stage,master,metadata):return Network(metadata)
def channel_objective(*args):raise RuntimeError('Original own SW DecisionObjective only')
