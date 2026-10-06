"""Modify only one original QKV module, train-time DropKey for that module only."""
import hashlib,types
from torch.nn import functional as F
import torch
from torch import nn
from compact_frequency_v104 import ROOT,SOURCE,Network as Positive,load_data
from train_stage_fullcv_v97 import Evaluator as ParentEvaluator
STUDY=ROOT/'results/deep_learning/sw_attention_distance_math_v221_20261001'
OUT=ROOT/'outputs/pure_dl_sw_attention_distance_math_v221_fold5'
PARENT=ROOT/'outputs/pure_dl_compact_frequency_v104_fold5/local/model_best.pt'
PARENT_SHA='07e0b8df208fe93c8ac07dcf18924459c47afb12ec84b1b7306a0e6017f58670'
STAGES={'distance':('SW',1024,True,1.,1e-3)}

def distance_forward(self,x,valid):
    b,t,c=x.shape
    q,k,v=self.qkv(self.norm(x)).reshape(b,t,3,self.heads,c//self.heads).permute(2,0,3,1,4).unbind(0)
    positions=torch.arange(t,device=x.device,dtype=torch.float32)
    distance=(positions[:,None]-positions[None,:]).abs()/max(t-1,1)
    strength=2*torch.tanh(self.distance_bias.float()/2)
    mask=(-strength[None,:,None,None]*distance[None,None]).expand(b,-1,-1,-1).masked_fill(~valid[:,None,None,:],float('-inf')).to(q.dtype)
    scores=(q.float()@k.float().transpose(-2,-1))/(q.shape[-1]**.5)
    z=((scores+mask.float()).softmax(-1)@v.float()).to(q.dtype)
    x=(x+self.project(z.transpose(1,2).reshape(b,t,c))).masked_fill(~valid[...,None],0)
    return (x+self.ff(self.ffnorm(x))).masked_fill(~valid[...,None],0)

class Network(nn.Module):
    def __init__(self,metadata):
        super().__init__();self.base=Positive(metadata,'local')
        assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
        self.base.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=False)['state_dict'],strict=True)
        self.base.requires_grad_(False).eval()
        block=self.base.body.global_blocks[-1]
        block.distance_bias=nn.Parameter(torch.zeros(block.heads))
        block.forward=types.MethodType(distance_forward,block)
    def train(self,mode=True):
        super().train(mode);self.base.eval()
        
        return self
    def groups(self):return [dict(params=[self.base.body.global_blocks[-1].distance_bias],lr=1e-3,name='SW_last_attention_distance_bias')]
    def initialize_priors(self,data):
        return dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,all_old_weights_frozen=True,
                    added_parameters=self.base.body.global_blocks[-1].heads,zero_distance_bias=True,explicit_float32_attention_due_to_CUDA_LSE_backward_failure=True,
                    distance_bound=2.,distance_normalization='token separation divided by maximum separation',
                    initial_function_mathematical_identity_mask_float_rounding_must_be_measured=True,
                    all_blocks_eval_no_dropkey=True,own_original_NLL_BCE_jointutility1_SAM01=True,
                    only_last_attention_module_modified=True,no_other_repair_weights_loaded=True,
                    executed_training_stages_including_source=6,added_SW_stages=1)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        return self.base(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
class Evaluator(ParentEvaluator):
    def __init__(self,data,master,c,length,mode):super().__init__(data,master,c,length)
def make_model(stage,master,metadata):return Network(metadata)
def channel_objective(*args):raise RuntimeError('Original own SW DecisionObjective only')
