"""Small amplitude-conditioned backbone residual with repeated-hidden control."""
import hashlib
import torch
from torch import nn
from compact_frequency_v104 import ROOT,SOURCE,Network as Positive
from compact_five_stage_v100 import load_data as original_data
from train_stage_fullcv_v97 import Evaluator as ParentEvaluator
STUDY=ROOT/'results/deep_learning/sw_amplitude_backbone_v229_20261001'
OUT=ROOT/'outputs/pure_dl_sw_amplitude_backbone_v229_fold5'
PARENT=ROOT/'outputs/pure_dl_compact_frequency_v104_fold5/local/model_best.pt'
PARENT_SHA='07e0b8df208fe93c8ac07dcf18924459c47afb12ec84b1b7306a0e6017f58670'
STAGES={m:('SW',1024,True,1.,3e-4) for m in ('amplitude','repeat')}
def load_data(stage):return original_data('por')
class Network(nn.Module):
    def __init__(self,metadata,mode):
        super().__init__();self.mode=mode;self.base=Positive(metadata,'local')
        assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
        self.base.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=False)['state_dict'],strict=True)
        self.base.requires_grad_(False).eval()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010);self.patch=nn.Sequential(nn.Linear(159,32),nn.GELU(),nn.Linear(32,96))
            nn.init.zeros_(self.patch[-1].weight);nn.init.zeros_(self.patch[-1].bias)
        self.side=None;self.handle=self.base.body.local[-1].register_forward_hook(self.repair)
    def repair(self,module,args,h):
        valid=args[1];side=self.side if self.mode=='amplitude' else h.detach()[...,:63]
        features=torch.cat((h.detach(),side),-1)
        bound=.1*h.detach().float().square().mean(-1,keepdim=True).add(1e-6).sqrt()
        delta=bound*torch.tanh(self.patch(features).float()/bound)
        return (h+delta.to(h.dtype)).masked_fill(~valid[...,None],0)
    def train(self,mode=True):super().train(mode);self.base.eval();return self
    def groups(self):return [dict(params=list(self.patch.parameters()),lr=3e-4,name='SW_96hidden_63side_zero_backbone_patch')]
    def initialize_priors(self,data):
        return dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,all104_weights_frozen=True,
                    same_initial_predictive_function=True,zero_output=True,new_parameters=8288,
                    point_RMS_residual_bound=.1,feature_side=self.mode,side_dimensions=63,
                    amplitude_source='existing compact100 training-fitted amplitude.npz',
                    same_capacity_initial_state_pair=True,effective_function_rank_not_matched=True,
                    old_condition_readout_retained=True,all_blocks_eval_no_keydrop=True,
                    only_one_local_backbone_module_repaired=True,no_other_repair_weights_loaded=True,
                    executed_training_stages_including_source=6,added_SW_stages=1)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        assert x.shape[-1]==269
        self.side=x[...,206:].detach()
        result=self.base(x[...,:206].contiguous(),relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
        self.side=None;return result
class Evaluator(ParentEvaluator):
    def __init__(self,data,master,c,length,mode):super().__init__(data,master,c,length)
def make_model(stage,master,metadata):return Network(metadata,stage)
def channel_objective(*args):raise RuntimeError('Own original SW DecisionObjective only')
