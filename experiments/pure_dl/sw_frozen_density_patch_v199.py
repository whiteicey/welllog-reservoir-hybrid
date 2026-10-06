"""Freeze verified SW104; one small conservative location/logweight repair."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from compact_frequency_v104 import ROOT,SOURCE,Network as Positive,load_data
from train_stage_fullcv_v97 import Evaluator as ParentEvaluator
STUDY=ROOT/'results/deep_learning/sw_frozen_density_patch_v199_20261001'
OUT=ROOT/'outputs/pure_dl_sw_frozen_density_patch_v199_fold5'
PARENT=ROOT/'outputs/pure_dl_compact_frequency_v104_fold5/local/model_best.pt'
PARENT_SHA='07e0b8df208fe93c8ac07dcf18924459c47afb12ec84b1b7306a0e6017f58670'
STAGES={'density':('SW',1024,True,1.,1e-4)}

class Network(nn.Module):
    def __init__(self,metadata):
        super().__init__();self.base=Positive(metadata,'local')
        assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
        saved=torch.load(PARENT,map_location='cpu',weights_only=False);self.base.load_state_dict(saved['state_dict'],strict=True)
        self.base.requires_grad_(False).eval()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010)
            self.patch=nn.Sequential(nn.Conv1d(128,32,1),nn.GELU(),nn.Conv1d(32,16,1))
            nn.init.zeros_(self.patch[-1].weight);nn.init.zeros_(self.patch[-1].bias)
        self.feature=None;self.handle=self.base.output.register_forward_pre_hook(self.capture)
    def capture(self,module,args):self.feature=args[0].detach().transpose(1,2)
    def train(self,mode=True):
        super().train(mode);self.base.eval();return self
    def groups(self):return [dict(params=list(self.patch.parameters()),lr=1e-4,name='SW_only_small_density_patch')]
    def initialize_priors(self,data):
        return dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,trained104_loaded_and_frozen=True,
                    old_gate_scale_body_condition_frequency_parameters_all_frozen=True,
                    only_new_density_patch_trainable=True,extra_active_parameters=4656,
                    zero_last_projection=True,same_initial_predictive_function=True,
                    location_bound_reference_sd=.25,logweight_bound=.25,
                    NLL_BCE_joint_utility1_retained=True,base_forced_eval_no_training_keydrop=True,
                    SAM_rho=.1,executed_training_stages_including_source=6,total_stage_upper_bound=8,
                    added_head_stages=dict(POR=0,PERM=0,SW=1),no_other_repair_weights_loaded=True)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.feature=None
        with torch.no_grad():gate,raw=self.base(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
        assert self.feature is not None
        u=self.patch(self.feature.masked_fill(~valid[:,None],0)).transpose(1,2).float().masked_fill(~valid[...,None],0)
        self.feature=None;own=raw[:,:,0];bound=.25*(F.softplus(own[:,:,2])+.025)
        location=bound*torch.tanh(u[...,:8]/bound);weight=.25*torch.tanh(u[...,8:]/.25)
        return gate,torch.stack((own[:,:,0]+weight,own[:,:,1]+location,own[:,:,2]),2).unsqueeze(2)

class Evaluator(ParentEvaluator):
    def __init__(self,data,master,c,length,mode):super().__init__(data,master,c,length)

def make_model(stage,master,metadata):return Network(metadata)

def channel_objective(*args):raise RuntimeError('SW uses original DecisionObjective only')
