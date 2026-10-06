"""Freeze all learned104; local feature adapter before unchanged Gaussian readout."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from compact_frequency_v104 import ROOT,SOURCE,Network as Positive,load_data
from train_stage_fullcv_v97 import Evaluator as ParentEvaluator
STUDY=ROOT/'results/deep_learning/sw_local_feature_adapter_v210_20261001'
OUT=ROOT/'outputs/pure_dl_sw_local_feature_adapter_v210_fold5'
PARENT=ROOT/'outputs/pure_dl_compact_frequency_v104_fold5/local/model_best.pt'
PARENT_SHA='07e0b8df208fe93c8ac07dcf18924459c47afb12ec84b1b7306a0e6017f58670'
STAGES={'local':('SW',1024,True,1.,1e-4)}

class LocalAdapter(nn.Module):
    def __init__(self):
        super().__init__();self.input=nn.Conv1d(128,32,1)
        self.depth=nn.Conv1d(32,32,9,padding=4,groups=32)
        self.output=nn.Conv1d(32,128,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,features,valid):
        x=features.transpose(1,2);mask=valid[:,None]
        count=F.avg_pool1d(mask.float(),9,stride=1,padding=4,count_include_pad=True)*9
        rms=(F.avg_pool1d(x.detach().float().square().masked_fill(~mask,0),9,stride=1,padding=4,count_include_pad=True)*9/count.clamp_min(1)).sqrt().clamp_min(.01)
        h=F.gelu(self.input((x/rms.to(x.dtype)).masked_fill(~mask,0))).masked_fill(~mask,0)
        h=F.gelu(self.depth(h)).masked_fill(~mask,0)
        return (.1*rms*torch.tanh(self.output(h).float()/.1)).masked_fill(~mask,0).transpose(1,2).to(features.dtype)

class Network(nn.Module):
    def __init__(self,metadata):
        super().__init__();self.base=Positive(metadata,'local')
        assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
        self.base.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=False)['state_dict'],strict=True)
        self.base.requires_grad_(False).eval()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010);self.adapter=LocalAdapter()
        self.current_valid=None;self.handle=self.base.output.register_forward_pre_hook(self.modify_readout_input)
    def modify_readout_input(self,module,args):
        assert self.current_valid is not None
        return (args[0]+self.adapter(args[0],self.current_valid),)
    def train(self,mode=True):super().train(mode);self.base.eval();return self
    def groups(self):return [dict(params=list(self.adapter.parameters()),lr=1e-4,name='SW_local_feature_adapter_only')]
    def initialize_priors(self,data):
        return dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,old104_all_parameters_frozen=True,
                    same_initial_predictive_function=True,retained_original_Gaussian_readout=True,
                    only_adapter_trainable=True,extra_active_parameters=sum(p.numel() for p in self.adapter.parameters()),
                    local_kernel=9,width=32,input_bound_masked_local_RMS=.1,
                    own_original_NLL_BCE_jointutility1_SAM01=True,gate_location_scale_weights_can_change_via_features=True,
                    base_eval_no_keydrop=True,executed_training_stages_including_source=6,
                    added_SW_stages=1,no_other_repair_weights_loaded=True)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.current_valid=valid
        try:return self.base(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
        finally:self.current_valid=None

class Evaluator(ParentEvaluator):
    def __init__(self,data,master,c,length,mode):super().__init__(data,master,c,length)
def make_model(stage,master,metadata):return Network(metadata)
def channel_objective(*args):raise RuntimeError('SW original own DecisionObjective only')
