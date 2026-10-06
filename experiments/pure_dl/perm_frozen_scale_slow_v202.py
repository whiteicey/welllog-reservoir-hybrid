"""One conservative scale-only additive patch on frozen learned154 density."""
import hashlib
import torch
from torch import nn
from perm_ranked_probability_v154 import ROOT,SOURCE,Network as Positive,load_data,Evaluator,channel_objective
STUDY=ROOT/'results/deep_learning/perm_frozen_scale_slow_v202_20261001'
OUT=ROOT/'outputs/pure_dl_perm_frozen_scale_slow_v202_fold5'
PARENT=ROOT/'outputs/pure_dl_perm_ranked_probability_v154_fold5/perm_conditional/model_best.pt'
PARENT_SHA='2b355bb8c3736990b05e2726373a4343c574a93ab6ab1a3eda0ef796ac70cf11'
STAGES={'scale':('PERM',256,False,0.,3e-5)}

class Network(Positive):
    def __init__(self,master,mode):
        super().__init__(master,mode)
        assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
        saved=torch.load(PARENT,map_location='cpu',weights_only=False);self.load_state_dict(saved['state_dict'],strict=True)
        self.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010)
            self.scale_patch=nn.Sequential(nn.Conv1d(128,32,1),nn.GELU(),nn.Conv1d(32,16,1))
            nn.init.zeros_(self.scale_patch[-1].weight);nn.init.zeros_(self.scale_patch[-1].bias)
        self.scale_feature=None
        self.scale_handle=self.context_head.output.register_forward_pre_hook(self.capture_scale)
    def capture_scale(self,module,args):self.scale_feature=args[0].detach()
    def groups(self):return [dict(params=list(self.scale_patch.parameters()),lr=3e-5,name='PERM_only_small_scale_patch')]
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,SOURCE_checkpoint=str(SOURCE),
                    trained154_head_loaded_and_frozen=True,only_new_scale_patch_trainable=True,
                    gate_location_logweights_backbone_oldscale_parameters_frozen=True,
                    raw_scale_additive_bound=.1,conditional_scale_function_can_change=True,
                    extra_active_parameters=4656,zero_initialized_last_projection=True,
                    same_initial_predictive_function=True,original_NLL_BCE_jointRPS_retained=True,
                    executed_training_stages_including_source=6,total_stage_upper_bound=8,
                    no197_weights_loaded=True)
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.scale_feature=None
        with torch.no_grad():
            gate,raw=super().forward(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
        assert self.scale_feature is not None
        feature=self.scale_feature.masked_fill(~valid[:,None],0)
        delta=self.scale_patch(feature).transpose(1,2).float().masked_fill(~valid[...,None],0)
        self.scale_feature=None;own=raw[:,:,0]
        corrected=torch.stack((own[:,:,0],own[:,:,1],own[:,:,2]+.1*torch.tanh(delta/.1)),2)
        return gate,corrected.unsqueeze(2)

def make_model(stage,master,metadata):return Network(master,stage)

