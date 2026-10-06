"""Original154 warm native function, real enc2 rank4 and readout joint update."""
import hashlib
import torch
from torch import nn
from perm_private_backbone_rps_v166 import Network as NativeForward
from perm_ranked_probability_v154 import ROOT,SOURCE,load_data,Evaluator,channel_objective as original_loss
STUDY=ROOT/'results/deep_learning/perm_native_encoder_lora_v348_20261002'
OUT=ROOT/'outputs/pure_dl_perm_native_encoder_lora_v348_fold5'
PARENT=ROOT/'outputs/pure_dl_perm_ranked_probability_v154_fold5/perm_conditional/model_best.pt'
PARENT_SHA='2b355bb8c3736990b05e2726373a4343c574a93ab6ab1a3eda0ef796ac70cf11'
STAGES={'encoder_lora':('PERM',256,False,0.,3e-5)}
class LowRankExpand(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base
        assert base.in_channels==128 and base.out_channels==256 and base.kernel_size==(1,)
        self.lora_A=nn.Conv1d(128,4,1,bias=False);self.lora_B=nn.Conv1d(4,256,1,bias=False)
        nn.init.zeros_(self.lora_B.weight)
    def forward(self,x):return self.base(x)+self.lora_B(self.lora_A(x))
class Network(NativeForward):
    def __init__(self,master,metadata,mode):
        super().__init__(master,'perm_conditional')
        assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
        self.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=False)['state_dict'],strict=True)
        self.requires_grad_(False);self.context_head.requires_grad_(True)
        for member in self.backbone.members:member.enc2.expand=LowRankExpand(member.enc2.expand)
        self.mode=mode;self.last_group_risk=0.;self.last_group_count=0;self.last_max_group_weight=0.
        assert sum(p.numel() for p in self.parameters() if p.requires_grad)==85792
    def groups(self):
        lora=[p for m in self.backbone.members for n,p in m.named_parameters() if 'lora_' in n]
        return [dict(params=list(self.context_head.parameters()),lr=3e-5,name='PERM_warm154_readout'),
                dict(params=lora,lr=3e-4,name='PERM_real_enc2_rank4')]
    def initialize_priors(self,data):
        return dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,initial_function_expected154=True,
            trained154_readout_loaded=True,real_encoder='two experts enc2.expand128to256',rank=4,lora_new_parameters=3072,
            zero_B=True,scaling=1.,native_forward_grad_enabled=True,capture_detach_removed=True,
            original_parameters_except_context_frozen=True,original_packed_gate_density_weights_frozen=True,
            native_gate_scale_functions_can_change_with_features=True,backbone_eval_dropout_disabled=True,
            original_Gaussian16_NLL_BCE_joint_RPS1_decoder_unchanged=True,one_added_PERM_stage=True,
            no_trained_repair_weights=True,historical_headonly270_not_concurrent_or_capacity_matched=True)
def make_model(stage,master,metadata):return Network(master,metadata,stage)
def channel_objective(output,labels,default,mask,quality,c,mode,model,well_ids):
    return original_loss(output,labels,default,mask,quality,c,'perm_conditional',model)
