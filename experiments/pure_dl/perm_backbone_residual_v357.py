"""Original154 warm native function, real enc2 rank4 and readout joint update."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from perm_private_backbone_rps_v166 import Network as NativeForward
from perm_ranked_probability_v154 import ROOT,SOURCE,load_data,Evaluator,channel_objective as original_loss
STUDY=ROOT/'results/deep_learning/perm_backbone_residual_v357_20261002'
OUT=ROOT/'outputs/pure_dl_perm_backbone_residual_v357_fold5'
PARENT=ROOT/'outputs/pure_dl_perm_ranked_probability_v154_fold5/perm_conditional/model_best.pt'
PARENT_SHA='2b355bb8c3736990b05e2726373a4343c574a93ab6ab1a3eda0ef796ac70cf11'
STAGES={a:('PERM',256,False,0.,3e-5) for a in ('local','point')}
class BackboneResidual(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode;self.down=nn.Conv1d(128,32,1);self.norm=nn.LayerNorm(32);self.depth=nn.Conv1d(32,32,17,padding=8,groups=32);self.expand=nn.Conv1d(32,64,1);self.up=nn.Conv1d(32,128,1)
        nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
    def forward(self,h,valid):
        mask=F.interpolate(valid[:,None].float(),size=h.shape[-1],mode='nearest').bool()
        z=F.gelu(self.down(h)).masked_fill(~mask,0);z=self.norm(z.transpose(1,2)).transpose(1,2).masked_fill(~mask,0)
        if self.mode=='local':z=self.depth(z)
        else:z=F.conv1d(z,self.depth.weight.sum(-1,keepdim=True),self.depth.bias,groups=32)
        a,b=self.expand(F.gelu(z).masked_fill(~mask,0)).chunk(2,1)
        return .25*torch.tanh(self.up(F.gelu(a)*b.sigmoid()).float()).masked_fill(~mask,0).to(h.dtype)
class Network(NativeForward):
    def __init__(self,master,metadata,mode):
        super().__init__(master,'perm_conditional')
        assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
        self.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=False)['state_dict'],strict=True)
        self.requires_grad_(False);self.context_head.requires_grad_(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261357);self.residual_branches=nn.ModuleList([BackboneResidual(mode) for _ in range(2)])
        self.patch_valid=None
        for i,member in enumerate(self.backbone.members):member.enc2.register_forward_hook(self.make_feature_patch(i))
        self.mode=mode;self.last_group_risk=0.;self.last_group_count=0;self.last_max_group_weight=0.
        assert sum(p.numel() for p in self.parameters() if p.requires_grad)==104928
    def make_feature_patch(self,i):
        def hook(module,args,h):
            assert h.shape[1]==128 and self.patch_valid is not None
            return h+self.residual_branches[i](h,self.patch_valid)
        return hook
    def forward(self,x,relative,valid,**kwargs):
        self.patch_valid=valid
        try:return super().forward(x,relative,valid,**kwargs)
        finally:self.patch_valid=None
    def groups(self):
        return [dict(params=list(self.context_head.parameters()),lr=3e-5,name='PERM_preserved_warm154_readout'),dict(params=list(self.residual_branches.parameters()),lr=3e-4,name='PERM_parallel_enc2_feature_residual')]
    def initialize_priors(self,data):
        return dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,initial_function_expected154=True,
            trained154_readout_loaded=True,original_readout_architecture_retained_and_jointly_updated=True,
            original_backbone_parameters_frozen=True,original_enc2_branch_retained=True,new_two_parallel_feature_branches=True,
            encoder_input=128,new_width=32,stored_depth_kernel=17,effective_depth_kernel=17 if self.mode=='local' else 1,
            new_encoder_and_zero128feature_projection_parameters=22208,total_trainable=104928,zero_exit=True,feature_residual_bound=.25,
            native_forward_grad_enabled=True,capture_detach_removed=True,backbone_eval_dropout_disabled=True,
            original_packed_output_router_weights_frozen=True,gate_scale_functions_can_change_with_features=True,
            original_Gaussian16_NLL_BCE_RPS1_decoder=True,one_added_PERM_stage=True,no_trained_repair_weights=True,
            historical349rank4negative=True,historical356outputresidualnegative=True,
            primary_control='local17 vs point sum same storedkernel/same104928params/init; effectivefunction differs')

def make_model(stage,master,metadata):return Network(master,metadata,stage)
def channel_objective(output,labels,default,mask,quality,c,mode,model,well_ids):
    return original_loss(output,labels,default,mask,quality,c,'perm_conditional',model)
