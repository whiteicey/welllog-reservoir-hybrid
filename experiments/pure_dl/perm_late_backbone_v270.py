"""Matched retained154 native head versus copied late backbone joint update."""
import copy,hashlib
import torch
from torch import nn
from torch.nn import functional as F
from perm_ranked_probability_v154 import ROOT,SOURCE,Network as Parent,load_data,Evaluator,channel_objective as original_loss
from compact_five_stage_v100 import CompactReadout
STUDY=ROOT/'results/deep_learning/perm_late_backbone_v270_20261001'
OUT=ROOT/'outputs/pure_dl_perm_late_backbone_v270_fold5'
PARENT=ROOT/'outputs/pure_dl_perm_ranked_probability_v154_fold5/perm_conditional/model_best.pt'
PARENT_SHA='2b355bb8c3736990b05e2726373a4343c574a93ab6ab1a3eda0ef796ac70cf11'
STAGES={a:('PERM',256,False,0.,3e-5) for a in ('head_only','late_joint')}
class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,'perm_conditional');self.mode=mode
        assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
        self.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=False)['state_dict'],strict=True);self.requires_grad_(False)
        self.late_dec0=nn.ModuleList([copy.deepcopy(m.dec0) for m in self.backbone.members]);self.late_combine=nn.ModuleList([copy.deepcopy(m.combine) for m in self.backbone.members])
        self.late_dec0.requires_grad_(mode=='late_joint');self.late_combine.requires_grad_(mode=='late_joint');self.context_head.requires_grad_(True)
        self.late_inputs={};self.late_points={};self.extra_hooks=[]
        self.last_group_risk=0.;self.last_group_count=0;self.last_max_group_weight=0.
        for i,m in enumerate(self.backbone.members):
            self.extra_hooks.append(m.dec0.register_forward_pre_hook(self.capture_late(i)))
            self.extra_hooks.append(m.combine.register_forward_pre_hook(self.capture_point(i)))
    def capture_late(self,i):
        def hook(module,args):self.late_inputs[i]=args[0].detach()
        return hook
    def capture_point(self,i):
        def hook(module,args):self.late_points[i]=args[0][:,64:].detach()
        return hook
    def train(self,mode=True):
        super().train(mode);self.backbone.eval();self.late_dec0.eval();self.late_combine.eval();return self
    def groups(self):
        groups=[dict(params=list(self.context_head.parameters()),lr=3e-5,name='PERM_retained154_native_readout')]
        if self.mode=='late_joint':groups.append(dict(params=list(self.late_dec0.parameters())+list(self.late_combine.parameters()),lr=1e-5,name='PERM_copied_actual_late_encoder'))
        return groups
    def initialize_priors(self,data):return dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,original_SOURCE_prior_gate_frozen=True,retained154_readout_updated=True,late_encoder_trainable=self.mode=='late_joint',new_copied_parameters=sum(p.numel() for p in self.late_dec0.parameters())+sum(p.numel() for p in self.late_combine.parameters()),initial_function_expected154=True,late_dropout_eval=True,one_added_PERM_stage=True,no_trained_repair_weights=True,continuous_native_Gaussian_NLL_BCE_joint_RPS1=True,same_registered_capacity_but_update_scope_differs=True)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.late_inputs.clear();self.late_points.clear();self.local_captured.clear()
        with torch.no_grad():gate,raw=CompactReadout.forward(self,x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
        hidden=[c(torch.cat((b(self.late_inputs[i]),self.late_points[i]),1)) for i,(b,c) in enumerate(zip(self.late_dec0,self.late_combine))]
        features=torch.cat((*hidden,x[...,206:].transpose(1,2)),1)
        u=self.context_head(features,valid);ref=F.softplus(raw[:,:,0,2].detach())+.025
        delta=ref*torch.tanh(u[...,:16]/ref);weight_delta=torch.tanh(u[...,16:]);own=raw[:,:,0]
        self.late_inputs.clear();self.late_points.clear();self.local_captured.clear()
        return gate,torch.stack((own[:,:,0]+weight_delta,own[:,:,1]+delta,own[:,:,2]),2).unsqueeze(2)
def make_model(stage,master,metadata):return Network(master,metadata,stage)
def channel_objective(output,labels,default,mask,quality,c,mode,model,well_ids):
    return original_loss(output,labels,default,mask,quality,c,'perm_conditional',model)
