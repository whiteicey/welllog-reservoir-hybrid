"""Full expert feature encoder jointly learns directly supervised conditional branch."""
import copy,hashlib
import torch
from torch import nn
from torch.nn import functional as F
from perm_direct_multiscale_v271 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as DirectParent
from por_t3_patch_loss import probability
STUDY=ROOT/'results/deep_learning/perm_full_encoder_v272_20261001'
OUT=ROOT/'outputs/pure_dl_perm_full_encoder_v272_fold5'
STAGES={'warm_full':('PERM',256,False,0.,3e-4)}
class Network(DirectParent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,'repeat');self.mode=mode
        del self.stem,self.branches,self.fuse,self.readout
        self.encoders=nn.ModuleList([copy.deepcopy(m) for m in self.base.backbone.members]);self.encoder_hidden={};self.encoder_handles=[]
        self.encoders.requires_grad_(True).eval()
        for i,m in enumerate(self.encoders):
            # Copied parent hooks close over the frozen parent; replace with gradient-enabled capture.
            m.main_head._forward_pre_hooks.clear();m.main_head.requires_grad_(False);m.aux_mid.requires_grad_(False);m.aux_coarse.requires_grad_(False)
            self.encoder_handles.append(m.main_head.register_forward_pre_hook(self.capture(i)))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261272);self.component=nn.Sequential(nn.Linear(319,128),nn.GELU(),nn.Linear(128,256))
    def capture(self,i):
        def hook(module,args):self.encoder_hidden[i]=args[0].transpose(1,2)
        return hook
    def train(self,mode=True):super().train(mode);self.encoders.eval();return self
    def groups(self):return [dict(params=[p for p in self.encoders.parameters() if p.requires_grad],lr=1e-5,name='PERM_full_expert_feature_encoders'),dict(params=list(self.component.parameters()),lr=3e-4,name='PERM_direct_categorical_readout')]
    def initialize_priors(self,data):return dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,original154_prior_gate_frozen=True,full_copied_expert_encoders_trainable=True,old_main_aux_heads_frozen_unused_in_predictions=True,original_dropout_eval_with_gradient_enabled=True,own_direct_CE_independent_of_alpha=True,nonzero_component_readout=True,fixed_alpha_ramp_epochs=5,final_alpha=.1,initial_alpha=0.,no271_or_other_repair_weights_loaded=True,one_added_PERM_stage=True,original_FEATURE_views_retained=True,full_encoder_not_just_terminal_blocks=True,teacher_supervision=False)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.aux_logits=None;self.encoder_hidden.clear()
        with torch.no_grad():
            gate,raw=self.base(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
            flat=raw[:,:,0].float().reshape(-1,3,16)
            prior=torch.cat([probability(1,c[:,0].softmax(-1),c[:,1],F.softplus(c[:,2])+.025,self.edges) for c in flat.split(4096)]).reshape(*x.shape[:2],256)
        for m in self.encoders:m(x[...,:206].contiguous(),relative,valid,absolute_depth=absolute_depth)
        features=torch.cat((self.encoder_hidden[0],self.encoder_hidden[1],x[...,206:]),dim=-1)
        logits=self.component(features).float();self.encoder_hidden.clear()
        if self.training:self.aux_logits=[logits]
        mass=(1-self.mix_alpha)*prior+self.mix_alpha*logits.softmax(-1)
        mass=mass.clamp_min(1e-12);mass=mass/mass.sum(-1,keepdim=True)
        return gate,mass.log().unsqueeze(2)
def make_model(stage,master,metadata):return Network(master,metadata,stage)
