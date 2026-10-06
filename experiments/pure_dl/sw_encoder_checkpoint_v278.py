"""Copied104 feature backbone jointly trained with direct continuous categorical readout."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from sw_direct_component_v275 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,DecisionObjective,channel_objective,Network as Parent
from por_t3_patch_loss import probability
STUDY=ROOT/'results/deep_learning/sw_encoder_checkpoint_v278_20261001'
OUT=ROOT/'outputs/pure_dl_sw_encoder_checkpoint_v278_fold5'
STAGES={k:('SW',1024,False,0.,3e-4) for k in ('full',)}
class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,'mix');self.mode=mode
        del self.stem,self.branches,self.fuse,self.readout
        self.encoder=copy.deepcopy(self.base);self.encoder.requires_grad_(False).eval()
        self.encoder.output._forward_pre_hooks.clear()
        if mode=='full':
            self.encoder.requires_grad_(True);self.encoder.output.requires_grad_(False)
        else:
            self.encoder.body.detail.requires_grad_(True);self.encoder.body.fuse.requires_grad_(True);self.encoder.norm.requires_grad_(True)
        self.hidden=None;self.encoder.output.register_forward_pre_hook(self.capture)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261277);self.readout=nn.Sequential(nn.Linear(128,48),nn.GELU(),nn.Linear(48,256))
    def capture(self,module,args):self.hidden=args[0]
    def train(self,mode=True):super().train(mode);self.encoder.eval();return self
    def groups(self):return [dict(params=[p for p in self.encoder.parameters() if p.requires_grad],lr=1e-5,name='SW_'+self.mode+'_feature_backbone'),dict(params=list(self.readout.parameters()),lr=3e-4,name='SW_direct_categorical_readout')]
    def initialize_priors(self,data):
        result=super().initialize_priors(data);result.update(mode=self.mode,feature_encoder_initialized_from104=True,old_output_frozen_unused=True,auxiliary_features_count=1,new_raw_encoder_and_direct_readout_jointly_trainable=False,copied_feature_backbone_and_direct_readout_jointly_trainable=True,new_active_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),full_or_terminal_scope= self.mode,encoder_dropout_eval_with_gradients=True,no275_or252_weights_loaded=True,not_v260_mass_correction=True,activation_checkpoint_full_encoder=True,failed277_no_trained_weights_loaded=True)
        return result
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.aux_features=None;kwargs=dict(absolute_depth=absolute_depth,well_inputs=well_inputs)
        with torch.no_grad():
            gate,raw=self.base(x,relative,valid,**kwargs);flat=raw[:,:,0].float().reshape(-1,3,8)
            prior=torch.cat([probability(2,c[:,0].softmax(-1),c[:,1],F.softplus(c[:,2])+.025,self.edges) for c in flat.split(4096)]).reshape(*x.shape[:2],256)
        def encode(xx,rr,vv,ww):
            self.hidden=None;self.encoder(xx,rr,vv,absolute_depth=absolute_depth,well_inputs=ww);hh=self.hidden;self.hidden=None;return hh
        if self.training and torch.is_grad_enabled():h=checkpoint(encode,x,relative,valid,well_inputs,use_reentrant=False)
        else:h=encode(x,relative,valid,well_inputs)
        logits=self.readout(h).float()
        if self.training:self.aux_features=[h]
        prior=prior.clamp_min(1e-12);prior=prior/prior.sum(-1,keepdim=True)
        mass=(1-self.mix_alpha)*prior+self.mix_alpha*logits.softmax(-1);mass=mass.clamp_min(1e-12);mass=mass/mass.sum(-1,keepdim=True)
        return gate,mass.log().unsqueeze(2)
def make_model(stage,master,metadata):return Network(master,metadata,stage)
