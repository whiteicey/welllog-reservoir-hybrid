"""Training-only cross-well masked MixStyle on the positive private128 backbone."""
import torch
from torch import nn
from torch.nn import functional as F
from divergent_models_v45 import LongSection,Attention
from independent_sw_scratch_v10 import TemporalBlock
from perm_well_sam_v177 import ROOT,SOURCE,PARENT,STAGES,Network as Parent,load_data,Evaluator,channel_objective,prepare_sampling,sample_anchors
STUDY=ROOT/'results/deep_learning/perm_mixstyle_v189_20261001'
OUT=ROOT/'outputs/pure_dl_perm_mixstyle_v189_fold5'
def masked_mixstyle(h,valid,training_wells):
    assert training_wells is not None, 'Train-only well groups required for MixStyle'
    if bool(torch.rand((),device=h.device)>=.5):return h
    x=h.float();mask=valid[...,None].float();count=mask.sum(1,keepdim=True).clamp_min(1)
    mean=(x*mask).sum(1,keepdim=True)/count
    variance=((x-mean).square()*mask).sum(1,keepdim=True)/(count-1).clamp_min(1)
    mean=mean.detach();std=(variance+1e-6).sqrt().detach()
    allowed=(training_wells[:,None]!=training_wells[None,:]).float()
    no_partner=allowed.sum(1)==0
    allowed[no_partner]=torch.eye(len(h),device=h.device)[no_partner]
    partner=torch.multinomial(allowed,1).squeeze(1)
    concentration=torch.tensor(.1,device=h.device)
    lam=torch.distributions.Beta(concentration,concentration).sample((len(h),1,1))
    mixed_mean=lam*mean+(1-lam)*mean[partner]
    mixed_std=lam*std+(1-lam)*std[partner]
    styled=(x-mean)/std*mixed_std+mixed_mean
    return styled.to(h.dtype).masked_fill(~valid[...,None],0)

class WiderBody(LongSection):
    def __init__(self):
        nn.Module.__init__(self)
        self.stem=nn.Linear(207,128)
        self.local=nn.ModuleList([TemporalBlock(128,d) for d in (1,2,4)])
        self.token_pos=nn.Linear(2,128);self.global_blocks=nn.ModuleList([Attention(128) for _ in range(4)])
        self.fuse=nn.Linear(256,128);self.detail=nn.ModuleList([TemporalBlock(128,d) for d in (1,2)])
    def forward(self,x,relative,valid,generator=None,training_wells=None):
        # Parent operations are unchanged; its96 pooled reshape becomes128.
        return wide_forward(self,x,relative,valid,generator,training_wells)
class Network(Parent):
    def __init__(self,master,mode):
        super().__init__(master,mode)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010);self.body=WiderBody()
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(TCN_width=128,token_attention_width=128,final_hidden_width=128,control_width=128,
                    no_trained_v177_weights_loaded=True,no_teacher=True,no_well_conditioning=True,
                    same_initial_distribution_priors=True,random_hidden_initialization_differs=False,
                    mixstyle_training_only=True,mixstyle_probability=.5,mixstyle_beta_alpha=.1,
                    mixstyle_after_first_local_block=True,partners_different_train_wells=True,
                    statistics_masked=True,statistics_detached=True,well_ID_not_prediction_feature=True,
                    SAM_mixstyle_rng_replayed=True,no_trained_v185_weights_loaded=True)
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None,training_wells=None):
        hidden,_,_=self.body(x[...,:206],relative,valid,training_wells=training_wells)
        return self.gate(hidden).float(),self.density(hidden).reshape(*hidden.shape[:2],1,3,16).float()

def make_model(stage,master,metadata):return Network(master,stage)

def wide_forward(self,x,relative,valid,generator=None,training_wells=None):
    b,l,_=x.shape
    h=F.gelu(self.stem(torch.cat((x,relative[...,None]),-1))).masked_fill(~valid[...,None],0)
    for i,block in enumerate(self.local):
        h=block(h,valid)
        if i==0 and self.training:h=masked_mixstyle(h,valid,training_wells)
    pv=valid.reshape(b,l//16,16);n=pv.sum(-1).clamp_min(1)
    token=(h.reshape(b,l//16,16,128)*pv[...,None]).sum(2)/n[...,None]
    positions=torch.linspace(-1,1,l,device=x.device).expand(b,-1)
    loc=torch.stack((relative,positions),-1).reshape(b,l//16,16,2)
    loc=(loc*pv[...,None]).sum(2)/n[...,None]
    tv=pv.any(-1);token=(token+self.token_pos(loc)).masked_fill(~tv[...,None],0)
    for block in self.global_blocks:token=block(token,tv)
    global_h=F.interpolate(token.transpose(1,2),size=l,mode='linear',align_corners=False).transpose(1,2)
    h=F.gelu(self.fuse(torch.cat((h,global_h),-1))).masked_fill(~valid[...,None],0)
    for block in self.detail:h=block(h,valid)
    return h,h.new_zeros(()),None
