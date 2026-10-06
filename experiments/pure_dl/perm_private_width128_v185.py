"""One bounded private PERM backbone width96to128 comparison, retain SAM/sampling."""
import torch
from torch import nn
from torch.nn import functional as F
from divergent_models_v45 import LongSection,Attention
from independent_sw_scratch_v10 import TemporalBlock
from perm_well_sam_v177 import ROOT,SOURCE,PARENT,STAGES,Network as Parent,load_data,Evaluator,channel_objective,prepare_sampling,sample_anchors
STUDY=ROOT/'results/deep_learning/perm_private_width128_v185_20261001'
OUT=ROOT/'outputs/pure_dl_perm_private_width128_v185_fold5'
class WiderBody(LongSection):
    def __init__(self):
        nn.Module.__init__(self)
        self.stem=nn.Linear(207,128)
        self.local=nn.ModuleList([TemporalBlock(128,d) for d in (1,2,4)])
        self.token_pos=nn.Linear(2,128);self.global_blocks=nn.ModuleList([Attention(128) for _ in range(4)])
        self.fuse=nn.Linear(256,128);self.detail=nn.ModuleList([TemporalBlock(128,d) for d in (1,2)])
    def forward(self,x,relative,valid,generator=None):
        # Parent operations are unchanged; its96 pooled reshape becomes128.
        return wide_forward(self,x,relative,valid,generator)
class Network(Parent):
    def __init__(self,master,mode):
        super().__init__(master,mode)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010);self.body=WiderBody()
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(TCN_width=128,token_attention_width=128,final_hidden_width=128,control_width=96,
                    no_trained_v177_weights_loaded=True,no_teacher=True,no_well_conditioning=True,
                    same_initial_distribution_priors=True,random_hidden_initialization_differs=True)
        return info
def make_model(stage,master,metadata):return Network(master,stage)

def wide_forward(self,x,relative,valid,generator=None):
    b,l,_=x.shape
    h=F.gelu(self.stem(torch.cat((x,relative[...,None]),-1))).masked_fill(~valid[...,None],0)
    for block in self.local:h=block(h,valid)
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
