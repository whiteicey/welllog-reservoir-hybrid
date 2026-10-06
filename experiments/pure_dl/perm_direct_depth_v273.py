"""Train fresh depth-local PERM backbone together with directly supervised density."""
import torch
from torch import nn
from torch.nn import functional as F
from perm_direct_multiscale_v271 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent
from por_t3_patch_loss import probability
STUDY=ROOT/'results/deep_learning/perm_direct_depth_v273_20261001'
OUT=ROOT/'outputs/pure_dl_perm_direct_depth_v273_fold5'
STAGES={'depth96':('PERM',256,False,0.,3e-4)}
class DepthBlock(nn.Module):
    def __init__(self,dilation):
        super().__init__();self.norm=nn.LayerNorm(96);self.depth=nn.Conv1d(96,96,9,padding=4*dilation,dilation=dilation,groups=96);self.expand=nn.Conv1d(96,192,1);self.project=nn.Conv1d(96,96,1)
    def forward(self,h,valid):
        z=self.norm(h.transpose(1,2)).transpose(1,2).masked_fill(~valid[:,None],0)
        a,b=self.expand(self.depth(z)).chunk(2,1)
        return (h+.1*self.project(F.gelu(a)*b.sigmoid())).masked_fill(~valid[:,None],0)
class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,'repeat');self.mode=mode
        del self.stem,self.branches,self.fuse,self.readout
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261273);self.stem=nn.Conv1d(207,96,1);self.blocks=nn.ModuleList([DepthBlock(d) for d in (1,4,16)]);self.readout=nn.Linear(96,256)
    def groups(self):return [dict(params=list(self.stem.parameters())+list(self.blocks.parameters()),lr=3e-4,name='PERM_new_depth_local_backbone'),dict(params=list(self.readout.parameters()),lr=3e-4,name='PERM_direct_categorical_readout')]
    def initialize_priors(self,data):return dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,all154_prior_gate_frozen=True,fresh_random_depth_backbone_trainable=True,width=96,depth_kernel=9,dilations=[1,4,16],direct_CE_weight=1.,nonzero_readout=True,alpha_ramp_epochs=5,final_alpha=.1,no271_trained_weights=True,one_PERM_additional_phase=True,capacity_and_architecture_differ_from271=True,input='original206+relative',teacher_supervision=False)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.aux_logits=None
        with torch.no_grad():
            gate,raw=self.base(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
            flat=raw[:,:,0].float().reshape(-1,3,16)
            prior=torch.cat([probability(1,c[:,0].softmax(-1),c[:,1],F.softplus(c[:,2])+.025,self.edges) for c in flat.split(4096)]).reshape(*x.shape[:2],256)
        inp=torch.cat((x[...,:206],relative.unsqueeze(-1)),-1);h=F.gelu(self.stem(inp.transpose(1,2))).masked_fill(~valid[:,None],0)
        for block in self.blocks:h=block(h,valid)
        logits=self.readout(h.transpose(1,2)).float()
        if self.training:self.aux_logits=[logits]
        mass=(1-self.mix_alpha)*prior+self.mix_alpha*logits.softmax(-1);mass=mass.clamp_min(1e-12);mass=mass/mass.sum(-1,keepdim=True)
        return gate,mass.log().unsqueeze(2)
def make_model(stage,master,metadata):return Network(master,metadata,stage)
