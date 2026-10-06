"""Limited DCN-style raw-encoder migration; matched cross/additive pair."""
import torch
from torch import nn
from torch.nn import functional as F
from por_direct_share_v290 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent
from compact_por_mixed_family_v113 import probability
STUDY=ROOT/'results/deep_learning/por_anchored_cross_v306_20261002'
OUT=ROOT/'outputs/pure_dl_por_anchored_cross_v306_fold5'
STAGES={a:('POR',256,False,0.,3e-4) for a in ('cross','additive')}

class AnchoredBlock(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode
        self.down=nn.Linear(48,8,bias=False);self.up=nn.Linear(8,48,bias=False);self.bias=nn.Parameter(torch.zeros(48))
        nn.init.zeros_(self.up.weight)
    def forward(self,h,h0):
        delta=self.up(self.down(h))+self.bias
        return h+.1*(h0*delta if self.mode=='cross' else delta)

class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,'raw');self.interaction_mode=mode
        # Remove inherited fresh three-branch encoder; keep protected original151.
        del self.branches
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261306)
            self.stem=nn.Sequential(nn.Linear(207,48),nn.LayerNorm(48))
            self.cross_blocks=nn.ModuleList([AnchoredBlock(mode) for _ in range(2)])
            self.fuse=nn.Sequential(nn.Linear(96,48),nn.GELU())
            self.tail=nn.Sequential(nn.Linear(48,96),nn.GELU(),nn.Linear(96,48))
    def groups(self):
        body=list(self.stem.parameters())+list(self.cross_blocks.parameters())+list(self.fuse.parameters())+list(self.tail.parameters())
        return [dict(params=body,lr=3e-4,name='POR_new_anchored_raw_backbone'),dict(params=list(self.readout.parameters()),lr=3e-4,name='POR_direct_categorical_readout')]
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(raw_encoder='207->48 LN, two anchored rank8 .1blocks, concat h0/h2 ->48 GELU and residualMLP',interaction_mode=self.interaction_mode,new_active_parameters=38272,new_body_parameters=25728,auxiliary_readout_features=['h0','h1','h2','fused_tail'],three_branch_encoder_replaced=True,no290_or302_or305_trained_weights=True,same_registered_parameters_and_initial_function_pair=True,same_effective_function_family=False,not_first_DCN=True,zero_up_nonzero_down_initialization=True,new_blocks_first_step_down_gradient_expected_zero=True)
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.aux_features=None
        with torch.no_grad():
            gate,raw=self.base(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
            flat=raw[:,:,0].float().reshape(-1,3,16)
            prior=torch.cat([probability('t3_cauchy',0,c[:,0].softmax(-1),c[:,1],F.softplus(c[:,2])+.025,self.edges) for c in flat.split(4096)]).reshape(*x.shape[:2],256)
        inp=torch.cat((x[...,:206],relative.unsqueeze(-1)),dim=-1)
        h0=self.stem(inp).masked_fill(~valid.unsqueeze(-1),0);h=h0;features=[h0]
        for block in self.cross_blocks:
            h=block(h,h0).masked_fill(~valid.unsqueeze(-1),0);features.append(h)
        fused=self.fuse(torch.cat((h0,h),-1));fused=(fused+self.tail(fused)).masked_fill(~valid.unsqueeze(-1),0)
        features.append(fused)
        if self.training:self.aux_features=features
        logits=self.readout(fused).float();prior=prior.clamp_min(1e-12);prior=prior/prior.sum(-1,keepdim=True)
        mass=(1-self.mix_alpha)*prior+self.mix_alpha*logits.softmax(-1)
        mass=mass.clamp_min(1e-12);mass=mass/mass.sum(-1,keepdim=True)
        return gate,mass.log().unsqueeze(2)

def make_model(stage,master,metadata):return Network(master,metadata,stage)
