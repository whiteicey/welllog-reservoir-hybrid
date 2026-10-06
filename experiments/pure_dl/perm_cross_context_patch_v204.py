"""Input-only cross-context repair, inspired by cross-attention, not a CATS reproduction."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from perm_ranked_probability_v154 import ROOT,SOURCE,Network as Positive,load_data,Evaluator,channel_objective
STUDY=ROOT/'results/deep_learning/perm_cross_context_patch_v204_20261001'
OUT=ROOT/'outputs/pure_dl_perm_cross_context_patch_v204_fold5'
PARENT=ROOT/'outputs/pure_dl_perm_ranked_probability_v154_fold5/perm_conditional/model_best.pt'
PARENT_SHA='2b355bb8c3736990b05e2726373a4343c574a93ab6ab1a3eda0ef796ac70cf11'
STAGES={'attention':('PERM',256,False,0.,1e-4)}

class ContextPatch(nn.Module):
    def __init__(self):
        super().__init__();self.norm=nn.LayerNorm(128)
        self.query=nn.Linear(128,16);self.key=nn.Linear(128,16);self.value=nn.Linear(128,16)
        self.output=nn.Linear(16,32)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,feature,valid):
        h=self.norm(feature.transpose(1,2)).masked_fill(~valid[...,None],0)
        counts=F.adaptive_avg_pool1d(valid[:,None].float(),16)
        tokens=(F.adaptive_avg_pool1d(h.transpose(1,2),16)/counts.clamp_min(1e-6)).transpose(1,2)
        score=(self.query(h)@self.key(tokens).transpose(1,2)).float()/4.
        score=score.masked_fill(~(counts[:,0]>0)[:,None,:],-1e4)
        read=score.softmax(-1).to(h.dtype)@self.value(tokens)
        return self.output(read).float().masked_fill(~valid[...,None],0)

class Network(Positive):
    def __init__(self,master,mode):
        super().__init__(master,mode)
        assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
        self.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=False)['state_dict'],strict=True)
        self.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010);self.patch=ContextPatch()
        self.feature=None
        self.handle=self.context_head.output.register_forward_pre_hook(self.capture)
    def capture(self,module,args):self.feature=args[0].detach()
    def groups(self):return [dict(params=list(self.patch.parameters()),lr=1e-4,name='PERM_cross_context_patch_only')]
    def initialize_priors(self,data):
        return dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,old_predictor_fully_frozen=True,
                    same_initial_predictive_function=True,extra_active_parameters=sum(p.numel() for p in self.patch.parameters()),
                    memory_tokens=16,query_width=16,input_features_only=True,no_history_target_labels=True,
                    own_NLL_BCE_jointRPS_retained=True,executed_training_stages_including_source=6,
                    added_PERM_stages=1,no198_202_weights_loaded=True,not_CATS_reproduction=True,
                    location_bound_reference_sd=.1,logweight_bound=.1)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.feature=None
        with torch.no_grad():gate,raw=super().forward(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
        delta=self.patch(self.feature,valid);self.feature=None;own=raw[:,:,0]
        sd=F.softplus(own[:,:,2].detach())+.025
        location=.1*sd*torch.tanh(delta[...,:16]/(.1*sd))
        weights=.1*torch.tanh(delta[...,16:]/.1)
        return gate,torch.stack((own[:,:,0]+weights,own[:,:,1]+location,own[:,:,2]),2).unsqueeze(2)

def make_model(stage,master,metadata):return Network(master,stage)
