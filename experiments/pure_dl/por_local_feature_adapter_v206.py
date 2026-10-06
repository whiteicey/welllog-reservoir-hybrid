"""Preserve trained151 density readout; add only bounded local input-feature adapter."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from por_ranked_probability_v151 import ROOT,SOURCE,Network as Positive,load_data,Evaluator,channel_objective
STUDY=ROOT/'results/deep_learning/por_local_feature_adapter_v206_20261001'
OUT=ROOT/'outputs/pure_dl_por_local_feature_adapter_v206_fold5'
PARENT=ROOT/'outputs/pure_dl_por_ranked_probability_v151_fold5/joint_rps/model_best.pt'
PARENT_SHA='39f2255b8fe61fa74607af027ece1dd352e5e2d0729b9bce95d1424c1b227603'
STAGES={'local':('POR',256,False,0.,1e-4)}

class LocalAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Conv1d(319,32,1)
        self.depth=nn.Conv1d(32,32,9,padding=4,groups=32)
        self.output=nn.Conv1d(32,319,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,x,valid):
        mask=valid[:,None]
        # Local masked input RMS uses inputs only and no fitted validation statistics.
        count=F.avg_pool1d(mask.float(),9,stride=1,padding=4,count_include_pad=True)*9
        rms=(F.avg_pool1d(x.detach().float().square().masked_fill(~mask,0),9,stride=1,padding=4,count_include_pad=True)*9/count.clamp_min(1)).sqrt().clamp_min(.01)
        normalized=(x/rms.to(x.dtype)).masked_fill(~mask,0)
        h=F.gelu(self.input(normalized)).masked_fill(~mask,0)
        h=F.gelu(self.depth(h)).masked_fill(~mask,0)
        delta=self.output(h).float()
        return (.1*rms*torch.tanh(delta/.1)).masked_fill(~mask,0)

class AdaptedHead(nn.Module):
    def __init__(self,old):
        super().__init__();self.old=old;self.patch=LocalAdapter()
    def forward(self,features,valid):
        return self.old(features+self.patch(features,valid).to(features.dtype),valid)

class Network(Positive):
    def __init__(self,master,mode):
        super().__init__(master,mode)
        assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
        self.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=False)['state_dict'],strict=True)
        self.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010);self.context_head=AdaptedHead(self.context_head)
    def groups(self):return [dict(params=list(self.context_head.patch.parameters()),lr=1e-4,name='POR_local_feature_adapter_only')]
    def initialize_priors(self,data):
        return dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,trained151_all_weights_loaded_and_frozen=True,
                    retained_original_readout=True,same_initial_predictive_function=True,
                    local_kernel=9,adapter_width=32,adapter_input=319,input_bound_local_RMS=.1,
                    local_RMS_input_only_masked=True,old_gate_scale_preserved=True,
                    extra_active_parameters=sum(p.numel() for p in self.context_head.patch.parameters()),
                    own_NLL_BCE_jointRPS_retained=True,executed_training_stages_including_source=6,
                    added_POR_stages=1,no_prior_repair_weights_loaded=True)

def make_model(stage,master,metadata):return Network(master,stage)
