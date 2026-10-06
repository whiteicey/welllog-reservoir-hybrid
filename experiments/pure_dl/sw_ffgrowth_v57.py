"""Function-preserving parallel capacity growth of the adopted additive SW FF."""
import hashlib,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sw_interaction_v56 import ROOT,Network as Parent,Evaluator,load_data

STUDY=ROOT/'results/deep_learning/sw_ffgrowth_v57_20260929'
PREVIOUS=ROOT/'results/deep_learning/sw_interaction_v56_20260929'
SOURCE=ROOT/'outputs/pure_dl_sw_interaction_v56_SW_additive_fold5/model_best.pt'
PARENT_PRED=SOURCE.with_name('channel_pred.npy')
MODES={m:dict(target='SW',c=2,length=1024) for m in ('base','grow128')}

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

class ExtraFF(nn.Module):
    def __init__(self):
        super().__init__();self.input=nn.Linear(96,256);self.norm=nn.LayerNorm(128);self.output=nn.Linear(128,96)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,x):
        a,b=self.input(x).chunk(2,-1)
        return self.output(self.norm((F.silu(a)+b)/math.sqrt(2)))

class GrownFF(nn.Module):
    def __init__(self,base,grow):
        super().__init__();self.base=base;self.extra=ExtraFF() if grow else None
    def forward(self,x):
        value=self.base(x)
        return value if self.extra is None else value+self.extra(x)

class Network(Parent):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('additive',metadata)
        saved=torch.load(SOURCE,map_location='cpu',weights_only=False)
        self.load_state_dict(saved['state_dict'],strict=True);self.study_mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261157)
            for block in self.body.global_blocks:block.ff=GrownFF(block.ff,mode=='grow128')
        self.initialization_info=dict(checkpoint=str(SOURCE.relative_to(ROOT)),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            source_best_epoch=saved['best_epoch'],source_target_score=saved['target_score'],optimizer_reset=True,
            initialization='exact adopted v56 additive function; old FF256 retained; optional parallel FF128 has zero output',
            scope='all parameters learn with common LR, own SW training labels only',
            expansion='four parallel128 branches with their own LayerNorm; NOT a single384-wide LayerNorm')
    def initialize_priors(self,data):return self.initialization_info
    def core_hash(self):
        return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for n,v in self.state_dict().items() if '.extra.' not in n)).hexdigest()
