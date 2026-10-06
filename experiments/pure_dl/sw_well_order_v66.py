"""Frozen SW predictor with matched order-sensitive whole-well residual condition."""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sw_interaction_v56 import ROOT,Network as Parent,Evaluator,load_data
from well_conditioned_v46 import WIDTHS
STUDY=ROOT/'results/deep_learning/sw_well_order_v66_20260929'
PREVIOUS=ROOT/'results/deep_learning/sw_transition_v65_20260929'
SOURCE=ROOT/'outputs/pure_dl_sw_interaction_v56_SW_additive_fold5/model_best.pt'
PARENT_PRED=SOURCE.with_name('channel_pred.npy')
MODES={m:dict(target='SW',c=2,length=1024) for m in ('permuted','ordered')}
def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')
class WellResidual(nn.Module):
    def __init__(self,old,mode):
        super().__init__();self.base=old;self.study_mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261166)
            self.conv=nn.Conv1d(64,32,3,padding=1)
            self.output=nn.Linear(64,2*sum(WIDTHS))
            nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
            self.register_buffer('permutation',torch.randperm(128))
    def forward(self,inputs):
        with torch.no_grad():
            original=self.base(inputs);z=self.base.point(inputs)
        if self.study_mode=='permuted':z=z[:,self.permutation]
        h=F.gelu(self.conv(z.transpose(1,2)))
        stats=torch.cat((h.mean(-1),h.var(-1,unbiased=False).add(1e-6).sqrt()),-1)
        delta=.1*torch.tanh(self.output(stats).float())
        return tuple(a+b for a,b in zip(original,delta.split([2*w for w in WIDTHS],-1)))
class Network(Parent):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('additive',metadata)
        saved=torch.load(SOURCE,map_location='cpu',weights_only=False)
        self.load_state_dict(saved['state_dict'],strict=True);self.study_mode=mode;self.requires_grad_(False)
        self.condition=WellResidual(self.condition,mode)
        self.initialization_info=dict(checkpoint=str(SOURCE.relative_to(ROOT)),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            best_epoch=saved['best_epoch'],target_score=saved['target_score'],optimizer_reset=True,
            initialization='exact v56 additive predictor; zero output whole-well residual FiLM',
            new_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
            comparison='ordered versus fixed input-independent permutation of same128 well-bank hidden points; same tensors')
    def initialize_priors(self,data):return self.initialization_info
    def core_hash(self):
        trainable={n for n,p in self.named_parameters() if p.requires_grad}
        return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for n,v in self.state_dict().items() if n not in trainable)).hexdigest()
    def train(self,mode=True):
        super().train(False);self.training=mode;return self
