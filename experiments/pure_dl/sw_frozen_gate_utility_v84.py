"""A frozen SW default predictor and a private trainable conditional density."""
import hashlib
import numpy as np
import torch
from torch import nn
from sw_utility_strength_v82 import ROOT,Network as Parent,Evaluator
from sw_population_utility_v83 import load_data,DecisionObjective as ParentObjective

STUDY=ROOT/'results/deep_learning/sw_frozen_gate_utility_v84_20260930'
PREVIOUS=ROOT/'results/deep_learning/sw_population_utility_v83_20260930'
SOURCE=ROOT/'outputs/pure_dl_sw_utility_strength_v82_SW_a100_fold5/model_best.pt'
CONTROL=SOURCE.parent
MODES={mode:dict(target='SW',c=2,length=1024,rate=.4,utility_weight=alpha,numeric_share=.5)
       for mode,alpha in (('likelihood',0.),('balanced',1.))}

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

class Network(nn.Module):
    def __init__(self,mode,metadata):
        super().__init__()
        if mode not in MODES:raise ValueError(mode)
        self.study_mode=mode
        saved=torch.load(SOURCE,map_location='cpu',weights_only=False)
        self.fixed=Parent('a100',metadata);self.density=Parent('a100',metadata)
        for branch in (self.fixed,self.density):branch.load_state_dict(saved['state_dict'],strict=True)
        self.fixed.requires_grad_(False);self.fixed.eval()
        self.initialization_info=dict(checkpoint=str(SOURCE.relative_to(ROOT)),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            best_epoch=saved['best_epoch'],target_score=saved['target_score'],optimizer_reset=True,
            initialization='two exact current v82 alpha1 SW copies; frozen gate and private density; no prior reset',
            training_key_drop_probability=.4,evaluation_key_drop_probability=0.,fixed_default_branch_eval=True,
            utility_weight=MODES[mode]['utility_weight'],numeric_share=.5,unused_density_gate_parameters=129,
            inference='one SW default atom/conditional8Gaussian distribution; no voting or label-based population selection')
    @property
    def body(self):return self.density.body
    @property
    def output(self):return self.density.output
    @property
    def condition(self):return self.density.condition
    def initialize_priors(self,data):return self.initialization_info
    def core_hash(self):
        return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in self.fixed.state_dict().values())).hexdigest()
    def train(self,mode=True):
        super().train(mode);self.fixed.eval();return self
    def forward(self,x,relative,valid,absolute_depth=None,return_aux=False,generator=None,well_inputs=None):
        self.fixed.eval()
        with torch.no_grad():gate,_=self.fixed(x,relative,valid,well_inputs=well_inputs)
        _,raw=self.density(x,relative,valid,well_inputs=well_inputs)
        output=(gate,raw)
        return (output,raw.new_zeros(()),None) if return_aux else output

class DecisionObjective(ParentObjective):
    def __init__(self,evaluator,metadata,mode):
        super().__init__(evaluator,metadata,'balanced')
        self.alpha=MODES[mode]['utility_weight']
