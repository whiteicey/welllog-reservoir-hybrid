"""Combine the adopted additive backbone and retained smooth density mechanism."""
import hashlib
import numpy as np
import torch
from torch import nn
from sw_interaction_v56 import ROOT,Network as Parent,load_data
from quantile_density_v48 import Network as Quantile,Evaluator,channel_objective,K

STUDY=ROOT/'results/deep_learning/sw_rqs_staging_v58_20260929'
PREVIOUS=ROOT/'results/deep_learning/sw_ffgrowth_v57_20260929'
SOURCE=ROOT/'outputs/pure_dl_sw_interaction_v56_SW_additive_fold5/model_best.pt'
MODES={m:dict(target='SW',c=2,length=1024) for m in ('joint','headstart')}

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

class Network(Parent):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('additive',metadata)
        saved=torch.load(SOURCE,map_location='cpu',weights_only=False)
        self.load_state_dict(saved['state_dict'],strict=True)
        self.mode='rqs';self.study_mode=mode
        self.gate=nn.Linear(128,1)
        with torch.no_grad():
            self.gate.weight.copy_(self.output.weight[:1]);self.gate.bias.copy_(self.output.bias[:1])
        self.output=nn.Linear(128,K+2);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.parent_info=dict(checkpoint=str(SOURCE.relative_to(ROOT)),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            best_epoch=saved['best_epoch'],target_score=saved['target_score'],optimizer_reset=True,
            initialization='v56 additive body/norm/FiLM/default gate retained; new RQS density fitted to training-only empirical prior; not parent function',
            density_implementation='unchanged quantile_density_v48.py RQS CDF/exact logpdf/harmonic slopes/exponential tails')
    initialize_priors=Quantile.initialize_priors
    forward=Quantile.forward
