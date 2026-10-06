"""Separate SW density backbone from a preserved default-state predictor."""
import hashlib
import numpy as np
import torch
from torch import nn
from sw_interaction_v56 import ROOT,Network as Parent,Evaluator,load_data

STUDY=ROOT/'results/deep_learning/sw_density_private_v62_20260929'
PREVIOUS=ROOT/'results/deep_learning/sw_upsample_v61_20260929'
SOURCE=ROOT/'outputs/pure_dl_sw_interaction_v56_SW_additive_fold5/model_best.pt'
PARENT_PRED=SOURCE.with_name('channel_pred.npy')
MODES={m:dict(target='SW',c=2,length=1024) for m in ('gate_aux','numeric_only')}

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

class Network(nn.Module):
    def __init__(self,mode,metadata):
        super().__init__()
        if mode not in MODES:raise ValueError(mode)
        self.mode=mode;self.study_mode=mode;self.length=1024;self.c=2
        saved=torch.load(SOURCE,map_location='cpu',weights_only=False)
        self.fixed=Parent('additive',metadata);self.density=Parent('additive',metadata)
        for branch in (self.fixed,self.density):branch.load_state_dict(saved['state_dict'],strict=True)
        for p in self.fixed.parameters():p.requires_grad_(False)
        self.initialization_info=dict(checkpoint=str(SOURCE.relative_to(ROOT)),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            best_epoch=saved['best_epoch'],target_score=saved['target_score'],optimizer_reset=True,
            initialization='two exact v56 copies; fixed branch supplies gate, private trainable branch supplies density',
            training_auxiliary_default_weight=1. if mode=='gate_aux' else 0.,
            trainable_body='all density branch backbone, FiLM and output',
            unused_training_output_parameters=129 if mode=='numeric_only' else 0,
            inference='fixed default logit plus private density; no prediction ensemble or external target model')
    @property
    def body(self):return self.density.body
    @property
    def output(self):return self.density.output
    @property
    def condition(self):return self.density.condition
    def initialize_priors(self,data):return self.initialization_info
    def core_hash(self):
        return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in self.fixed.state_dict().values())).hexdigest()
    def forward(self,x,relative,valid,absolute_depth=None,return_aux=False,generator=None,well_inputs=None):
        self.fixed.eval()
        with torch.no_grad():gate,_=self.fixed(x,relative,valid,well_inputs=well_inputs)
        own_gate,raw=self.density(x,relative,valid,well_inputs=well_inputs)
        output=(gate,raw)
        return (output,own_gate,None) if return_aux else output
