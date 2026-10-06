"""Matched full SW fine-tuning: AdamW versus two-pass L2 SAM."""
import hashlib,math
import numpy as np
import torch
from torch.nn import functional as F
from well_conditioned_v46 import ROOT,Network as Parent,Evaluator,load_data

STUDY=ROOT/'results/deep_learning/sw_sharpness_v55_20260929'
PREVIOUS=ROOT/'results/deep_learning/sw_stratified_v54_20260929'
PARENT=ROOT/'outputs/pure_dl_well_conditioned_v46_SW_well_fold5/model_best.pt'
PARENT_PRED=PARENT.with_name('channel_pred.npy')
MODES={m:dict(target='SW',c=2,length=1024) for m in ('adamw','sam')}

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

class Network(Parent):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('well',metadata)
        saved=torch.load(PARENT,map_location='cpu',weights_only=False)
        self.load_state_dict(saved['state_dict'],strict=True);self.study_mode=mode
        self.initialization_info=dict(checkpoint=str(PARENT.relative_to(ROOT)),
            sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),best_epoch=saved['best_epoch'],
            target_score=saved['target_score'],optimizer_reset=True,
            initialization='exact v46 SW private predictor, all parameters trainable; no output reset')
    def initialize_priors(self,data):return self.initialization_info

