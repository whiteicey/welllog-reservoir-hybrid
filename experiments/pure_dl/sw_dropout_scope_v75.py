"""One shared SW model, matched two-pass dropout scope for gate versus density."""
import numpy as np
from sw_key_dropout_v74 import ROOT,Network as Parent,Evaluator,load_data

STUDY=ROOT/'results/deep_learning/sw_dropout_scope_v75_20260929'
PREVIOUS=ROOT/'results/deep_learning/sw_key_dropout_v74_20260929'
MODES={m:dict(target='SW',c=2,length=1024) for m in ('both','gate_only')}

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

class Network(Parent):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('dropkey',metadata)
        self.study_mode=mode
        self.parent_info.update(training_scope=mode,forward_passes_per_training_batch=2,
            gate_loss='dropkey10percent',density_loss='dropkey10percent' if mode=='both' else 'full clean context',
            shares_all_parameters_across_passes=True,validation='one clean forward; unchanged full prediction')

    def set_key_dropout(self,enabled):
        for block in self.body.global_blocks:block.use_key_dropout=bool(enabled)
