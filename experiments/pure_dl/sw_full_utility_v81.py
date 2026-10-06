"""Matched full SW training with the retained DropKey recipe and joint-action loss."""
import numpy as np
from sw_dropout_extend_v77 import ROOT,Network as Parent,Evaluator
from sw_key_dropout_v74 import SOURCE
from sw_head_utility_v80r1 import load_data,DecisionObjective

STUDY=ROOT/'results/deep_learning/sw_full_utility_v81_20260930'
PREVIOUS=ROOT/'results/deep_learning/sw_head_utility_v80r1_20260930'
MODES={mode:dict(target='SW',c=2,length=1024,rate=.4,utility_weight=alpha) for mode,alpha in (('likelihood',0.),('joint',.5))}

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

class Network(Parent):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('p40',metadata);self.study_mode=mode
        for name in ('historical_control','historical_control_retrained','only_hyperparameter_changed'):
            self.parent_info.pop(name,None)
        self.parent_info.update(initialization='v56 additive backbone/FiLM with SW output reset to training prior',
            loaded_v77_trained_weights=False,loaded_v80_trained_weights=False,
            historical_p40_reference='outputs/pure_dl_sw_dropout_extend_v77_SW_p40_fold5',
            matched_full_loss_comparison=True,all_network_trainable=True,
            only_hyperparameter_changed='joint utility coefficient 0 versus .5; common loss graph and one model forward per batch')
