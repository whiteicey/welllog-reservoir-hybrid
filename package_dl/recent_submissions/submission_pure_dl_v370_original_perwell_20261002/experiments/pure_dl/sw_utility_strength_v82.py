"""Two predefined joint-utility strengths around the verified alpha=.5 control."""
import numpy as np
from sw_full_utility_v81 import ROOT,Network as Parent,Evaluator,load_data,SOURCE
from sw_head_utility_v80r1 import DecisionObjective as ParentObjective

STUDY=ROOT/'results/deep_learning/sw_utility_strength_v82_20260930'
PREVIOUS=ROOT/'results/deep_learning/sw_full_utility_v81_20260930'
CONTROL=ROOT/'outputs/pure_dl_sw_full_utility_v81_SW_joint_fold5'
MODES={mode:dict(target='SW',c=2,length=1024,rate=.4,utility_weight=alpha)
       for mode,alpha in (('a025',.25),('a100',1.))}

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

class Network(Parent):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('joint',metadata);self.study_mode=mode
        self.parent_info.update(utility_weight=MODES[mode]['utility_weight'],
            historical_alpha05_reference=str(CONTROL.relative_to(ROOT)),
            historical_control_reused=True,loaded_v81_trained_weights=False,
            only_hyperparameter_changed='joint utility coefficient .25/.5/1; .5 historical completed control; same original v56 body and reset SW prior')

class DecisionObjective(ParentObjective):
    def __init__(self,evaluator,metadata,mode):
        super().__init__(evaluator,metadata,'joint')
        self.alpha=MODES[mode]['utility_weight']
