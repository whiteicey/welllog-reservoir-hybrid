"""Fixed 30%/40% continuation after the adopted 20% DropKey gain."""
import numpy as np
from sw_dropout_rate_v76 import ROOT,Network as Parent,Evaluator,load_data,ZERO

STUDY=ROOT/'results/deep_learning/sw_dropout_extend_v77_20260930'
PREVIOUS=ROOT/'results/deep_learning/sw_dropout_rate_v76_20260930'
CONTROL=ROOT/'outputs/pure_dl_sw_dropout_rate_v76_SW_p20_fold5'
MODES={'p30':dict(target='SW',c=2,length=1024,rate=.3),'p40':dict(target='SW',c=2,length=1024,rate=.4)}

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

class Network(Parent):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('p20',metadata)
        self.study_mode=mode;rate=MODES[mode]['rate']
        for block in self.body.global_blocks:block.drop_probability=rate
        self.parent_info.update(training_key_drop_probability=rate,historical_control=str(CONTROL.relative_to(ROOT)),
            historical_control_retrained=False,loaded_v76_trained_weights=False,
            only_hyperparameter_changed='training key drop probability; same original v56 weights and reset prior')
