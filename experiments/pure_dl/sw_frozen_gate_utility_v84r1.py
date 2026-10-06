"""Repair only the evaluator interface of the preserved failed v84 attempt."""
from sw_frozen_gate_utility_v84 import ROOT,PREVIOUS,CONTROL,SOURCE,MODES,Evaluator,load_data,reference_prediction,DecisionObjective,Network as Parent

STUDY=ROOT/'results/deep_learning/sw_frozen_gate_utility_v84r1_20260930'

class Network(Parent):
    def __init__(self,mode,metadata):
        super().__init__(mode,metadata)
        self.length=1024;self.c=2
        self.initialization_info.update(evaluation_interface_repair='length1024 and target index2; failed v84 had no epochs or validation score',
            original_failed_run='outputs/pure_dl_sw_frozen_gate_utility_v84_SW_likelihood_fold5')
