"""Matched direct SW output paths from reference/own input-only whole-well code."""
import hashlib
import numpy as np
import torch
from torch import nn
from sw_utility_strength_v82 import ROOT,Network as Parent,Evaluator,load_data,SOURCE,DecisionObjective as ParentObjective

STUDY=ROOT/'results/deep_learning/sw_direct_well_v90_20260930'
PREVIOUS=ROOT/'results/deep_learning/sw_student4_v89_20260930'
CONTROL=ROOT/'outputs/pure_dl_sw_utility_strength_v82_SW_a100_fold5'
MODES={m:dict(target='SW',c=2,length=1024,rate=.4,utility_weight=1.) for m in ('reference','well')}

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

class Network(Parent):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('a100',metadata);self.study_mode=mode;self.direct_mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261190);self.global_output=nn.Linear(96,25)
        nn.init.zeros_(self.global_output.weight);nn.init.zeros_(self.global_output.bias)
        self._own_code=None
        self._code_hook=self.condition.code.register_forward_hook(self._capture)
        self.parent_info.update(added_parameters=2425,only_hyperparameter_changed='direct output branch reads own versus training-reference code; existing FiLM always own',
            direct_context=mode,base_film_context='own well in both arms',both_codes_computed=True,
            historical_gaussian_control=str(CONTROL.relative_to(ROOT)),loaded_gaussian_control_weights=False,
            zero_initial_direct_output=True,matched_capacity_and_initial_function=True)
    def _capture(self,module,args,output):self._own_code=output
    def base_state_hash(self):
        return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for n,v in self.state_dict().items() if not n.startswith('global_output.'))).hexdigest()
    def close(self):
        self._code_hook.remove();self._own_code=None
    def forward(self,x,relative,valid,absolute_depth=None,return_aux=False,generator=None,well_inputs=None):
        self._own_code=None
        try:
            result=super().forward(x,relative,valid,absolute_depth=absolute_depth,return_aux=return_aux,generator=generator,well_inputs=well_inputs)
            own=self._own_code
            if own is None:raise RuntimeError('Missing existing whole-well code')
            z=self.condition.point(self.reference_inputs[None])
            stats=torch.cat((z.mean(1),z.var(1,unbiased=False).add(1e-6).sqrt(),z.amax(1)),1)
            ref=self.condition.code(stats)
            code=own if self.direct_mode=='well' else ref.expand(x.shape[0],-1)
            delta=self.global_output(code).float()
            output=result[0] if return_aux else result
            gate,raw=output
            adjusted=(gate+delta[:,None,:1],raw+delta[:,None,1:].reshape(x.shape[0],1,1,3,8))
            return (adjusted,result[1],result[2]) if return_aux else adjusted
        finally:self._own_code=None

class DecisionObjective(ParentObjective):
    def __init__(self,evaluator,metadata,mode):
        assert mode in MODES
        super().__init__(evaluator,metadata,'a100')
