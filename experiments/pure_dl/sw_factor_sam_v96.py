"""Matched conditional-density objectives under the adopted fixed SAM atom."""
import hashlib
import numpy as np
import torch
from sw_factorization_v95 import ROOT,Network as Parent,Evaluator,load_data,GATE,DENSITY
from sw_utility_strength_v82 import DecisionObjective as ParentObjective
STUDY=ROOT/'results/deep_learning/sw_factor_sam_v96_20260930'
PREVIOUS=ROOT/'results/deep_learning/sw_factorization_v95_20260930'
MODES={m:dict(target='SW',c=2,length=1024,rate=.4,utility_weight=a,rho=.1) for m,a in (('likelihood',0.),('joint',1.))}
def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

class Network(Parent):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('separate',metadata);self.study_mode=mode
        self.density.requires_grad_(True)
        self.parent_info=dict(checkpoint=str(DENSITY.relative_to(ROOT)),sha256=hashlib.sha256(DENSITY.read_bytes()).hexdigest(),
            atom_checkpoint=str(GATE.relative_to(ROOT)),atom_sha256=hashlib.sha256(GATE.read_bytes()).hexdigest(),
            initialization='complete adopted v95 fixed atom/separate density function; no output reset',
            optimizer_reset=True,frozen_atom_eval_mode=True,private_density_backbone_and_head_trainable=True,
            unused_density_gate_parameters=129,training_key_drop_probability=.4,evaluation_key_drop_probability=0.,
            two_pass_sam_rho=.1,source_scope='both networks trained only original64 training wells')
    @property
    def body(self):return self.density.body
    @property
    def output(self):return self.density.output
    @property
    def condition(self):return self.density.condition
    def initialize_priors(self,data):return self.parent_info
    def fixed_hash(self):return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in self.atom.state_dict().values())).hexdigest()
    def train(self,mode=True):
        super().train(mode);self.atom.eval();return self
    def forward(self,x,relative,valid,absolute_depth=None,return_aux=False,generator=None,well_inputs=None):
        self.atom.eval()
        with torch.no_grad():gate,_=self.atom(x,relative,valid,well_inputs=well_inputs)
        _,raw=self.density(x,relative,valid,well_inputs=well_inputs)
        out=(gate,raw)
        return (out,raw.new_zeros(()),None) if return_aux else out

class DecisionObjective(ParentObjective):
    def __init__(self,evaluator,metadata,mode):
        super().__init__(evaluator,metadata,'a100');self.alpha=MODES[mode]['utility_weight']
