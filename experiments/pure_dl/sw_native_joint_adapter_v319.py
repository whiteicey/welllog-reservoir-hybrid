"""Native Gaussian SW backbone/readout joint adaptation, original104 only."""
import hashlib
import torch
from torch import nn
from compact_frequency_v104 import ROOT,SOURCE,Network as Positive,load_data
from train_stage_fullcv_v97 import Evaluator as ParentEvaluator
from sw_utility_strength_v82 import DecisionObjective as ParentObjective
from sw_global_ff_adapter_v225 import ResidualFF
STUDY=ROOT/'results/deep_learning/sw_native_joint_adapter_v319_20261002'
OUT=ROOT/'outputs/pure_dl_sw_native_joint_adapter_v319_fold5'
PARENT=ROOT/'outputs/pure_dl_compact_frequency_v104_fold5/local/model_best.pt'
PARENT_SHA='07e0b8df208fe93c8ac07dcf18924459c47afb12ec84b1b7306a0e6017f58670'
STAGES={m:('SW',1024,True,1.,3e-5) for m in ('control','adapter')}
class Network(nn.Module):
    def __init__(self,master,metadata,mode):
        super().__init__();self.mode=mode
        assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
        self.base=Positive(metadata,'local')
        self.base.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=False)['state_dict'],strict=True)
        if mode=='adapter':self.base.body.global_blocks[-1].ff=ResidualFF(self.base.body.global_blocks[-1].ff)
        self.base.requires_grad_(True)
        assert sum(p.numel() for p in self.parameters())==(852185 if mode=='adapter' else 845913)
    def groups(self):
        patch=list(self.base.body.global_blocks[-1].ff.patch.parameters()) if self.mode=='adapter' else []
        readout=list(self.base.output.parameters())
        patch_ids={id(p) for p in patch};readout_ids={id(p) for p in readout}
        body=[p for p in self.parameters() if id(p) not in patch_ids|readout_ids]
        groups=[dict(params=body,lr=3e-5,name='SW_original_body_condition_norm'),dict(params=readout,lr=3e-5,name='SW_original_state_density_readout')]
        if patch:groups.append(dict(params=patch,lr=3e-4,name='SW_new_FF_adapter'))
        return groups
    def initialize_priors(self,data):
        return dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,all_original_weights_loaded=True,original_readout_not_reset=True,full_body_state_density_joint=True,new_adapter_parameters=6272 if self.mode=='adapter' else 0,trained_repair_weights_loaded=False,dropkey=.4,SAM_rho=.1)
    def forward(self,*args,**kwargs):return self.base(*args,**kwargs)
class Evaluator(ParentEvaluator):
    def __init__(self,data,master,c,length,mode):super().__init__(data,master,c,length)
class DecisionObjective(ParentObjective):
    def __init__(self,evaluator,metadata,mode):super().__init__(evaluator,metadata,'a100')
def make_model(stage,master,metadata):return Network(master,metadata,stage)
def channel_objective(*args):raise RuntimeError('SW only')
