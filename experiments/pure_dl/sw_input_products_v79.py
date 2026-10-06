"""Frozen SW predictor with matched marginal or cross-curve input basis adapters."""
import hashlib
import numpy as np
import torch
from torch import nn
from sw_dropout_extend_v77 import ROOT,Network as Parent,Evaluator,load_data
from divergent_models_v45 import CURVES

STUDY=ROOT/'results/deep_learning/sw_input_products_v79_20260930'
PREVIOUS=ROOT/'results/deep_learning/sw_dropout_heads_v78_20260930'
SOURCE=ROOT/'outputs/pure_dl_sw_dropout_extend_v77_SW_p40_fold5/model_best.pt'
MODES={m:dict(target='SW',c=2,length=1024,rate=0.) for m in ('marginal','cross')}

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

class AdaptedStem(nn.Module):
    def __init__(self,base,indices,mode):
        super().__init__();self.base=base;self.mode=mode
        self.register_buffer('indices',torch.tensor(indices))
        self.register_buffer('pairs',torch.triu_indices(9,9))
        self.adapter=nn.Linear(45,96,bias=False);nn.init.zeros_(self.adapter.weight)
    def forward(self,packed):
        z=torch.tanh(packed.index_select(-1,self.indices).float()/3.)
        if self.mode=='marginal':features=torch.cat([z.pow(k) for k in range(2,7)],-1)
        else:features=z[...,self.pairs[0]]*z[...,self.pairs[1]]
        original=self.base(packed)
        delta=.25*torch.tanh(self.adapter(features).float()/.25)
        return original+delta.to(original.dtype)

class Network(Parent):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('p40',metadata)
        saved=torch.load(SOURCE,map_location='cpu',weights_only=False);self.load_state_dict(saved['state_dict'],strict=True)
        self.requires_grad_(False);self.study_mode=mode
        indices=[metadata['selected_feature_names'].index(curve) for curve in CURVES]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261179);self.body.stem=AdaptedStem(self.body.stem,indices,mode)
        self.parent_info=dict(checkpoint=str(SOURCE.relative_to(ROOT)),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            best_epoch=saved['best_epoch'],target_score=saved['target_score'],initialization='complete v77 p40 SW function plus zero input adapter',
            optimizer_reset=True,all_network_trainable=False,added_parameters=4320,original_weights_frozen=True,
            input_adapter_view=mode,adapter_dimensions=45,adapter_bound=.25,bound_unit='each pre-GELU stem activation',
            transformed_curves=list(CURVES),input_feature_indices=indices,curve_transform='tanh(original fold-scaled input/3)',
            training_key_drop_probability=0.,evaluation_key_drop_probability=0.,frozen_predictor_eval_mode=True)
        self.train(False)
    def initialize_priors(self,data):return self.parent_info
    def train(self,mode=True):
        super().train(False);self.training=mode;self.body.stem.adapter.train(mode);return self
