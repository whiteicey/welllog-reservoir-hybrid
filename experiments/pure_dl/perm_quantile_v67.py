"""PERM conditional quantile density: fixed matched linear versus smooth CDF."""
import hashlib
import numpy as np
import torch
from torch import nn
from perm_transfer_v50 import ROOT,Network as Parent,load_data
from quantile_density_v48 import Network as QuantileForward,Evaluator as QuantileEvaluator,channel_objective,K,TAIL
STUDY=ROOT/'results/deep_learning/perm_quantile_v67_20260929'
PREVIOUS=ROOT/'results/deep_learning/sw_well_order_v66_20260929'
SOURCE=ROOT/'outputs/pure_dl_perm_transfer_v50_PERM_sw_transfer_fold5/model_best.pt'
MODES={m:dict(target='PERM',c=1,length=1024) for m in ('linear','rqs')}
def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')
class Network(Parent):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('sw_transfer',metadata)
        saved=torch.load(SOURCE,map_location='cpu',weights_only=False);self.load_state_dict(saved['state_dict'],strict=True)
        self.mode=mode;self.study_mode=mode;self.c=1
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261167);self.gate=nn.Linear(128,1)
            with torch.no_grad():self.gate.weight.copy_(self.output.weight[:1]);self.gate.bias.copy_(self.output.bias[:1])
            self.output=nn.Linear(128,K+2);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.parent_info=dict(checkpoint=str(SOURCE.relative_to(ROOT)),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            best_epoch=saved['best_epoch'],target_score=saved['target_score'],optimizer_reset=True,
            initialization='own-fold PERM body/norm/FiLM/default gate; reset density to train64 empirical quantile prior',
            matching='same parameters and knots; interpolation changes initial CDF, not identical full function')
    @torch.no_grad()
    def initialize_priors(self,data):
        ids=data['train_raw'];selected=ids[~data['default'][ids,1]]
        prior=torch.quantile(data['labels'][selected,1].float(),torch.linspace(TAIL,1-TAIL,K+1,device='cuda'))
        gap=prior.diff().clamp_min(.0011);span=(gap-.001).sum().clamp_min(.101)
        self.output.bias[0].copy_((prior[0]+prior[-1])/2)
        self.output.bias[1].copy_(torch.log(torch.expm1(span-.1)))
        self.output.bias[2:].copy_((gap-.001).log())
        return self.parent_info|dict(prior_quantiles=prior.cpu().tolist(),knots=K+1,tail_mass=TAIL,
            prior_training_nondefault_rows=len(selected),prior_fit='original64 training PERM labels only',minimum_interval=.001)
    forward=QuantileForward.forward
class Evaluator(QuantileEvaluator):
    @torch.no_grad()
    def __call__(self,model):
        pred,_,scores=super().__call__(model)
        return pred,float(scores['PERM']),scores
