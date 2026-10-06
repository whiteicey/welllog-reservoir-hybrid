"""Frozen retained Gaussian SW with matched reference/own direct code readouts."""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sw_direct_well_v90 import ROOT,CONTROL,DecisionObjective,load_data
from sw_utility_strength_v82 import Network as GaussianNetwork,Evaluator as GaussianEvaluator
from multiscale_sequence import window_indices
from por_t3_patch_loss import probability
import train_invariant as base

STUDY=ROOT/'results/deep_learning/sw_frozen_direct_v91_20260930'
PREVIOUS=ROOT/'results/deep_learning/sw_direct_well_v90_20260930'
SOURCE=CONTROL/'model_best.pt'
TARGETS=('POR','PERM','SW');MODES=('reference','well');LEARNING_RATE=1e-4;LENGTH=1024

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

class Network(nn.Module):
    def __init__(self,target,metadata):
        super().__init__();assert target=='SW'
        self.base=GaussianNetwork('a100',metadata)
        saved=torch.load(SOURCE,map_location='cpu',weights_only=False);self.base.load_state_dict(saved['state_dict'],strict=True)
        self.base.requires_grad_(False);self.base.eval();self.length=LENGTH;self._code=None
        self._hook=self.base.condition.code.register_forward_hook(self._capture)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261191);self.heads=nn.ModuleDict({m:nn.Linear(96,25) for m in MODES})
        for head in self.heads.values():nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        self.metadata=dict(checkpoint=str(SOURCE.relative_to(ROOT)),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),best_epoch=saved['best_epoch'],target_score=saved['target_score'],
            mechanism='same zero Linear96→25; only new input reference/own; shared frozen original body and FiLM',added_parameters=2425,
            original_model_frozen=True,existing_film_always_own=True,both_codes_computed=True,frozen_model_inference_mode_no_dropkey=True,
            initialization='actual adopted v82 alpha1 best25; not source v56 prior reset',bank_manifest=metadata['bank_manifest'])
    def _capture(self,module,args,output):self._code=output
    def close(self):self._hook.remove();self._code=None
    def train(self,mode=True):
        super().train(mode);self.base.eval();return self
    def frozen_state(self):return self.base.state_dict()
    def trainable_groups(self,mode):return [dict(params=list(self.heads[mode].parameters()),name='direct_output')]
    @torch.no_grad()
    def extract(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self._code=None
        with torch.autocast('cuda',dtype=torch.bfloat16):
            old=self.base(x,relative,valid,well_inputs=well_inputs)
            own=self._code.detach()
            z=self.base.condition.point(self.base.reference_inputs[None])
            stats=torch.cat((z.mean(1),z.var(1,unbiased=False).add(1e-6).sqrt(),z.amax(1)),1)
            ref=self.base.condition.code(stats).detach()
        self._code=None
        return {'reference':ref.expand(x.shape[0],-1),'well':own},old
    def apply(self,mode,codes,old):
        with torch.autocast('cuda',dtype=torch.bfloat16):delta=self.heads[mode](codes[mode]).float()
        gate,raw=old;b=gate.shape[0]
        return gate+delta[:,None,:1],raw+delta[:,None,1:].reshape(b,1,1,3,8)

class Evaluator(GaussianEvaluator):
    @torch.no_grad()
    def many(self,model,modes):
        model.eval();d=self.data;lo=LENGTH//2-64;hi=lo+128
        pred={m:np.full(len(d['validation']),np.nan,np.float64) for m in modes};coverage=np.zeros(len(d['validation']),np.int8)
        for begin in range(0,len(self.centers),16):
            centers=torch.tensor(self.centers[begin:begin+16],device='cuda');anchors=torch.tensor(self.anchors[begin:begin+16],device='cuda')
            ix,valid=window_indices(centers,d['lower'],d['upper'],length=LENGTH,anchors=anchors)
            rel=torch.tanh((d['depth'][ix]-d['depth'][anchors,None])/10)
            codes,old=model.extract(d['x'][ix],rel,valid,well_inputs=d['well_bank'][d['row_well'][anchors]])
            rows=d['raw_to_valid'][ix[:,lo:hi].cpu().numpy()];use=valid[:,lo:hi].cpu().numpy()&(rows>=0);dest=self.local[rows[use]]
            assert (dest>=0).all() and (coverage[dest]==0).all() and len(np.unique(dest))==len(dest)
            coverage[dest]+=1;take=torch.as_tensor(use,device='cuda')
            for m in modes:
                go,raw=model.apply(m,codes,old);g=go[:,lo:hi][take][:,0].float().sigmoid();r=raw[:,lo:hi][take][:,0].float()
                p=probability(2,r[:,0].softmax(-1),r[:,1],F.softplus(r[:,2])+.025,self.edges)
                mass=torch.cat((g[:,None],(1-g[:,None])*p),-1)
                pred[m][dest]=self.actions[(mass@self.utilities).argmax(-1).cpu().numpy()]
        assert (coverage==1).all();values={}
        for m in modes:
            assert np.isfinite(pred[m]).all();combined=self.reference.copy();combined[:,2]=pred[m];scores=base.metric(d['truth'][d['validation']],combined)
            values[m]=(pred[m],float(scores['SW']),float(scores['TOTAL']))
        return values
