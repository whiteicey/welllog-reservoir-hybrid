"""Matched Q50 versus uniform likelihood for bounded full posterior readouts."""
import torch
from torch import nn
from torch.nn import functional as F
from crosscurve_v32 import Network as ParentBuilder, ROOT, PREVIOUS, reference_prediction
from champion_all_heads_v26 import Evaluator, load_data, TARGETS

STUDY=ROOT/'results/deep_learning/quality_readout_v37_20260929'
MODES=('q50','uniform')

class Correction(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261138)
            self.input=nn.Conv1d(319,32,1)
            self.output=nn.Conv1d(32,49,1)
            nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,hidden,parent,context,side,valid):
        x=torch.cat((hidden[0].float(),hidden[1].float(),side.transpose(1,2).float()),1)
        x=x.masked_fill(~valid[:,None],0)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z=self.output(F.gelu(self.input(x))).transpose(1,2).float()
        z=z.masked_fill(~valid[...,None],0)
        gate,raw=parent
        g=gate.float()+.25*torch.tanh(z[...,:1]/.25)
        w,mu,s=raw[:,:,0].float().unbind(-2)
        dw,dm,ds=z[...,1:].reshape(*z.shape[:2],3,16).unbind(-2)
        scale=F.softplus(s.detach())+.025
        # Original raw scale stays unchanged at zero; additive raw-scale bound .25.
        r=torch.stack((w+.25*torch.tanh(dw/.25),mu+.25*scale*torch.tanh(dm/.25),s+.25*torch.tanh(ds/.25)),-2)
        return g,r.unsqueeze(2)

class Network(ParentBuilder):
    def __init__(self,target,master,modes=MODES):
        super().__init__(target,master,modes=())
        self.heads=nn.ModuleDict({m:Correction() for m in modes})
        self.factor='bounded_all_posterior_factors'
        self.source_path=self.choice['checkpoint'];self.retained_path=None
        self.train(False)

    @torch.no_grad()
    def extract(self,x,relative,valid,absolute):
        assert x.shape[-1]==269
        h,o,ctx,_=self.parent.extract(x,relative,valid,absolute)
        old=self.parent.heads[self.parent_mode](h,o,ctx,x[...,206:],valid)
        return h,old,None,None

    def trainable_groups(self,mode):
        return [dict(params=list(self.heads[mode].parameters()),lr=3e-4,name='posterior_correction')]

@torch.no_grad()
def factor_scope(model,data,mode):
    return dict(parent_predictor_frozen=True,all_factors_trainable=True,
                gate_weight_rawscale_bound=.25,location_bound_in_parent_sd=.25,
                quality='fixed_source_Q50' if mode=='q50' else 'uniform_one')
