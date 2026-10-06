"""Direct regression of official action utility, anchored on a frozen density model.

Values are expected rewards, not normalized probability masses. The loss uses
training labels only. Point/local candidates differ only in fixed pooling.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from quality_readout_v37 import Network as FrozenBuilder,ROOT,PREVIOUS,load_data,reference_prediction,TARGETS
from train_scaling_v21 import Evaluator as ParentEvaluator
from por_t3_patch_loss import probability
from multiscale_sequence import window_indices
import train_invariant as base

STUDY=ROOT/'results/deep_learning/action_value_v38_20260929'
MODES=('point','local5')
BOUND=.25

class ActionHead(nn.Module):
    def __init__(self,mode,n_actions):
        super().__init__();self.mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261139)
            self.input=nn.Conv1d(319,32,1)
            self.output=nn.Conv1d(32,n_actions,1)
            nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def delta(self,hidden,side,valid):
        x=torch.cat((hidden[0].float(),hidden[1].float(),side.transpose(1,2).float()),1)
        x=x.masked_fill(~valid[:,None],0)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            h=F.gelu(self.input(x)).masked_fill(~valid[:,None],0)
            if self.mode=='local5':
                count=F.avg_pool1d(valid[:,None].float(),5,stride=1,padding=2).clamp_min(.2)
                h=F.avg_pool1d(h,5,stride=1,padding=2)/count
            z=self.output(h).transpose(1,2).float()
        return (BOUND*torch.tanh(z/BOUND)).masked_fill(~valid[...,None],0)

    def forward(self,hidden,parent_values,context,side,valid):
        return parent_values+self.delta(hidden,side,valid)

class Network(FrozenBuilder):
    def __init__(self,target,master,modes=MODES):
        super().__init__(target,master,modes=())
        c=TARGETS.index(target);ct=np.asarray(master['centers'][c])
        action=np.linspace(ct[0],ct[-1],800) if c==2 else np.geomspace(max(ct[0]*.9,.001),ct[-1]*1.1,800)
        actions=np.unique(np.r_[base.DEFAULTS[target],np.clip(action,*base.BOUNDS[target])])
        self.actions_numpy=actions
        self.register_buffer('actions',torch.as_tensor(actions,dtype=torch.float32))
        self.register_buffer('utilities',torch.as_tensor(base.kernel_matrix(actions,np.r_[base.DEFAULTS[target],ct],target).T,dtype=torch.float32))
        self.register_buffer('edges',torch.as_tensor(master['density_edges'][c],dtype=torch.float32))
        self.median=float(master['regression_median'][c]);self.scale=float(master['regression_scale'][c])
        self.heads=nn.ModuleDict({m:ActionHead(m,len(actions)) for m in modes})
        self.factor='direct_action_utility';self.train(False)

    @torch.no_grad()
    def expected(self,parent):
        go,raw=parent;shape=go.shape[:-1]
        g=go.reshape(-1).float().sigmoid();r=raw[...,0,:,:].reshape(-1,3,raw.shape[-1]).float()
        p=probability(self.c,r[:,0].softmax(-1),r[:,1],F.softplus(r[:,2])+.025,self.edges)
        mass=torch.cat((g[:,None],(1-g[:,None])*p),-1)
        return (mass@self.utilities).reshape(*shape,len(self.actions))

    def forward(self,x,relative,valid,absolute_depth=None,factors=None):
        h,parent,ctx,_=self.extract(x,relative,valid,absolute_depth)
        values=self.expected(parent)
        modes=tuple(self.heads) if factors is None else tuple(factors)
        return {m:self.heads[m](h,values,ctx,x[...,206:],valid) for m in modes}

class Objective:
    def __init__(self,model):self.model=model

    @torch.no_grad()
    def labels(self,normalized,default,mask):
        m=self.model;c=m.c;d=default[mask][:,c]
        y=torch.exp(normalized[mask][:,c].float()*m.scale+m.median)
        y=torch.where(d,torch.full_like(y,base.DEFAULTS[m.target]),y)
        if c==1:
            return (1-(m.actions.clamp_min(1e-5).log10()[None]-y.clamp_min(1e-5).log10()[:,None]).abs()).clamp_min(0)
        tolerance=.08 if c==0 else .05
        return (1-(m.actions[None]-y[:,None]).abs()/(tolerance*(y.abs()[:,None]+1e-5))).clamp_min(0)

    def __call__(self,values,labels,mask):
        # Uniform weighting intentionally retained from the v37 matched contrast.
        loss=(values[mask].float()-labels).square().mean()/3
        return loss,loss.detach(),loss.detach()

class Evaluator(ParentEvaluator):
    @torch.no_grad()
    def many(self,model,factors):
        model.eval();d=self.data;c=self.c;factors=tuple(factors)
        assert np.array_equal(self.actions,model.actions_numpy)
        predictions={m:np.full(len(d['validation']),np.nan,np.float64) for m in factors}
        coverage=np.zeros(len(d['validation']),np.int8)
        for begin in range(0,len(self.centers),32):
            centers=torch.tensor(self.centers[begin:begin+32],device='cuda')
            anchors=torch.tensor(self.anchors[begin:begin+32],device='cuda')
            ix,valid=window_indices(centers,d['lower'],d['upper'],anchors=anchors)
            relative=torch.tanh((d['depth'][ix]-d['depth'][anchors,None])/10)
            h,parent,ctx,_=model.extract(d['x'][ix],relative,valid,d['absolute'][ix])
            rows=d['raw_to_valid'][ix[:,64:192].cpu().numpy()]
            use=valid[:,64:192].cpu().numpy()&(rows>=0);loc=self.local[rows[use]]
            assert (loc>=0).all() and (coverage[loc]==0).all() and len(np.unique(loc))==len(loc)
            coverage[loc]+=1;take=torch.as_tensor(use,device='cuda')
            # Preserve the original decoder's exact row population and arithmetic.
            cropped=tuple(x[:,64:192][take] for x in parent)
            expected=model.expected(cropped)
            for m in factors:
                delta=model.heads[m].delta(h,d['x'][ix,...,206:],valid)[:,64:192][take]
                predictions[m][loc]=self.actions[(expected+delta).argmax(-1).cpu().numpy()]
        assert (coverage==1).all()
        result={}
        for m,p in predictions.items():
            assert np.isfinite(p).all();combined=self.reference.copy();combined[:,c]=p
            scores=base.metric(d['truth'][d['validation']],combined)
            result[m]=(p,float(scores[TARGETS[c]]),float(scores['TOTAL']))
        return result

@torch.no_grad()
def factor_scope(model,data,mode):
    return dict(parent_frozen=True,head_type=mode,action_count=len(model.actions),action_utility_delta_bound=BOUND,
                original_action_grid=True,values_are_not_probabilities=True,training_weight='uniform')
