"""Private SW t4 conditional family; unchanged raw architecture and joint utility."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sw_utility_strength_v82 import ROOT,Network as Parent,load_data,SOURCE,Evaluator as OriginalEvaluator
from multiscale_sequence import window_indices
import train_invariant as base

STUDY=ROOT/'results/deep_learning/sw_student4_v89_20260930'
PREVIOUS=ROOT/'results/deep_learning/perm_grid_readout_v88_20260930'
CONTROL=ROOT/'outputs/pure_dl_sw_utility_strength_v82_SW_a100_fold5'
MODES={'student4':dict(target='SW',c=2,length=1024,rate=.4,utility_weight=1.)}

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

class Network(Parent):
    def __init__(self,mode,metadata):
        if mode!='student4':raise ValueError(mode)
        super().__init__('a100',metadata);self.study_mode=mode
        self.parent_info.update(density_family='Student-t4',density_std_parameterization=True,
            density_scale='std/sqrt2',historical_gaussian_control=str(CONTROL.relative_to(ROOT)),
            loaded_gaussian_control_weights=False,matched_raw_initialization_and_capacity=True,
            initial_predictive_function_equal_to_gaussian=False,
            only_hyperparameter_changed='conditional family Gaussian to t4; consistent NLL and CDF, same alpha1/DropKey/source/raw initialization/budget')

def student4_probability(weight,mean,std,edges):
    # t4 CDF = 1/2+3w/4-w^3/4, w=z/sqrt(z^2+4).
    # With t4 scale=std/sqrt2, w=(edge-mean)/sqrt((edge-mean)^2+2std^2).
    safe=torch.where(torch.isfinite(edges),edges,torch.zeros_like(edges))
    delta=safe[None,:,None]-mean[:,None,:]
    w=delta/(delta.square()+2*std[:,None,:].square()).sqrt()
    cdf=.5+.75*w-.25*w.pow(3)
    cdf=torch.where(torch.isneginf(edges)[None,:,None],torch.zeros_like(cdf),cdf)
    cdf=torch.where(torch.isposinf(edges)[None,:,None],torch.ones_like(cdf),cdf)
    mass=(cdf.diff(dim=1)*weight[:,None,:]).sum(-1).clamp_min(0)
    return mass/mass.sum(-1,keepdim=True).clamp_min(1e-12)

class Evaluator(OriginalEvaluator):
    @torch.no_grad()
    def __call__(self,model):
        model.eval();d=self.data;lo=model.length//2-64;hi=lo+128
        pred=np.full(len(d['validation']),np.nan,np.float64);coverage=np.zeros(len(pred),np.int8)
        for begin in range(0,len(self.centers),16):
            centers=torch.tensor(self.centers[begin:begin+16],device='cuda');anchors=torch.tensor(self.anchors[begin:begin+16],device='cuda')
            ix,valid=window_indices(centers,d['lower'],d['upper'],length=model.length,anchors=anchors)
            relative=torch.tanh((d['depth'][ix]-d['depth'][anchors,None])/10)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                go,raw=model(d['x'][ix],relative,valid,well_inputs=d['well_bank'][d['row_well'][anchors]])
            rows=d['raw_to_valid'][ix[:,lo:hi].cpu().numpy()];use=valid[:,lo:hi].cpu().numpy()&(rows>=0);dest=self.local[rows[use]]
            assert (dest>=0).all() and (coverage[dest]==0).all() and len(np.unique(dest))==len(dest)
            coverage[dest]+=1;take=torch.as_tensor(use,device='cuda')
            g=go[:,lo:hi][take][:,0].float().sigmoid();r=raw[:,lo:hi][take][:,0].float()
            p=student4_probability(r[:,0].softmax(-1),r[:,1],F.softplus(r[:,2])+.025,self.edges)
            mass=torch.cat((g[:,None],(1-g[:,None])*p),-1)
            pred[dest]=self.actions[(mass@self.utilities).argmax(-1).cpu().numpy()]
        assert (coverage==1).all() and np.isfinite(pred).all()
        combined=self.reference.copy();combined[:,2]=pred;scores=base.metric(d['truth'][d['validation']],combined)
        return pred,float(scores['SW']),scores

class DecisionObjective(nn.Module):
    def __init__(self,evaluator,metadata,mode):
        super().__init__();assert mode=='student4'
        self.register_buffer('actions',torch.as_tensor(evaluator.actions,device='cuda',dtype=torch.float32))
        self.register_buffer('utilities',evaluator.utilities.detach().clone())
        self.register_buffer('edges',evaluator.edges.detach().clone())
        self.median=metadata['sw_target_median'];self.target_scale=metadata['sw_target_scale']
        self.constant=math.lgamma(2.5)-math.lgamma(2.)-.5*math.log(4*math.pi)
    def forward(self,output,labels,default,mask,quality):
        gate,raw=output;g=gate[mask][:,0].float();r=raw[mask][:,0].float()
        d=default[mask][:,2];z=labels[mask][:,2].float();q=quality[mask].detach().float()
        sd=F.softplus(r[:,2])+.025;scale=sd/math.sqrt(2.);mu=r[:,1]
        y=z.masked_fill(d,0)
        logpdf=self.constant-scale.log()-2.5*torch.log1p(((y[:,None]-mu)/scale).square()/4.)
        nll=-(r[:,0].log_softmax(-1)+logpdf).logsumexp(-1).masked_fill(d,0)
        bce=F.binary_cross_entropy_with_logits(g,d.float(),reduction='none')
        likelihood=(q*(bce+nll)).mean()/3
        p=student4_probability(r[:,0].softmax(-1),mu,sd,self.edges)
        pg=g.sigmoid();mass=torch.cat((pg[:,None],(1-pg[:,None])*p),-1)
        ap=((mass@self.utilities)/.05).softmax(-1)
        physical_y=torch.where(d,torch.full_like(z,99.9),torch.exp(z*self.target_scale+self.median))
        true_utility=(1-(self.actions[None]-physical_y[:,None]).abs()/(.05*(physical_y.abs()[:,None]+1e-5))).clamp_min(0)
        decision=(q*(1-(ap*true_utility).sum(-1))).mean()/3
        return likelihood+decision,likelihood,decision
