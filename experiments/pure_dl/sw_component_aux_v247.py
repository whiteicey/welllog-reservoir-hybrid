"""Frozen104 Gaussian prior plus a bounded neural correction on original256 bin masses."""
import hashlib,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from compact_frequency_v104 import ROOT,SOURCE,Network as Positive,load_data
from train_stage_fullcv_v97 import Evaluator as ParentEvaluator
from por_t3_patch_loss import probability
from multiscale_sequence import window_indices
import train_invariant as base
STUDY=ROOT/'results/deep_learning/sw_component_aux_v247_20261001'
OUT=ROOT/'outputs/pure_dl_sw_component_aux_v247_fold5'
PARENT=ROOT/'outputs/pure_dl_compact_frequency_v104_fold5/local/model_best.pt'
PARENT_SHA='07e0b8df208fe93c8ac07dcf18924459c47afb12ec84b1b7306a0e6017f58670'
STAGES={'aux':('SW',1024,True,0.,1e-4)}

class Network(nn.Module):
    def __init__(self,master,metadata,mode):
        super().__init__();self.mode=mode;self.base=Positive(metadata,'local')
        assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
        self.base.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=False)['state_dict'],strict=True)
        self.base.requires_grad_(False).eval()
        self.register_buffer('edges',torch.as_tensor(master['density_edges'][2],dtype=torch.float32))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010)
            self.patch=nn.Sequential(nn.Linear(128,32),nn.GELU(),nn.Linear(32,257))
            nn.init.zeros_(self.patch[-1].weight);nn.init.zeros_(self.patch[-1].bias)
            with torch.no_grad():self.patch[-1].bias[-1]=math.log(.001/.999)
        self.feature=None;self.handle=self.base.output.register_forward_pre_hook(self.capture_feature)
    def capture_feature(self,module,args):self.feature=args[0].detach()
    def train(self,mode=True):super().train(mode);self.base.eval();return self
    def groups(self):return [dict(params=list(self.patch.parameters()),lr=1e-4,name='SW_probability_mass_patch_only')]
    def initialize_priors(self,data):
        return dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,all104_weights_frozen=True,
                    mode=self.mode,extra_active_parameters=12609,initial_mix=.001,
                    initial_new_component_uniform=True,initial_function_changes_from104=True,
                    initial_score_separate_from_learned_gain=True,paired_initial_distributions_equal=True,
                    learned_alpha_sigmoid_uncapped=True,original_default_gate_frozen=True,
                    additive_new_component_softmax=True if self.mode in ('additive','aux') else False,
                    multiplicative_logmass_bound=.5 if self.mode=='multiplicative' else None,
                    same_halfhard_halftolerance_CE_SAM=True,own_new_component_auxiliary_CE_weight=.25,added_SW_stages=1,
                    executed_training_stages_including_source=6,no_trained_repair_weights=True)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.feature=None
        with torch.no_grad():
            gate,raw=self.base(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
            flat=raw[:,:,0].float().reshape(-1,3,8);parts=[]
            for chunk in flat.split(4096):
                parts.append(probability(2,chunk[:,0].softmax(-1),chunk[:,1],F.softplus(chunk[:,2])+.025,self.edges))
            prior=torch.cat(parts).reshape(*x.shape[:2],256)
        raw_patch=self.patch(self.feature).float();self.feature=None
        alpha=raw_patch[:,:,-1:].sigmoid();logits=raw_patch[:,:,:256]
        if self.mode in ('additive','aux'):
            mass=(1-alpha)*prior+alpha*logits.softmax(-1)
            logmass=mass.clamp_min(1e-12).log().log_softmax(-1)
        else:
            background=(1-alpha)*prior+alpha/256
            correction=.5*torch.tanh(logits/.5)
            logmass=(background.clamp_min(1e-12).log()+correction).log_softmax(-1)
        return gate,logmass.unsqueeze(2),logits.log_softmax(-1)

class Evaluator(ParentEvaluator):
    def __init__(self,data,master,c,length,mode):super().__init__(data,master,c,length)
    @torch.no_grad()
    def __call__(self,model):
        d=self.data;model.eval();lo=self.length//2-64;hi=lo+128
        result=np.full(len(d['validation']),np.nan,np.float64);coverage=np.zeros(len(result),np.int8)
        for begin in range(0,len(self.centers),self.batch):
            cc=torch.tensor(self.centers[begin:begin+self.batch],device='cuda')
            anchor=torch.tensor(self.anchors[begin:begin+self.batch],device='cuda')
            ix,valid=window_indices(cc,d['lower'],d['upper'],length=self.length,anchors=anchor)
            rel=torch.tanh((d['depth'][ix]-d['depth'][anchor,None])/10)
            context=d['well_bank'][d['row_well'][anchor]]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                gate,logmass,_=model(d['x'][ix],rel,valid,absolute_depth=d['absolute'][ix],well_inputs=context)
            rows=d['raw_to_valid'][ix[:,lo:hi].cpu().numpy()]
            use=valid[:,lo:hi].cpu().numpy()&(rows>=0);dest=self.local[rows[use]]
            assert (dest>=0).all() and (coverage[dest]==0).all() and len(np.unique(dest))==len(dest)
            coverage[dest]+=1;take=torch.as_tensor(use,device='cuda')
            g=gate[:,lo:hi][take][:,0].float().sigmoid()
            mass=logmass[:,lo:hi][take][:,0].float().exp()
            joint=torch.cat((g[:,None],(1-g[:,None])*mass),-1)
            result[dest]=self.actions[(joint@self.utilities).argmax(-1).cpu().numpy()]
        assert (coverage==1).all() and np.isfinite(result).all()
        y=d['truth'][d['validation']].copy();pred=y.copy();pred[:,2]=result
        return result,float(base.metric(y,pred)['SW'])

class DecisionObjective(nn.Module):
    def __init__(self,evaluator,metadata,mode):
        super().__init__();self.alpha=0.;self.temperature=.05
        self.register_buffer('actions',torch.as_tensor(evaluator.actions,dtype=torch.float32))
        self.register_buffer('utilities',evaluator.utilities.detach().clone())
        self.register_buffer('edges',evaluator.edges.detach().clone())
        self.median=metadata['sw_target_median'];self.scale=metadata['sw_target_scale']
        self.register_buffer('physical_centers',torch.as_tensor(metadata['centers'][2],dtype=torch.float32))
    def forward(self,output,labels,default,mask,quality):
        gate,logmass,newlogmass=output;g=gate[mask][:,0].float();logp=logmass[mask][:,0].float()
        d=default[mask][:,2];z=labels[mask][:,2].float();q=quality[mask].detach().float()
        bins=torch.bucketize(z.contiguous(),self.edges[1:-1])
        hard=-logp.gather(1,bins[:,None]).squeeze(1)
        truth=torch.exp(z*self.scale+self.median)
        soft=(1-(self.physical_centers[None]-truth[:,None]).abs()/(.05*(truth.abs()[:,None]+1e-5))).clamp_min(0)
        total=soft.sum(-1,keepdim=True)
        fallback=F.one_hot(bins,num_classes=logp.shape[-1]).float()
        soft=torch.where(total>0,soft/total.clamp_min(1e-12),fallback).detach()
        nll=(.5*hard-.5*(soft*logp).sum(-1)).masked_fill(d,0)
        bce=F.binary_cross_entropy_with_logits(g,d.float(),reduction='none')
        likelihood=(q*(bce+nll)).mean()/3
        prob=g.sigmoid()[:,None];joint=torch.cat((prob,(1-prob)*logp.exp()),-1)
        actionprob=((joint@self.utilities)/self.temperature).softmax(-1)
        physical=torch.where(d,torch.full_like(z,99.9),torch.exp(z*self.scale+self.median))
        utility=(1-(self.actions[None]-physical[:,None]).abs()/(.05*(physical.abs()[:,None]+1e-5))).clamp_min(0)
        decision=(q*(1-(actionprob*utility).sum(-1))).mean()/3
        newlogp=newlogmass[mask].float()
        newhard=-newlogp.gather(1,bins[:,None]).squeeze(1)
        newnll=(.5*newhard-.5*(soft*newlogp).sum(-1)).masked_fill(d,0)
        auxiliary=(q*newnll).mean()/3
        return likelihood+.25*auxiliary,likelihood,auxiliary

def make_model(stage,master,metadata):return Network(master,metadata,stage)
def channel_objective(*args):raise RuntimeError('SW own bin-mass objective only')


