"""Frozen154 Gaussian16 prior plus bounded PERM conditional mass readout."""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from perm_ranked_probability_v154 import ROOT,SOURCE,Network as Positive,load_data
from train_stage_fullcv_v97 import Evaluator as ParentEvaluator
from por_t3_patch_loss import probability
from multiscale_sequence import window_indices
import train_invariant as base
STUDY=ROOT/'results/deep_learning/perm_tolerance_mass_v234_20261001'
OUT=ROOT/'outputs/pure_dl_perm_tolerance_mass_v234_fold5'
PARENT=ROOT/'outputs/pure_dl_perm_ranked_probability_v154_fold5/perm_conditional/model_best.pt'
PARENT_SHA='2b355bb8c3736990b05e2726373a4343c574a93ab6ab1a3eda0ef796ac70cf11'
STAGES={'mass':('PERM',256,False,0.,1e-4)}

class Network(nn.Module):
    def __init__(self,master,metadata):
        super().__init__();self.base=Positive(master,'perm_conditional')
        assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
        self.base.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=False)['state_dict'],strict=True)
        self.base.requires_grad_(False).eval()
        self.register_buffer('edges',torch.as_tensor(master['density_edges'][1],dtype=torch.float32))
        self.register_buffer('physical_logcenters',torch.log10(torch.as_tensor(master['centers'][1],dtype=torch.float32).clamp_min(1e-5)))
        self.register_buffer('target_scale',torch.tensor(float(master['regression_scale'][1])))
        self.register_buffer('target_median',torch.tensor(float(master['regression_median'][1])))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010)
            self.patch=nn.Sequential(nn.Linear(128,32),nn.GELU(),nn.Linear(32,256))
            nn.init.zeros_(self.patch[-1].weight);nn.init.zeros_(self.patch[-1].bias)
        self.feature=None;self.handle=self.base.context_head.expand.register_forward_pre_hook(self.capture_feature)
    def capture_feature(self,module,args):self.feature=args[0].detach().transpose(1,2)
    def train(self,mode=True):super().train(mode);self.base.eval();return self
    def groups(self):return [dict(params=list(self.patch.parameters()),lr=1e-4,name='PERM_probability_mass_patch_only')]
    def initialize_priors(self,data):
        return dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,all154_positive_weights_frozen=True,
                    patch_input='frozen154 context_head expand input,128 hidden',zero_output=True,
                    extra_active_parameters=sum(p.numel() for p in self.patch.parameters()),
                    logmass_bound=.5,hard_target_weight=.5,tolerance_target_weight=.5,prior_floor=1e-12,own_bin_NLL_BCE_plus_joint_RPS1=True,
                    original_Gaussian16_prior_retained=True,old_gate_frozen=True,
                    initial_function_identity_except_floor_float_renormalization=True,
                    executed_training_stages_including_source=6,added_PERM_stages=1,no_other_repair_weights_loaded=True)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.feature=None
        with torch.no_grad():
            gate,raw=self.base(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
            flat=raw[:,:,0].float().reshape(-1,3,16);parts=[]
            for chunk in flat.split(4096):
                parts.append(probability(1,chunk[:,0].softmax(-1),chunk[:,1],F.softplus(chunk[:,2])+.025,self.edges))
            prior=torch.cat(parts).reshape(*x.shape[:2],256)
        correction=.5*torch.tanh(self.patch(self.feature).float()/.5)
        self.feature=None
        logmass=(prior.clamp_min(1e-12).log()+correction).log_softmax(-1)
        return gate,logmass.unsqueeze(2)

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
                gate,logmass=model(d['x'][ix],rel,valid,absolute_depth=d['absolute'][ix],well_inputs=context)
            rows=d['raw_to_valid'][ix[:,lo:hi].cpu().numpy()]
            use=valid[:,lo:hi].cpu().numpy()&(rows>=0);dest=self.local[rows[use]]
            assert (dest>=0).all() and (coverage[dest]==0).all() and len(np.unique(dest))==len(dest)
            coverage[dest]+=1;take=torch.as_tensor(use,device='cuda')
            g=gate[:,lo:hi][take][:,0].float().sigmoid()
            mass=logmass[:,lo:hi][take][:,0].float().exp()
            joint=torch.cat((g[:,None],(1-g[:,None])*mass),-1)
            result[dest]=self.actions[(joint@self.utilities).argmax(-1).cpu().numpy()]
        assert (coverage==1).all() and np.isfinite(result).all()
        y=d['truth'][d['validation']].copy();pred=y.copy();pred[:,1]=result
        return result,float(base.metric(y,pred)['PERM'])

def make_model(stage,master,metadata):return Network(master,metadata)

def channel_objective(output,labels,default,mask,quality,c,mode,model):
    assert c==1
    gate,logmass=output;g=gate[mask][:,0].float();logp=logmass[mask][:,0].float()
    d=default[mask][:,1];z=labels[mask][:,1].float();q=quality[mask].detach().float()
    edges=model.edges[1:-1];bins=torch.bucketize(z.contiguous(),edges)
    hard=-logp.gather(1,bins[:,None]).squeeze(1)
    logtrue=z*model.target_scale+model.target_median
    soft=(1-(model.physical_logcenters[None]-logtrue[:,None]).abs()).clamp_min(0)
    total=soft.sum(-1,keepdim=True);fallback=F.one_hot(bins,num_classes=logp.shape[-1]).float()
    soft=torch.where(total>0,soft/total.clamp_min(1e-12),fallback).detach()
    nll=(.5*hard-.5*(soft*logp).sum(-1)).masked_fill(d,0)
    bce=F.binary_cross_entropy_with_logits(g,d.float(),reduction='none')
    likelihood=(q*(bce+nll)).mean()/3
    prob=g.sigmoid()[:,None];atom=(model.base.default_z<=edges).float()[None]
    joint=prob*atom+(1-prob)*logp.exp().cumsum(-1)[:,:-1]
    target=torch.where(d,model.base.default_z,z)
    empirical=(target[:,None]<=edges[None]).float()
    rps=(q*(joint-empirical).square().mean(-1)).mean()/3
    return likelihood+rps,(q*bce).mean(),(q*nll).mean()+3*rps
