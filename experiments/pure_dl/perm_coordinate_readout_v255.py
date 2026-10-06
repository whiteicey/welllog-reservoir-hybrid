"""Frozen154 Gaussian16 prior plus bounded PERM conditional mass readout."""
import hashlib,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from perm_ranked_probability_v154 import ROOT,SOURCE,Network as Positive,load_data
from train_stage_fullcv_v97 import Evaluator as ParentEvaluator
from por_t3_patch_loss import probability
from multiscale_sequence import window_indices
import train_invariant as base
STUDY=ROOT/'results/deep_learning/perm_coordinate_readout_v255_20261001'
OUT=ROOT/'outputs/pure_dl_perm_coordinate_readout_v255_fold5'
PARENT=ROOT/'outputs/pure_dl_perm_ranked_probability_v154_fold5/perm_conditional/model_best.pt'
PARENT_SHA='2b355bb8c3736990b05e2726373a4343c574a93ab6ab1a3eda0ef796ac70cf11'
STAGES={mode:('PERM',256,False,0.,1e-4) for mode in ('bilinear','nonlinear')}

class Network(nn.Module):
    def __init__(self,master,metadata,mode):
        super().__init__();self.mode=mode;self.base=Positive(master,'perm_conditional')
        assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
        self.base.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=False)['state_dict'],strict=True)
        self.base.requires_grad_(False).eval()
        self.register_buffer('edges',torch.as_tensor(master['density_edges'][1],dtype=torch.float32))
        coordinate=torch.log10(torch.as_tensor(master['centers'][1],dtype=torch.float32).clamp_min(1e-5))
        coordinate=2*(coordinate-coordinate[0])/(coordinate[-1]-coordinate[0]).clamp_min(1e-8)-1
        self.register_buffer('coordinate',coordinate[:,None])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010)
            self.query=nn.Sequential(nn.Linear(128,32),nn.GELU())
            self.value=nn.Sequential(nn.Linear(1,32),nn.GELU(),nn.Linear(32,32))
            self.projection=nn.Linear(32,1,bias=False)
            self.background=nn.Linear(32,1)
            nn.init.zeros_(self.projection.weight);nn.init.zeros_(self.background.weight)
            nn.init.constant_(self.background.bias,math.log(.001/.999))
        self.feature=None;self.handle=self.base.context_head.expand.register_forward_pre_hook(self.capture_feature)
    def capture_feature(self,module,args):self.feature=args[0].detach().transpose(1,2)
    def train(self,mode=True):super().train(mode);self.base.eval();return self
    def groups(self):return [dict(params=[p for p in self.parameters() if p.requires_grad],lr=1e-4,name='PERM_coordinate_conditioned_readout')]
    def initialize_priors(self,data):
        return dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,all154_positive_weights_frozen=True,
                    patch_input='frozen154 context_head expand input,128 hidden',zero_bin_correction=True,initial_uniform_background=.001,initial_function_changes_from154=True,initial_score_separate_from_learned_gain=True,
                    extra_active_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),coordinate_mode=self.mode,grid_values_train_fitted=True,
                    logmass_bound=.5,prior_floor=1e-12,own_bin_NLL_BCE_plus_joint_RPS1=True,
                    original_Gaussian16_prior_retained=True,old_gate_frozen=True,continuous_NLL_separately_normalized=True,learned_uniform_background_sigmoid=True,
                    initial_uniform_mix_nonidentity=True,
                    executed_training_stages_including_source=6,added_PERM_stages=1,no_other_repair_weights_loaded=True)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.feature=None
        with torch.no_grad():
            gate,raw=self.base(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
            flat=raw[:,:,0].float().reshape(-1,3,16);parts=[]
            for chunk in flat.split(4096):
                parts.append(probability(1,chunk[:,0].softmax(-1),chunk[:,1],F.softplus(chunk[:,2])+.025,self.edges))
            prior=torch.cat(parts).reshape(*x.shape[:2],256)
        feature=self.feature;self.feature=None
        def chunk_readout(h,p):
            query=self.query(h);value=self.value(self.coordinate)
            interaction=query[:,None,:]*value[None] if self.mode=='bilinear' else torch.tanh(query[:,None,:]+value[None])
            logits=self.projection(interaction).squeeze(-1).float()
            alpha=self.background(query).float().sigmoid()
            return (((1-alpha)*p+alpha/256).clamp_min(1e-12).log()+.5*torch.tanh(logits/.5)).log_softmax(-1)
        h=feature.reshape(-1,128);p=prior.reshape(-1,256);parts=[]
        for hh,pp in zip(h.split(512),p.split(512)):
            if self.training:parts.append(checkpoint(chunk_readout,hh,pp,use_reentrant=False,preserve_rng_state=False))
            else:parts.append(chunk_readout(hh,pp))
        logmass=torch.cat(parts).reshape(*x.shape[:2],256)
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

def make_model(stage,master,metadata):return Network(master,metadata,stage)

def channel_objective(output,labels,default,mask,quality,c,mode,model):
    assert c==1
    gate,logmass=output;g=gate[mask][:,0].float();logp=logmass[mask][:,0].float()
    d=default[mask][:,1];z=labels[mask][:,1].float();q=quality[mask].detach().float()
    edges=model.edges[1:-1];bins=torch.bucketize(z.contiguous(),edges)
    nll=-logp.gather(1,bins[:,None]).squeeze(1).masked_fill(d,0)
    bce=F.binary_cross_entropy_with_logits(g,d.float(),reduction='none')
    continuous_weight=(q*(~d).float()).sum().clamp_min(1e-12)
    continuous=(q*nll).sum()/continuous_weight*q.mean()
    likelihood=((q*bce).mean()+continuous)/3
    prob=g.sigmoid()[:,None];atom=(model.base.default_z<=edges).float()[None]
    joint=prob*atom+(1-prob)*logp.exp().cumsum(-1)[:,:-1]
    target=torch.where(d,model.base.default_z,z)
    empirical=(target[:,None]<=edges[None]).float()
    rps=(q*(joint-empirical).square().mean(-1)).mean()/3
    return likelihood+rps,(q*bce).mean(),(q*nll).mean()+3*rps
