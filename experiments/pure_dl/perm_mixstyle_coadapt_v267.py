"""Frozen154 Gaussian16 prior plus bounded PERM conditional mass readout."""
import hashlib,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from perm_ranked_probability_v154 import ROOT,SOURCE,Network as Positive,load_data
from train_stage_fullcv_v97 import Evaluator as ParentEvaluator
from por_t3_patch_loss import probability
from multiscale_sequence import window_indices
import train_invariant as base
STUDY=ROOT/'results/deep_learning/perm_mixstyle_coadapt_v267_20261001'
OUT=ROOT/'outputs/pure_dl_perm_mixstyle_coadapt_v267_fold5'
PARENT=ROOT/'outputs/pure_dl_perm_ranked_probability_v154_fold5/perm_conditional/model_best.pt'
PARENT_SHA='2b355bb8c3736990b05e2726373a4343c574a93ab6ab1a3eda0ef796ac70cf11'
STAGES={a:('PERM',256,False,0.,1e-4) for a in ('control','mixstyle')}

class Network(nn.Module):
    def __init__(self,master,metadata,mode):
        super().__init__();self.mode=mode;self.training_wells=None;self.style_valid=None;self.base=Positive(master,'perm_conditional')
        assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
        self.base.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=False)['state_dict'],strict=True)
        self.base.requires_grad_(False).eval();self.base.context_head.requires_grad_(True)
        self.register_buffer('edges',torch.as_tensor(master['density_edges'][1],dtype=torch.float32))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010)
            self.patch=nn.Sequential(nn.Linear(128,32),nn.GELU(),nn.Linear(32,257))
            nn.init.zeros_(self.patch[-1].weight);nn.init.zeros_(self.patch[-1].bias)
            with torch.no_grad():self.patch[-1].bias[-1]=math.log(.001/.999)
        self.feature=None;self.handle=self.base.context_head.expand.register_forward_pre_hook(self.capture_feature)
    def capture_feature(self,module,args):
        h=args[0];out=h
        if self.training:
            assert self.training_wells is not None and self.style_valid is not None
            ids=self.training_wells;valid=self.style_valid[:,None].float();x=h.float();denom=valid.sum(-1,keepdim=True).clamp_min(1)
            mean=(x*valid).sum(-1,keepdim=True)/denom;var=((x-mean).square()*valid).sum(-1,keepdim=True)/denom
            mean=mean.detach();std=(var+1e-6).sqrt().detach();n=len(h)
            rank=torch.rand(n,n,device=h.device);rank=rank.masked_fill(ids[:,None]==ids[None,:],float('inf'));partner=rank.argmin(-1)
            apply=(torch.rand(n,device=h.device)<.25)&(ids!=ids[partner]);strength=.1*torch.rand(n,1,1,device=h.device)*apply[:,None,None]
            mixed=(x-mean)/std*((1-strength)*std+strength*std[partner])+(1-strength)*mean+strength*mean[partner]
            if self.mode=='mixstyle':out=mixed.to(h.dtype).masked_fill(~self.style_valid[:,None],0)
        self.feature=out.transpose(1,2);return (out,)

    def train(self,mode=True):super().train(mode);self.base.eval();return self
    def groups(self):return [dict(params=list(self.patch.parameters()),lr=1e-4,name='PERM_probability_mass_patch'),dict(params=list(self.base.context_head.parameters()),lr=1e-5,name='PERM_existing_context_readout')]
    def initialize_priors(self,data):
        return dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,original_SOURCE_heads_routes_frozen=True,existing154_context_head_trainable=True,training_style_probability=.25,maximum_style_strength=.1,eval_style_identity=True,
                    patch_input='frozen154 context_head expand input,128 hidden',zero_bin_correction=True,initial_uniform_background=.001,initial_function_changes_from154=True,initial_score_separate_from_learned_gain=True,
                    extra_active_parameters=sum(p.numel() for p in self.patch.parameters()),
                    logmass_bound=.5,prior_floor=1e-12,own_bin_NLL_BCE_plus_joint_RPS1=True,
                    original_Gaussian16_prior_retained=True,old_gate_frozen=True,continuous_NLL_separately_normalized=True,learned_uniform_background_sigmoid=True,
                    initial_uniform_mix_nonidentity=True,
                    executed_training_stages_including_source=6,added_PERM_stages=1,no_other_repair_weights_loaded=True)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.feature=None
        self.style_valid=valid
        gate,raw=self.base(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
        from sw_ff_mass_coadapt_v242 import finite_probability
        from torch.utils.checkpoint import checkpoint
        flat=raw[:,:,0].float().reshape(-1,3,16)
        prior=torch.cat([checkpoint(finite_probability,chunk,self.edges,use_reentrant=False,preserve_rng_state=False) for chunk in flat.split(512)]).reshape(*x.shape[:2],256)
        patch=self.patch(self.feature).float();self.feature=None
        alpha=patch[:,:,-1:].sigmoid();correction=.5*torch.tanh(patch[:,:,:256]/.5)
        background=(1-alpha)*prior+alpha/256
        logmass=(background.clamp_min(1e-12).log()+correction).log_softmax(-1)
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

def channel_objective(output,labels,default,mask,quality,c,mode,model,well_ids):
    assert c==1
    gate,logmass=output;g=gate[mask][:,0].float();logp=logmass[mask][:,0].float()
    d=default[mask][:,1];z=labels[mask][:,1].float();q=quality[mask].detach().float()
    edges=model.edges[1:-1];bins=torch.bucketize(z.contiguous(),edges)
    nll=-logp.gather(1,bins[:,None]).squeeze(1).masked_fill(d,0)
    bce=F.binary_cross_entropy_with_logits(g,d.float(),reduction='none')
    continuous_weight=(q*(~d).float()).sum().clamp_min(1e-12)
    continuous=(q*nll).sum()/continuous_weight*q.mean()
    ids=well_ids[mask];unique,inverse=torch.unique(ids,sorted=True,return_inverse=True)
    weights=q*(~d).float();denom=torch.zeros(len(unique),device=q.device).scatter_add(0,inverse,weights)
    counts=torch.zeros(len(unique),device=q.device).scatter_add(0,inverse,(~d).float())
    sums=torch.zeros(len(unique),device=q.device).scatter_add(0,inverse,q*nll)
    eligible=counts>=8
    means=sums[eligible]/denom[eligible].clamp_min(1e-12)
    model.last_group_risk=float(continuous.detach());model.last_group_count=len(means);model.last_max_group_weight=0.
    if len(means):
        equal=means.mean()
        grouped=equal
        continuous=.5*continuous+.5*grouped*q.mean()
        model.last_group_risk=float(grouped.detach());model.last_group_count=len(means)
        model.last_max_group_weight=1/len(means)
    likelihood=((q*bce).mean()+continuous)/3
    prob=g.sigmoid()[:,None];atom=(model.base.default_z<=edges).float()[None]
    joint=prob*atom+(1-prob)*logp.exp().cumsum(-1)[:,:-1]
    target=torch.where(d,model.base.default_z,z)
    empirical=(target[:,None]<=edges[None]).float()
    rps=(q*(joint-empirical).square().mean(-1)).mean()/3
    return likelihood+rps,(q*bce).mean(),(q*nll).mean()+3*rps
