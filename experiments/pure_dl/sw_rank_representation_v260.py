"""Own continuous-density backbone versus frozen-body readout, fixed atom network."""
import hashlib,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from compact_frequency_v104 import ROOT,SOURCE,Network as Positive,load_data
from train_stage_fullcv_v97 import Evaluator as ParentEvaluator
from multiscale_sequence import window_indices
import train_invariant as base
STUDY=ROOT/'results/deep_learning/sw_rank_representation_v260_20261001'
OUT=ROOT/'outputs/pure_dl_sw_rank_representation_v260_fold5'
PARENT=ROOT/'outputs/pure_dl_compact_frequency_v104_fold5/local/model_best.pt'
PARENT_SHA='07e0b8df208fe93c8ac07dcf18924459c47afb12ec84b1b7306a0e6017f58670'
STAGES={mode:('SW',1024,False,0.,1e-4) for mode in ('ce','rnc')}

def transport_probability(raw,logits,edges,mode):
    from sw_ff_mass_coadapt_v242 import finite_probability
    prior=finite_probability(raw,edges)
    alpha=logits[:,-1:].sigmoid();background=(1-alpha)*prior+alpha/256
    return (background.clamp_min(1e-12).log()+.5*torch.tanh(logits[:,:256]/.5)).softmax(-1)

class Network(nn.Module):
    def __init__(self,master,metadata,mode):
        super().__init__();self.mode=mode
        assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
        state=torch.load(PARENT,map_location='cpu',weights_only=False)['state_dict']
        self.base=Positive(metadata,'local');self.base.load_state_dict(state,strict=True)
        self.base.requires_grad_(True).eval()
        self.atom=Positive(metadata,'local');self.atom.load_state_dict(state,strict=True);self.atom.requires_grad_(False).eval()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010);self.patch=nn.Sequential(nn.Linear(128,32),nn.GELU(),nn.Linear(32,257))
            nn.init.zeros_(self.patch[-1].weight);nn.init.zeros_(self.patch[-1].bias)
            with torch.no_grad():self.patch[-1].bias[-1]=math.log(.001/.999)
        self.feature=None;self.handle=self.base.output.register_forward_pre_hook(self.capture)
    def capture(self,module,args):self.feature=args[0]
    def train(self,mode=True):super().train(mode);self.base.eval();self.atom.eval();return self
    def groups(self):
        groups=[dict(params=list(self.patch.parameters()),lr=1e-4,name='SW_mass_readout')]
        if True:groups.append(dict(params=list(self.base.parameters()),lr=3e-5,name='SW_existing_continuous_backbone'))
        return groups
    def initialize_priors(self,data):
        return dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,mode=self.mode,
                    original_atom_network_frozen=True,base_body_trainable=True,
                    old_density_output_trainable=True,unused_base_gate_rows_receive_no_loss=True,
                    trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    new_mass_parameters=12609,initial_uniform_background=.001,
                    eval_mode_no_DropKey_but_gradient_enabled=True,SAM_rho=0,
                    one_joint_added_SW_stage=True,no_trained_repair_weights=True)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        kwargs=dict(absolute_depth=absolute_depth,well_inputs=well_inputs)
        with torch.no_grad():gate,_=self.atom(x,relative,valid,**kwargs)
        self.feature=None
        if True:_,raw=self.base(x,relative,valid,**kwargs)
        else:
            with torch.no_grad():_,raw=self.base(x,relative,valid,**kwargs)
        hidden=self.feature;logits=self.patch(hidden).float();self.feature=None
        return gate,raw,logits,hidden

class Evaluator(ParentEvaluator):
    def __init__(self,data,master,c,length,mode):super().__init__(data,master,c,length);self.mode=mode
    @torch.no_grad()
    def __call__(self,model):
        d=self.data;model.eval();lo=self.length//2-64;hi=lo+128
        result=np.full(len(d['validation']),np.nan,np.float64);coverage=np.zeros(len(result),np.int8)
        for begin in range(0,len(self.centers),self.batch):
            cc=torch.tensor(self.centers[begin:begin+self.batch],device='cuda');anchor=torch.tensor(self.anchors[begin:begin+self.batch],device='cuda')
            ix,valid=window_indices(cc,d['lower'],d['upper'],length=self.length,anchors=anchor)
            rel=torch.tanh((d['depth'][ix]-d['depth'][anchor,None])/10);context=d['well_bank'][d['row_well'][anchor]]
            with torch.autocast('cuda',dtype=torch.bfloat16):gate,raw,logits,_=model(d['x'][ix],rel,valid,absolute_depth=d['absolute'][ix],well_inputs=context)
            rows=d['raw_to_valid'][ix[:,lo:hi].cpu().numpy()];use=valid[:,lo:hi].cpu().numpy()&(rows>=0);dest=self.local[rows[use]]
            assert (dest>=0).all() and (coverage[dest]==0).all() and len(np.unique(dest))==len(dest)
            coverage[dest]+=1;take=torch.as_tensor(use,device='cuda')
            g=gate[:,lo:hi][take][:,0].float().sigmoid();raw=raw[:,lo:hi][take][:,0].float();logits=logits[:,lo:hi][take].float()
            mass=transport_probability(raw,logits,self.edges,self.mode)
            joint=torch.cat((g[:,None],(1-g[:,None])*mass),-1)
            result[dest]=self.actions[(joint@self.utilities).argmax(-1).cpu().numpy()]
        assert (coverage==1).all() and np.isfinite(result).all()
        y=d['truth'][d['validation']].copy();pred=y.copy();pred[:,2]=result
        return result,float(base.metric(y,pred)['SW'])

def ordered_representation_loss(hidden,physical,raw_rows,well_ids):
    # Fixed row spacing and quotas use training row/well metadata, never model inputs.
    take=torch.nonzero(raw_rows.remainder(32)==0,as_tuple=False).flatten()
    if take.numel()<3:return hidden.sum()*0,0
    rows=raw_rows[take];unique,inverse=torch.unique(rows,sorted=True,return_inverse=True)
    first=torch.full((unique.numel(),),take.numel(),device=take.device,dtype=torch.long)
    first.scatter_reduce_(0,inverse,torch.arange(take.numel(),device=take.device),reduce='amin',include_self=True)
    take=take[first]
    hashed=(raw_rows[take]*1103515245+12345).bitwise_and(2147483647)
    take=take[hashed.argsort(stable=True)]
    take=take[well_ids[take].argsort(stable=True)]
    _,counts=torch.unique_consecutive(well_ids[take],return_counts=True)
    start=torch.repeat_interleave(counts.cumsum(0)-counts,counts)
    within=torch.arange(take.numel(),device=take.device)-start
    take=take[within<8][:256]
    if take.numel()<3:return hidden.sum()*0,int(take.numel())
    h=hidden[take].float();y=physical[take].detach().float();n=h.shape[0]
    remove=~torch.eye(n,device=h.device,dtype=torch.bool)
    diffs=(y[:,None]-y[None]).abs()[remove].reshape(n,n-1)
    logits=(-torch.cdist(h,h,p=2)/2)[remove].reshape(n,n-1)
    logits=logits-logits.max(-1,keepdim=True).values.detach()
    order=diffs.argsort(dim=-1,stable=True);distance=diffs.gather(1,order);ordered=logits.gather(1,order)
    tails=ordered.exp().flip(-1).cumsum(-1).flip(-1)
    tied_first=torch.searchsorted(distance.contiguous(),distance.contiguous(),right=False)
    denominator=tails.gather(1,tied_first).clamp_min(1e-20)
    return (denominator.log()-ordered).mean(),n

class DecisionObjective(nn.Module):
    def __init__(self,evaluator,metadata,mode):
        super().__init__();self.alpha=0.;self.mode=mode;self.last_rnc=torch.tensor(0.);self.last_samples=0
        self.register_buffer('edges',evaluator.edges.detach().clone())
        self.register_buffer('physical_centers',torch.as_tensor(metadata['centers'][2],dtype=torch.float32))
        self.median=metadata['sw_target_median'];self.scale=metadata['sw_target_scale']
    def forward(self,output,labels,default,mask,quality,raw_rows,well_ids):
        gate,raw,logits,hidden=output;g=gate[mask][:,0].float();raw=raw[mask][:,0].float();logits=logits[mask].float()
        def prob(r,h):return transport_probability(r,h,self.edges,self.mode)
        parts=[checkpoint(prob,a,b,use_reentrant=False,preserve_rng_state=False) for a,b in zip(raw.split(1024),logits.split(1024))]
        logp=torch.cat(parts).clamp_min(1e-12).log().log_softmax(-1)
        d=default[mask][:,2];z=labels[mask][:,2].float();q=quality[mask].detach().float();bins=torch.bucketize(z.contiguous(),self.edges[1:-1])
        hard=-logp.gather(1,bins[:,None]).squeeze(1);truth=torch.exp(z*self.scale+self.median)
        soft=(1-(self.physical_centers[None]-truth[:,None]).abs()/(.05*(truth.abs()[:,None]+1e-5))).clamp_min(0)
        total=soft.sum(-1,keepdim=True);fallback=F.one_hot(bins,num_classes=256).float()
        soft=torch.where(total>0,soft/total.clamp_min(1e-12),fallback).detach()
        nll=(.5*hard-.5*(soft*logp).sum(-1)).masked_fill(d,0)
        bce=F.binary_cross_entropy_with_logits(g,d.float(),reduction='none');loss=(q*(bce+nll)).mean()/3
        continuous=~d
        aux,n=ordered_representation_loss(hidden[mask][continuous],truth[continuous],raw_rows[mask][continuous],well_ids[mask][continuous])
        self.last_rnc=aux.detach();self.last_samples=n
        return loss+(.1/3 if self.mode=='rnc' else 0.)*aux,loss,aux

def make_model(stage,master,metadata):return Network(master,metadata,stage)
def channel_objective(*args):raise RuntimeError('Own SW conditional mass objective only')
