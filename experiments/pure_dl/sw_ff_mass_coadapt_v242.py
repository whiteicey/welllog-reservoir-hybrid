"""One-stage joint input memory and probability readout on frozen104."""
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
STUDY=ROOT/'results/deep_learning/sw_ff_mass_coadapt_v242_20261001'
OUT=ROOT/'outputs/pure_dl_sw_ff_mass_coadapt_v242_fold5'
PARENT=ROOT/'outputs/pure_dl_compact_frequency_v104_fold5/local/model_best.pt'
PARENT_SHA='07e0b8df208fe93c8ac07dcf18924459c47afb12ec84b1b7306a0e6017f58670'
STAGES={'joint':('SW',1024,True,0.,1e-4)}
def finite_probability(raw,edges):
    weight=raw[:,0].softmax(-1);mu=raw[:,1];sd=F.softplus(raw[:,2])+.025
    z=(edges[None,1:-1,None]-mu[:,None,:])/sd[:,None,:]
    finite=.5*(1+torch.erf(z/math.sqrt(2)))
    cdf=torch.cat((torch.zeros_like(finite[:,:1]),finite,torch.ones_like(finite[:,:1])),1)
    mass=(cdf.diff(dim=1).clamp_min(0)*weight[:,None,:]).sum(-1)
    return mass/mass.sum(-1,keepdim=True).clamp_min(1e-12)
class Network(nn.Module):
    def __init__(self,master,metadata):
        super().__init__();self.base=Positive(metadata,'local')
        assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
        self.base.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=False)['state_dict'],strict=True)
        self.base.requires_grad_(False).eval()
        self.base.body.global_blocks[-1].ff.requires_grad_(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010);self.patch=nn.Sequential(nn.Linear(128,32),nn.GELU(),nn.Linear(32,256))
            nn.init.zeros_(self.patch[-1].weight);nn.init.zeros_(self.patch[-1].bias)
        self.feature=None;self.handle=self.base.output.register_forward_pre_hook(self.capture)
    def capture(self,module,args):self.feature=args[0]
    def train(self,mode=True):
        super().train(mode);self.base.eval();return self
    def groups(self):return [dict(params=list(self.base.body.global_blocks[-1].ff.parameters()),lr=3e-5,name='SW_existing_final_FF_only'),dict(params=list(self.patch.parameters()),lr=1e-4,name='SW_new_mass_readout_only')]
    def initialize_priors(self,data):
        return dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,
                    only_existing_final_global_FF_and_new_mass_head_train=True,
                    original_FF_low_lr=3e-5,mass_head_lr=1e-4,mass_parameters=12576,
                    trainable_FF_parameters=sum(p.numel() for p in self.base.body.global_blocks[-1].ff.parameters()),
                    all_other_positive_parent_parameters_frozen=True,
                    no_new_memory_or_trained_repair_weights=True,one_joint_stage=True,
                    categoricalNLL_BCE_utility0=True,finite_edge_CDF_gradient=True,
                    gate_may_change_via_feature_coadapt=True,executed_training_stages_including_source=6,added_SW_stages=1)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.feature=None
        gate,raw=self.base(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
        correction=.5*torch.tanh(self.patch(self.feature).float()/.5);self.feature=None
        return gate,raw,correction
class Evaluator(ParentEvaluator):
    def __init__(self,data,master,c,length,mode):super().__init__(data,master,c,length)
    @torch.no_grad()
    def __call__(self,model):
        d=self.data;model.eval();lo=self.length//2-64;hi=lo+128
        result=np.full(len(d['validation']),np.nan,np.float64);coverage=np.zeros(len(result),np.int8)
        for begin in range(0,len(self.centers),self.batch):
            cc=torch.tensor(self.centers[begin:begin+self.batch],device='cuda');anchor=torch.tensor(self.anchors[begin:begin+self.batch],device='cuda')
            ix,valid=window_indices(cc,d['lower'],d['upper'],length=self.length,anchors=anchor)
            rel=torch.tanh((d['depth'][ix]-d['depth'][anchor,None])/10);context=d['well_bank'][d['row_well'][anchor]]
            with torch.autocast('cuda',dtype=torch.bfloat16):gate,raw,correction=model(d['x'][ix],rel,valid,absolute_depth=d['absolute'][ix],well_inputs=context)
            rows=d['raw_to_valid'][ix[:,lo:hi].cpu().numpy()];use=valid[:,lo:hi].cpu().numpy()&(rows>=0);dest=self.local[rows[use]]
            assert (dest>=0).all() and (coverage[dest]==0).all() and len(np.unique(dest))==len(dest)
            coverage[dest]+=1;take=torch.as_tensor(use,device='cuda')
            g=gate[:,lo:hi][take][:,0].float().sigmoid();r=raw[:,lo:hi][take][:,0].float();delta=correction[:,lo:hi][take].float()
            cp=finite_probability(r,self.edges);mass=(cp.clamp_min(1e-12).log()+delta).softmax(-1)
            joint=torch.cat((g[:,None],(1-g[:,None])*mass),-1)
            result[dest]=self.actions[(joint@self.utilities).argmax(-1).cpu().numpy()]
        assert (coverage==1).all() and np.isfinite(result).all()
        y=d['truth'][d['validation']].copy();pred=y.copy();pred[:,2]=result
        return result,float(base.metric(y,pred)['SW'])
class DecisionObjective(nn.Module):
    def __init__(self,evaluator,metadata,mode):
        super().__init__();self.alpha=0.;self.register_buffer('edges',evaluator.edges.detach().clone())
    def forward(self,output,labels,default,mask,quality):
        gate,raw,correction=output;g=gate[mask][:,0].float();r=raw[mask][:,0].float();delta=correction[mask].float()
        parts=[checkpoint(finite_probability,chunk,self.edges,use_reentrant=False,preserve_rng_state=False) for chunk in r.split(1024)]
        prior=torch.cat(parts);logp=(prior.clamp_min(1e-12).log()+delta).log_softmax(-1)
        d=default[mask][:,2];z=labels[mask][:,2].float();q=quality[mask].detach().float();bins=torch.bucketize(z.contiguous(),self.edges[1:-1])
        nll=-logp.gather(1,bins[:,None]).squeeze(1).masked_fill(d,0);bce=F.binary_cross_entropy_with_logits(g,d.float(),reduction='none')
        loss=(q*(nll+bce)).mean()/3
        return loss,loss,loss*0

def make_model(stage,master,metadata):return Network(master,metadata)
def channel_objective(*args):raise RuntimeError('Own SW joint bin-likelihood only')
