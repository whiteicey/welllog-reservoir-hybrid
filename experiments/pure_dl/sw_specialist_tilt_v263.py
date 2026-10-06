"""Matched terminal representation and continuous-density readout coupling, frozen104."""
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
STUDY=ROOT/'results/deep_learning/sw_specialist_tilt_v263_20261001'
OUT=ROOT/'outputs/pure_dl_sw_specialist_tilt_v263_fold5'
PARENT=ROOT/'outputs/pure_dl_compact_frequency_v104_fold5/local/model_best.pt'
PARENT_SHA='07e0b8df208fe93c8ac07dcf18924459c47afb12ec84b1b7306a0e6017f58670'
STAGES={mode:('SW',1024,False,0.,3e-4) for mode in ('tcn','bigru')}

def transport_probability(raw,logits,edges,mode):
    from sw_ff_mass_coadapt_v242 import finite_probability
    prior=finite_probability(raw,edges)
    background=.999*prior+.001/256
    return (background.clamp_min(1e-12).log()+3*torch.tanh(logits/3)).softmax(-1)

class SpecialistBlock(nn.Module):
    def __init__(self,dilation):
        super().__init__();self.norm=nn.LayerNorm(64);self.depth=nn.Conv1d(64,64,5,padding=2*dilation,dilation=dilation,groups=64)
        self.ff=nn.Sequential(nn.Conv1d(64,128,1),nn.GELU(),nn.Conv1d(128,64,1))
    def forward(self,h,valid):
        z=self.norm(h).masked_fill(~valid[...,None],0).transpose(1,2)
        return (h+self.ff(self.depth(z)).transpose(1,2)).masked_fill(~valid[...,None],0)

class Network(nn.Module):
    def __init__(self,master,metadata,mode):
        super().__init__();self.mode=mode;self.base=Positive(metadata,'local')
        assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
        self.base.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=False)['state_dict'],strict=True);self.base.requires_grad_(False).eval()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010);self.stem=nn.Sequential(nn.Linear(207,64),nn.LayerNorm(64),nn.GELU())
            if mode=='tcn':
                self.encoder=nn.ModuleList([SpecialistBlock(d) for d in (1,4,16,64)]);width=64
            else:self.encoder=nn.GRU(64,64,batch_first=True,bidirectional=True);width=128
            self.readout=nn.Linear(width,256);nn.init.zeros_(self.readout.weight);nn.init.zeros_(self.readout.bias)
    def train(self,mode=True):super().train(mode);self.base.eval();return self
    def groups(self):return [dict(params=[p for p in self.parameters() if p.requires_grad],lr=3e-4,name='SW_new_continuous_specialist_backbone_and_density')]
    def initialize_priors(self,data):
        return dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,all_parent_weights_frozen=True,original_atom_gate_frozen=True,
                    new_private_continuous_backbone=self.mode,input_features=207,prior_uniform_floor=.001,logmass_residual_bound=3.,initial_residual_zero=True,
                    trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    own_component_conditional_CE_coefficient=.5,continuous_loss_normalized=True,no_teacher=True,
                    one_joint_added_SW_stage=True,no_trained_repair_weights=True,remaining_folds_started=False)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        with torch.no_grad():gate,raw=self.base(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
        h=self.stem(torch.cat((x[...,:206],relative.unsqueeze(-1)),-1)).masked_fill(~valid[...,None],0)
        if self.mode=='tcn':
            for block in self.encoder:h=block(h,valid)
        else:
            # Keep cuDNN recurrent math float32; no label/ID/context teacher input.
            with torch.autocast(device_type=h.device.type,enabled=False):h,_=self.encoder(h.float())
            h=h.masked_fill(~valid[...,None],0)
        return gate,raw,self.readout(h).float()
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
            with torch.autocast('cuda',dtype=torch.bfloat16):gate,raw,logits=model(d['x'][ix],rel,valid,absolute_depth=d['absolute'][ix],well_inputs=context)
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

class DecisionObjective(nn.Module):
    def __init__(self,evaluator,metadata,mode):
        super().__init__();self.mode=mode;self.alpha=0.;self.last_expert_CE=torch.tensor(0.)
        self.register_buffer('edges',evaluator.edges.detach().clone())
        self.register_buffer('physical_centers',torch.as_tensor(metadata['centers'][2],dtype=torch.float32))
        self.median=metadata['sw_target_median'];self.scale=metadata['sw_target_scale']
    def forward(self,output,labels,default,mask,quality,**kwargs):
        gate,raw,logits=output;g=gate[mask][:,0].float();raw=raw[mask][:,0].float();logits=logits[mask].float()
        def prob(a,b):return transport_probability(a,b,self.edges,self.mode)
        parts=[checkpoint(prob,a,b,use_reentrant=False,preserve_rng_state=False) for a,b in zip(raw.split(1024),logits.split(1024))]
        logp=torch.cat(parts).clamp_min(1e-12).log();expert_logp=logits.log_softmax(-1)
        d=default[mask][:,2];z=labels[mask][:,2].float();q=quality[mask].detach().float();bins=torch.bucketize(z.contiguous(),self.edges[1:-1])
        truth=torch.exp(z*self.scale+self.median)
        soft=(1-(self.physical_centers[None]-truth[:,None]).abs()/(.05*(truth.abs()[:,None]+1e-5))).clamp_min(0)
        total=soft.sum(-1,keepdim=True);fallback=F.one_hot(bins,num_classes=256).float()
        soft=torch.where(total>0,soft/total.clamp_min(1e-12),fallback).detach()
        def ce(lp):return (.5*(-lp.gather(1,bins[:,None]).squeeze(1))-.5*(soft*lp).sum(-1)).masked_fill(d,0)
        mass_ce=ce(logp);own_ce=ce(expert_logp);denom=(q*(~d).float()).sum().clamp_min(1e-12)
        mass=(q*mass_ce).sum()/denom*q.mean();own=(q*own_ce).sum()/denom*q.mean()
        bce=(q*F.binary_cross_entropy_with_logits(g,d.float(),reduction='none')).mean()
        self.last_expert_CE=own.detach()
        return (bce+mass+.5*own)/3,mass,own

def make_model(stage,master,metadata):return Network(master,metadata,stage)
def channel_objective(*args):raise RuntimeError('SW own conditional specialist objective only')
