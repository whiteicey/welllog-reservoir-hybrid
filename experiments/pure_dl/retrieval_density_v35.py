"""Frozen SW plus a supervised cross-well reference density.
Two equal-capacity models differ only in reference key/label correspondence.
Fixed input Top16 candidates; learned within-pool attention, bandwidth and mixture.
"""
from pathlib import Path
import hashlib,json,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from crosscurve_v32 import Network as FrozenBuilder
from champion_all_heads_v26 import load_data as parent_data,TARGETS
from train_scaling_v21 import Evaluator as OriginalEvaluator
from por_t3_patch_loss import probability
from multiscale_sequence import window_indices
import train_invariant as base

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/retrieval_density_v35_20260929'
PREVIOUS=ROOT/'results/deep_learning/mechanism_checks_v31_20260929'
MODES=('matched','shuffled')
LEARNING_RATE=3e-4

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')
def load_data():return parent_data()

class RetrievalHead(nn.Module):
    def __init__(self,mode):
        super().__init__()
        if mode not in MODES:raise ValueError(mode)
        self.mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261137)
            self.query=nn.Linear(319,32)
            self.key=nn.Linear(27,32)
            self.output=nn.Linear(64,2)
            nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        with np.load(STUDY/'memory.npz') as z:
            self.register_buffer('reference_features',torch.from_numpy(z['features'].copy()),persistent=False)
            self.register_buffer('reference_mu',torch.from_numpy(z['target' if mode=='matched' else 'shuffled_target'].copy()),persistent=False)
        self.initial_scale_raw=float(np.log(np.expm1(.15-.025)))

    def forward(self,hidden,parent,context,side,valid):
        ids,dist=context
        if side.shape[-1]!=63:raise ValueError('Parent amplitude63 is unchanged')
        x=torch.cat((hidden[0].transpose(1,2).float(),hidden[1].transpose(1,2).float(),side.float()),-1)
        x=x.masked_fill(~valid[...,None],0)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            q=F.gelu(self.query(x));keys=self.key(self.reference_features)
        k=keys[ids].float()
        logits=(q.float()[...,None,:]*k).sum(-1)/math.sqrt(32)-dist.float()
        weights=logits.softmax(-1)
        message=(weights[...,None]*k).sum(-2)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            params=self.output(torch.cat((q.float(),message),-1)).float()
        # The closed interval contains zero. torch.clamp has an inward derivative
        # at its boundary; negative proposals remain inactive and are diagnosed.
        alpha=params[...,0].clamp(0.,.25).masked_fill(~valid,0)
        sd=.025+F.softplus(self.initial_scale_raw+.5*torch.tanh(params[...,1]/.5))
        return dict(parent=parent,logits=logits,mu=self.reference_mu[ids],sd=sd,alpha=alpha)

class Network(FrozenBuilder):
    def __init__(self,target,master,modes=MODES):
        if target!='SW':raise ValueError('This bounded study updates SW only')
        super().__init__(target,master,modes=())
        meta=json.loads((STUDY/'memory_manifest.json').read_text(encoding='utf-8'))
        assert meta['whole_query_well_exclusion_verified'] and not meta['validation_labels_used_for_memory']
        assert hashlib.sha256((STUDY/'memory.npz').read_bytes()).hexdigest()==meta['memory_sha256']
        self.register_buffer('neighbor_ids',torch.from_numpy(np.load(STUDY/'neighbor_ids.npy')).long(),persistent=False)
        self.register_buffer('neighbor_distance',torch.from_numpy(np.load(STUDY/'neighbor_distance.npy')),persistent=False)
        self.heads=nn.ModuleDict({m:RetrievalHead(m) for m in modes})
        self.metadata=dict(parent=self.choice,parent_input_dimensions=269,query_input=319,reference_input=27,latent_width=32,
            reference_count=meta['reference_count'],candidate_count=16,maximum_reference_mixture=.25,initial_bandwidth=.15,
            memory_manifest=str((STUDY/'memory_manifest.json').relative_to(ROOT)),memory_sha256=meta['memory_sha256'],
            mixture='(1-alpha)*parent_conditional + alpha*reference_Gaussian_mixture; default probability fixed',
            selection='matched may be adopted; shuffled is a negative control only',
            initialization='alpha=0 exactly; parent distribution arithmetic unchanged',
            conditional_log_density='float64 stable scaled mixture; original own-target NLL/Q50, not discretized CE',
            candidate_scope='fixed input distance, learned within-pool reranking; not full-memory learned nearest neighbors')
        self.train(False)

    @torch.no_grad()
    def extract(self,x,relative,valid,absolute,rows=None):
        if rows is None or x.shape[-1]!=269:raise ValueError('Need aligned raw indices and unchanged269 input')
        h,o,ctx,_=self.parent.extract(x,relative,valid,absolute)
        old=self.parent.heads[self.parent_mode](h,o,ctx,x[...,206:],valid)
        return h,old,(self.neighbor_ids[rows],self.neighbor_distance[rows]),None

    def forward(self,x,relative,valid,absolute_depth=None,factors=None,rows=None):
        h,o,ctx,_=self.extract(x,relative,valid,absolute_depth,rows=rows)
        modes=tuple(self.heads) if factors is None else tuple(factors)
        return {m:self.heads[m](h,o,ctx,x[...,206:],valid) for m in modes}
    def trainable_groups(self,mode):
        return [dict(params=list(self.heads[mode].parameters()),lr=LEARNING_RATE,name='reference_density')]


def channel_objective(output,truth,default,mask,quality,target_index):
    if target_index!=2:raise ValueError('SW objective only')
    gate,raw=output['parent'];r=raw[mask][:,0].double();g=gate[mask][:,0].float()
    d=default[mask][:,2];y=truth[mask][:,2].masked_fill(d,0).double()
    sd=F.softplus(r[:,2])+.025
    lp=-.5*((y[:,None]-r[:,1])/sd).square()-sd.log()-.5*math.log(2*math.pi)
    base_log=(r[:,0].log_softmax(-1)+lp).logsumexp(-1)
    mu=output['mu'][mask].double();rs=output['sd'][mask].double()[:,None]
    lp=-.5*((y[:,None]-mu)/rs).square()-rs.log()-.5*math.log(2*math.pi)
    ref_log=(output['logits'][mask].double().log_softmax(-1)+lp).logsumexp(-1)
    a=output['alpha'][mask].double()
    shift=torch.maximum(base_log,ref_log).detach()
    scaled=(1-a)*torch.exp(base_log-shift)+a*torch.exp(ref_log-shift)
    # Float64 avoids 0*exp(large) and preserves the mixture gradient at alpha=0.
    mixed=shift+scaled.clamp_min(torch.finfo(torch.float64).tiny).log()
    nll=(-mixed).masked_fill(d,0)
    bce=F.binary_cross_entropy_with_logits(g,d.float(),reduction='none')
    w=quality[mask].detach().double()
    loss=(w*(bce.double()+nll)).mean()/3
    return loss.float(),(w*bce).mean().float(),(w*nll).mean().float()

class Evaluator(OriginalEvaluator):
    @torch.no_grad()
    def many(self,model,factors):
        if self.c!=2:raise ValueError('SW only')
        model.eval();d=self.data;factors=tuple(factors)
        predictions={m:np.full(len(d['validation']),np.nan,np.float64) for m in factors}
        coverage=np.zeros(len(d['validation']),np.int8)
        for begin in range(0,len(self.centers),32):
            centers=torch.tensor(self.centers[begin:begin+32],device='cuda')
            anchors=torch.tensor(self.anchors[begin:begin+32],device='cuda')
            ix,valid=window_indices(centers,d['lower'],d['upper'],anchors=anchors)
            rel=torch.tanh((d['depth'][ix]-d['depth'][anchors,None])/10)
            outputs=model(d['x'][ix],rel,valid,absolute_depth=d['absolute'][ix],factors=factors,rows=ix)
            rows=d['raw_to_valid'][ix[:,64:192].cpu().numpy()]
            use=valid[:,64:192].cpu().numpy()&(rows>=0);loc=self.local[rows[use]]
            assert (loc>=0).all() and (coverage[loc]==0).all() and len(np.unique(loc))==len(loc)
            coverage[loc]+=1;take=torch.as_tensor(use,device='cuda')
            first=outputs[factors[0]];go,raw=first['parent']
            g=go[:,64:192][take][:,0].float().sigmoid();r=raw[:,64:192][take][:,0].float()
            p0=probability(2,r[:,0].softmax(-1),r[:,1],F.softplus(r[:,2])+.025,self.edges)
            for m,o in outputs.items():
                weights=o['logits'][:,64:192][take].float().softmax(-1)
                mu=o['mu'][:,64:192][take].float();sd=o['sd'][:,64:192][take].float()[:,None].expand_as(mu)
                pr=probability(2,weights,mu,sd,self.edges)
                alpha=o['alpha'][:,64:192][take].float()[:,None]
                p=p0+alpha*(pr-p0)
                mass=torch.cat((g[:,None],(1-g[:,None])*p),-1)
                predictions[m][loc]=self.actions[(mass@self.utilities).argmax(-1).cpu().numpy()]
        assert (coverage==1).all()
        result={}
        for m,p in predictions.items():
            assert np.isfinite(p).all();combined=self.reference.copy();combined[:,2]=p
            scores=base.metric(d['truth'][d['validation']],combined)
            result[m]=(p,float(scores['SW']),float(scores['TOTAL']))
        return result

@torch.no_grad()
def factor_scope(model,data,mode):
    anchors=data['train_raw'][torch.tensor([0,len(data['train_raw'])//2],device='cuda')]
    ix,valid=window_indices(anchors,data['lower'],data['upper'])
    rel=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
    h,parent,ctx,_=model.extract(data['x'][ix],rel,valid,data['absolute'][ix],rows=ix)
    out=model.heads[mode](h,parent,ctx,data['x'][ix,...,206:],valid)
    assert torch.equal(out['parent'][0],parent[0]) and torch.equal(out['parent'][1],parent[1])
    alpha=out['alpha'][valid];sd=out['sd'][valid]
    assert bool(torch.isfinite(alpha).all()) and float(alpha.min())>=0 and float(alpha.max())<=.250001
    return dict(mode=mode,default_and_all_parent_factors_unchanged=True,alpha_mean=float(alpha.mean()),
        alpha_max=float(alpha.max()),active_fraction=float((alpha>0).float().mean()),bandwidth_mean=float(sd.mean()),
        scope='Two fixed training windows; total reference mixture restricted to[0,.25]')

def mechanism_probe(model,data,mode):
    result=factor_scope(model,data,mode)
    assert result['alpha_max']==0
    result.update(exact_parent_initial_function=True,whole_query_well_excluded=True,reference_values_from_training_only=True)
    return result
