"""Low-dimensional monotone SW density transforms; original default mass fixed.

F_new(z|x)=F_parent(T(z;x)|x), f_new=f_parent(T(z))*T'(z).
Affine: two bounded translation/log-scale outputs, positive slope on R.
PWL4: three bounded knot displacements, four linear pieces, identity tails.
This is a piecewise-linear flow, not a reproduction of rational-quadratic NSF.
"""
from pathlib import Path
import json,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from champion_all_heads_v26 import load_data as parent_data,TARGETS
from champion_split_route_v29 import Network as ParentModel
from train_scaling_v21 import Evaluator as OriginalEvaluator
import train_invariant as base

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/mechanism_checks_v31_20260929'
PREVIOUS=ROOT/'results/deep_learning/champion_coadapt_v30_20260929'
MODES=('affine','pwl4')
LEARNING_RATE=3e-4


def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')


def load_data():
    data,master=parent_data()
    tr=data['train_raw'];v=data['labels'][tr,2][~data['default'][tr,2]].float()
    knots=torch.quantile(v,torch.tensor([0.,.25,.5,.75,1.],device=v.device)).cpu().numpy()
    if not np.all(np.diff(knots)>1e-5):raise RuntimeError('Non-distinct training SW quantile knots')
    master=dict(master);master['v31_knots']=knots.tolist()
    return data,master


def transform(z,parameters,knots):
    """z:NxM, parameters:Nx2/3, knots:5 -> transformed z and log derivative.

    Infinite CDF edges are preserved explicitly; no 0*infinity calculations.
    """
    z=z.float();parameters=parameters.float()
    finite=torch.isfinite(z);safe=torch.where(finite,z,torch.zeros_like(z))
    if parameters.shape[-1]==2:
        log_scale=.125*torch.tanh(parameters[:,0:1]/.125)
        translation=.25*torch.tanh(parameters[:,1:2]/.25)
        value=safe+torch.expm1(log_scale)*safe+translation
        derivative_log=log_scale.expand_as(safe)
    elif parameters.shape[-1]==3:
        widths=knots[1:]-knots[:-1]
        bounds=.25*torch.minimum(widths[:-1],widths[1:])
        internal=bounds*torch.tanh(parameters/bounds)
        offsets=F.pad(internal,(1,1),value=0.)
        index=torch.bucketize(safe.contiguous(),knots[1:-1].contiguous(),right=True)
        width=widths[index];fraction=(safe-knots[index])/width
        left=offsets.gather(1,index);right=offsets.gather(1,index+1)
        inside=(safe>=knots[0])&(safe<=knots[-1])
        shift=left+(right-left)*fraction
        slope=1+(right-left)/width
        value=safe+torch.where(inside,shift,torch.zeros_like(shift))
        derivative_log=torch.where(inside,slope.log(),torch.zeros_like(slope))
    else:raise ValueError('Expected two affine or three PWL parameters')
    return torch.where(finite,value,z),derivative_log


class WarpHead(nn.Module):
    def __init__(self,mode,knots):
        super().__init__();self.mode=mode
        self.register_buffer('knots',torch.tensor(knots,dtype=torch.float32))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261107)
            self.input=nn.Conv1d(319,32,1)
            self.output=nn.Conv1d(32,2 if mode=='affine' else 3,1)
            nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,hidden,parent,context,side,valid):
        x=torch.cat((hidden[0],hidden[1],side.transpose(1,2).to(hidden[0].dtype)),1).masked_fill(~valid[:,None],0)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            parameters=self.output(F.gelu(self.input(x))).transpose(1,2).float()
        parameters=parameters.masked_fill(~valid[...,None],0)
        return parent[0],parent[1],parameters,self.knots


class Network(ParentModel):
    def __init__(self,target,master,modes=MODES):
        if target!='SW':raise ValueError('v31 density transform is SW only')
        recipe=json.loads((PREVIOUS/'CURRENT_RECIPE.json').read_text(encoding='utf-8'))
        self.choice=next(c for c in recipe['choices'] if c['target']==target)
        parent_mode=self.choice['mode']
        super().__init__(target,master,modes=(parent_mode,))
        cp=torch.load(ROOT/self.choice['checkpoint'],map_location='cpu',weights_only=False)
        self.heads[parent_mode].load_state_dict(cp['delta_state'],strict=True)
        self.retained_v29=self.heads[parent_mode];self.retained_v29.requires_grad_(False)
        self.heads=nn.ModuleDict({m:WarpHead(m,master['v31_knots']) for m in modes})
        self.factor='conditional_density_transform';self.train(False)
        self.metadata=dict(knots=master['v31_knots'],knot_quantiles=[0,.25,.5,.75,1],knot_fit='only SW nondefault labels of original64 training wells',
            pwl_internal_shift_bound='0.25*min(adjacent knot spacings)',pwl_slope_lower_bound=.5,
            affine_log_scale_bound=.125,affine_translation_bound=.25,default_probability_fixed=True,
            density_family='frozen16component Gaussian transformed with exact Jacobian')

    @torch.no_grad()
    def extract(self,x,relative,valid,absolute):
        h,o,contexts,_=super().extract(x,relative,valid,absolute)
        parent=self.retained_v29(h,o,contexts,x[...,206:],valid)
        return h,parent,None,None

    def trainable_groups(self,mode):
        return [dict(params=list(self.heads[mode].parameters()),lr=LEARNING_RATE,name='monotone_warp')]


def channel_objective(output,truth,default,mask,quality,target_index):
    if target_index!=2:raise ValueError(target_index)
    gate,raw,parameters,knots=output
    g=gate[mask][:,0].float();r=raw[mask][:,0].float();q=parameters[mask]
    d=default[mask][:,2];y=truth[mask][:,2].masked_fill(d,0)
    z,log_jacobian=transform(y[:,None],q,knots)
    mu=r[:,1];std=F.softplus(r[:,2])+.025
    logp=-.5*((z-mu)/std).square()-std.log()-.5*math.log(2*math.pi)
    log_density=(r[:,0].log_softmax(-1)+logp).logsumexp(-1)+log_jacobian[:,0]
    nll=-log_density.masked_fill(d,0)
    bce=F.binary_cross_entropy_with_logits(g,d.float(),reduction='none')
    w=quality[mask].detach().float()
    return (w*(bce+nll)).mean()/3,(w*bce).mean(),(w*nll).mean()


def grid_probability(raw,parameters,knots,edges):
    weight=raw[:,0].softmax(-1);mu=raw[:,1];std=F.softplus(raw[:,2])+.025
    mapped,_=transform(edges[None,:].expand(len(raw),-1),parameters,knots)
    z=(mapped[:,:,None]-mu[:,None,:])/std[:,None,:]
    cdf=.5*(1+torch.erf(z/np.sqrt(2)))
    p=((cdf[:,1:]-cdf[:,:-1]).clamp_min(0)*weight[:,None,:]).sum(-1)
    return p/p.sum(1,keepdim=True).clamp_min(1e-12)


class Evaluator(OriginalEvaluator):
    @torch.no_grad()
    def many(self,model,modes):
        from multiscale_sequence import window_indices
        model.eval();d=self.data;modes=tuple(modes)
        predictions={m:np.full(len(d['validation']),np.nan,np.float64) for m in modes}
        coverage=np.zeros(len(d['validation']),np.int8)
        for begin in range(0,len(self.centers),32):
            centers=torch.tensor(self.centers[begin:begin+32],device='cuda')
            anchors=torch.tensor(self.anchors[begin:begin+32],device='cuda')
            ix,valid=window_indices(centers,d['lower'],d['upper'],anchors=anchors)
            relative=torch.tanh((d['depth'][ix]-d['depth'][anchors,None])/10)
            outputs=model(d['x'][ix],relative,valid,absolute_depth=d['absolute'][ix],factors=modes)
            rows=d['raw_to_valid'][ix[:,64:192].cpu().numpy()]
            use=valid[:,64:192].cpu().numpy()&(rows>=0);loc=self.local[rows[use]]
            assert (loc>=0).all() and (coverage[loc]==0).all() and len(np.unique(loc))==len(loc)
            coverage[loc]+=1;take=torch.as_tensor(use,device='cuda')
            for m,(gate,raw,params,knots) in outputs.items():
                g=gate[:,64:192][take][:,0].float().sigmoid()
                r=raw[:,64:192][take][:,0].float();q=params[:,64:192][take]
                p=grid_probability(r,q,knots,self.edges)
                mass=torch.cat((g[:,None],(1-g[:,None])*p),-1)
                predictions[m][loc]=self.actions[(mass@self.utilities).argmax(-1).cpu().numpy()]
        assert (coverage==1).all()
        result={}
        for m,pred in predictions.items():
            assert np.isfinite(pred).all();combined=self.reference.copy();combined[:,2]=pred
            s=base.metric(d['truth'][d['validation']],combined)
            result[m]=(pred,float(s['SW']),float(s['TOTAL']))
        return result


@torch.no_grad()
def factor_scope(model,data,mode):
    from multiscale_sequence import window_indices
    anchors=data['train_raw'][torch.tensor([0,len(data['train_raw'])//2],device='cuda')]
    ix,valid=window_indices(anchors,data['lower'],data['upper'])
    rel=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
    h,parent,ctx,_=model.extract(data['x'][ix],rel,valid,data['absolute'][ix])
    g,r,q,knots=model.heads[mode](h,parent,ctx,data['x'][ix,...,206:],valid)
    assert torch.equal(g,parent[0]) and torch.equal(r,parent[1])
    raw_y=data['labels'][ix][valid][:,2]
    mapped,lj=transform(raw_y[:,None],q[valid],knots)
    assert bool(torch.isfinite(mapped).all()) and bool(torch.isfinite(lj).all())
    return dict(default_logit_exact=True,all_original_mixture_parameters_exact=True,
        derivative_min=float(lj.exp().min()),derivative_max=float(lj.exp().max()),
        max_abs_transformed_shift=float((mapped[:,0]-raw_y).abs().max()),
        knots=knots.cpu().tolist(),scope='two fixed training windows; input-only-conditioned transform')


def mechanism_probe(model,data,mode):
    head=model.heads[mode];k=head.knots
    z=torch.stack([(k[:-1]+k[1:])/2,(k[:-1]+k[1:])/2]).detach().requires_grad_(True)
    n=2 if mode=='affine' else 3
    q=torch.linspace(-.3,.3,2*n,device='cuda').reshape(2,n)
    mapped,lj=transform(z,q,k)
    auto=torch.autograd.grad(mapped.sum(),z)[0]
    error=float((auto-lj.exp()).abs().max())
    assert error<1e-5 and float(lj.exp().min())>0
    zero=torch.zeros_like(q);identity,logj=transform(z.detach(),zero,k)
    assert torch.equal(identity,z.detach()) and bool((logj==0).all())
    edge=torch.tensor([[-float('inf'),float('inf')]],device='cuda').expand(2,-1)
    ends,_=transform(edge,q,k);assert torch.equal(ends,edge)
    return dict(analytical_jacobian_vs_autograd_max_error=error,positive_derivative=True,identity_exact=True,
        infinite_edges_preserved=True,knot_tails='identity for pwl4, globally affine for affine',
        density_jacobian_in_objective=True,scope='integrated necessary transform identity/Jacobian check; no parameter update')
