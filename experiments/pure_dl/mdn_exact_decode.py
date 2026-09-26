"""Label-independent analytic official-score utilities for Gaussian MDN outputs.

PERM is exact in log10 coordinates. POR/SW omit the official denominator's
1e-5 only in the continuous lognormal integral; the default atom keeps the exact
official kernel. No truth is consulted while selecting actions.
"""
from pathlib import Path
import argparse, json, sys, time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'package'))
from src.utils import DEFAULTS, BOUNDS, kernel_matrix

TARGETS = ['POR','PERM','SW']
SQRT_2PI = np.sqrt(2*np.pi)


def _phi(x):
    return torch.exp(-.5*x*x)/SQRT_2PI


@torch.no_grad()
def mixture_utility(actions, weight, mean, std, target, support_bounds=None):
    """Return (N,A) conditional utility, given (N,K) mixture parameters.

    mean/std MUST be in physical transformed coordinates: ln(POR),
    log10(PERM), or ln(SW), after undoing training robust normalization.
    Input tensors determine device/dtype; float64 provides reference precision.
    With raw-space support_bounds=(lo,hi), mass outside the interval is placed
    at its endpoints, matching the original quadrature grid's infinite tails.
    """
    actions=torch.as_tensor(actions,device=mean.device,dtype=mean.dtype)
    assert target in TARGETS and actions.ndim==1
    assert weight.shape==mean.shape==std.shape and mean.ndim==2
    assert torch.all(std>0) and torch.all(actions>0)
    w=weight/weight.sum(1,keepdim=True).clamp_min(1e-20)
    mu=mean[:,None,:];sd=std[:,None,:]
    a=actions[None,:,None]
    if target=='PERM':
        c=torch.log10(a)
        lo,hi=(-np.inf,np.inf) if support_bounds is None else np.log10(support_bounds)
        ll=torch.clamp(c-1,min=float(lo));lh=torch.clamp(c,max=float(hi))
        rl=torch.clamp(c,min=float(lo));rh=torch.clamp(c+1,max=float(hi))
        al=(ll-mu)/sd;bl=(lh-mu)/sd;ar=(rl-mu)/sd;br=(rh-mu)/sd
        pl=torch.special.ndtr(bl)-torch.special.ndtr(al)
        pr=torch.special.ndtr(br)-torch.special.ndtr(ar)
        left=(1-c+mu)*pl+sd*(_phi(al)-_phi(bl))
        right=(1+c-mu)*pr-sd*(_phi(ar)-_phi(br))
        left=torch.where(lh>ll,left,0);right=torch.where(rh>rl,right,0)
        result=left+right
    else:
        r=.08 if target=='POR' else .05
        c=torch.log(a)
        lo,hi=(-np.inf,np.inf) if support_bounds is None else np.log(support_bounds)
        ll=torch.clamp(c-np.log1p(r),min=float(lo));lh=torch.clamp(c,max=float(hi))
        rl=torch.clamp(c,min=float(lo));rh=torch.clamp(c-np.log1p(-r),max=float(hi))
        al=(ll-mu)/sd;bl=(lh-mu)/sd;ar=(rl-mu)/sd;br=(rh-mu)/sd
        pl=torch.special.ndtr(bl)-torch.special.ndtr(al)
        pr=torch.special.ndtr(br)-torch.special.ndtr(ar)
        # e^-z * Normal(z;mu,sd) has tilted mean mu-sd^2.
        ml=torch.special.ndtr(bl+sd)-torch.special.ndtr(al+sd)
        mr=torch.special.ndtr(br+sd)-torch.special.ndtr(ar+sd)
        am=a*torch.exp(-mu+.5*sd*sd)
        left=(1+1/r)*pl-am/r*ml;right=(1-1/r)*pr+am/r*mr
        result=torch.where(lh>ll,left,0)+torch.where(rh>rl,right,0)
    if support_bounds is not None:
        below=torch.special.ndtr((float(lo)-mu)/sd)
        above=torch.special.ndtr((mu-float(hi))/sd)
        if target=='PERM':
            kl=(1-abs(c-float(lo))).clamp_min(0)
            kh=(1-abs(c-float(hi))).clamp_min(0)
        else:
            kl=(1-abs(a-float(support_bounds[0]))/(r*(abs(float(support_bounds[0]))+1e-5))).clamp_min(0)
            kh=(1-abs(a-float(support_bounds[1]))/(r*(abs(float(support_bounds[1]))+1e-5))).clamp_min(0)
        result=result+below*kl+above*kh
    return (result.clamp(0,1)*w[:,None,:]).sum(-1)


def restore_parameters(pack, target_index, *, device='cpu', dtype=torch.float32):
    """Restore saved normalized float16 mixture parameters to target coordinates."""
    scale=float(pack['regression_scale'][target_index])
    med=float(pack['regression_median'][target_index])
    w=torch.as_tensor(pack['mixture_weight'][:,target_index].astype(np.float32),device=device,dtype=dtype)
    mean=torch.as_tensor(pack['mixture_mu'][:,target_index].astype(np.float32),device=device,dtype=dtype)*scale+med
    std=torch.as_tensor(pack['mixture_std'][:,target_index].astype(np.float32),device=device,dtype=dtype)*scale
    return w,mean,std


def self_test():
    """Only synthetic distributions; integrate the exact same zero-eps kernels."""
    torch.set_num_threads(3)
    weights=np.array([[.2,.5,.3],[.75,.2,.05],[.1,.1,.8]])
    records=[]
    for target in TARGETS:
        if target=='PERM':
            means=np.array([[-.5,1.,2.],[0.,1.6,2.8],[-1.,.8,2.5]])
            stds=np.array([[.12,.3,.7],[.4,.15,.6],[.2,.7,.35]])
            actions=np.geomspace(.01,1500,47)
        else:
            physical=np.array([[8.,13.,22.],[10.,15.,18.],[7.,20.,30.]]) if target=='POR' else np.array([[35.,60.,85.],[50.,65.,80.],[40.,75.,95.]])
            means=np.log(physical)
            stds=np.array([[.015,.05,.17],[.08,.025,.15],[.3,.08,.018]])
            actions=np.geomspace(2.,40.,47) if target=='POR' else np.linspace(8.305,99.9,47)
        exact=mixture_utility(torch.tensor(actions),torch.tensor(weights),torch.tensor(means),torch.tensor(stds),target).numpy()
        calc32=mixture_utility(torch.tensor(actions,dtype=torch.float32),torch.tensor(weights,dtype=torch.float32),torch.tensor(means,dtype=torch.float32),torch.tensor(stds,dtype=torch.float32),target).numpy()
        normal_grid=np.linspace(-9,9,60001)
        normal_pdf=np.exp(-.5*normal_grid**2)/SQRT_2PI
        reference=np.zeros_like(exact)
        for row in range(len(means)):
            for comp in range(means.shape[1]):
                z=means[row,comp]+stds[row,comp]*normal_grid
                raw=10**z if target=='PERM' else np.exp(z)
                if target=='PERM':
                    kernel=np.maximum(0,1-abs(np.log10(actions[:,None])-z[None,:]))
                else:
                    r=.08 if target=='POR' else .05
                    kernel=np.maximum(0,1-abs(actions[:,None]-raw[None,:])/(r*raw[None,:]))
                reference[row]+=weights[row,comp]*np.trapz(kernel*normal_pdf[None,:],normal_grid,axis=1)
        max_error=float(np.max(abs(exact-reference)))
        float32_error=float(np.max(abs(exact-calc32)))
        assert max_error<2e-6,(target,max_error)
        assert float32_error<5e-5,(target,float32_error)
        assert np.isfinite(exact).all() and np.all((exact>=0)&(exact<=1))
        bounds={'POR':(9.,20.),'PERM':(.15,130.),'SW':(45.,83.)}[target]
        clipped=mixture_utility(torch.tensor(actions),torch.tensor(weights),torch.tensor(means),torch.tensor(stds),target,support_bounds=bounds).numpy()
        clip_reference=np.zeros_like(clipped)
        for row in range(len(means)):
            for comp in range(means.shape[1]):
                z=means[row,comp]+stds[row,comp]*normal_grid
                unbounded=10**z if target=='PERM' else np.exp(z)
                raw=np.clip(unbounded,*bounds)
                if target=='PERM':
                    kernel=np.maximum(0,1-abs(np.log10(actions[:,None])-np.log10(raw[None,:])))
                else:
                    r=.08 if target=='POR' else .05
                    epsilon=np.where((unbounded<bounds[0])|(unbounded>bounds[1]),1e-5,0)
                    kernel=np.maximum(0,1-abs(actions[:,None]-raw[None,:])/(r*(raw+epsilon)[None,:]))
                clip_reference[row]+=weights[row,comp]*np.trapz(kernel*normal_pdf[None,:],normal_grid,axis=1)
        clip_error=float(np.max(abs(clipped-clip_reference)))
        assert clip_error<2e-6,(target,'clipped',clip_error)
        records.append({'target':target,'max_abs_error_vs_60001_point_integral':max_error,'max_float32_error':float32_error,
                        'clipped_support_max_abs_error':clip_error})
    result={'passed':True,'fixtures':'synthetic only; 3 mixtures x 3 components x 47 actions per target',
            'results':records,'continuous_relative_kernel_epsilon':'1e-5 omitted analytically; default atom retains exact epsilon'}
    print(json.dumps(result,indent=2),flush=True)
    return result


def run_decode(source, destination, *, device='cuda', batch=128, actions=800):
    # The decode loops do not read truth. Truth is read only after every action
    # has been fixed, for complete precommitted reporting and row validation.
    from train import metric, write
    torch.set_num_threads(3)
    destination.mkdir(parents=True,exist_ok=True)
    source_config=json.loads((source/'config.json').read_text())
    start=time.perf_counter()
    config={**source_config,'out':str(destination),'decode':'analytic continuous MDN expected official utility; fixed 800 action grid; default atom exact',
            'source':str(source),'continuous_epsilon_approximation':'relative continuous denominator omits 1e-5',
            'parameter_archive_precision':'float16 saved mixture/gate arrays', 'action_count':actions,
            'support_handling':'same as original 128 quadrature: mass below/above outer-training nondefault min/max is placed at endpoints'}
    assembled=[]
    timing=[]
    for fold in range(1,6):
        pack=np.load(source/f'fold_{fold}_distributions.npz')
        indices=pack['indices'];prediction=np.empty((len(indices),3),np.float64)
        gates=pack['gate'].astype(np.float32)
        for j,target in enumerate(TARGETS):
            support=pack[f'{target}_centers']
            grid=np.linspace(support[0],support[-1],actions) if j==2 else np.geomspace(max(support[0]*.9,.001),support[-1]*1.1,actions)
            grid=np.unique(np.r_[DEFAULTS[target],np.clip(grid,*BOUNDS[target])])
            weight,mean,std=restore_parameters(pack,j,device=device)
            default_u=torch.as_tensor(kernel_matrix(grid,np.array([DEFAULTS[target]]),target)[:,0],device=device,dtype=torch.float32)
            for pos in range(0,len(indices),batch):
                end=min(pos+batch,len(indices))
                cu=mixture_utility(grid,weight[pos:end],mean[pos:end],std[pos:end],target,
                                   support_bounds=(support[0],support[-1]))
                gate=torch.as_tensor(gates[pos:end,j,None],device=device)
                full=gate*default_u+(1-gate)*cu
                assert torch.isfinite(full).all()
                best=full.argmax(1).cpu().numpy()
                prediction[pos:end,j]=grid[best]
        np.save(destination/f'fold_{fold}_pred.npy',prediction)
        assembled.append((indices,prediction))
        timing.append(time.perf_counter()-start)
        print(f'FOLD_DECODED {fold}/5 cumulative_seconds={timing[-1]:.2f}',flush=True)
    original=np.load(source/'oof.npz')
    source_report=json.loads((source/'report.json').read_text())
    pred=np.full_like(original['truth'],np.nan)
    for indices,prediction in assembled:pred[indices]=prediction
    assert np.isfinite(pred).all()
    truth=original['truth'];fold_ids=original['fold'];wells=original['well']
    fold_reports=[{'fold':f+1,'scores':metric(truth[fold_ids==f],pred[fold_ids==f])} for f in range(5)]
    report={'config':config,'oof_scores':metric(truth,pred),'folds':fold_reports,
            'seconds':time.perf_counter()-start,'original_128_grid_scores':source_report['oof_scores'],
            'score_delta':{k:metric(truth,pred)[k]-source_report['oof_scores'][k] for k in ['POR','PERM','SW','TOTAL']}}
    np.savez_compressed(destination/'oof.npz',truth=truth,pred=pred,fold=fold_ids,well=wells)
    write(destination/'config.json',config);write(destination/'report.json',report)
    print('FINAL',json.dumps(report),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--source',type=Path,default=ROOT/'outputs/pure_dl_residual')
    ap.add_argument('--out',type=Path,default=ROOT/'outputs/pure_dl_residual/analytic_800')
    ap.add_argument('--device',default='cuda');ap.add_argument('--batch',type=int,default=128)
    args=ap.parse_args()
    if args.self_test:self_test()
    else:run_decode(args.source,args.out,device=args.device,batch=args.batch)
