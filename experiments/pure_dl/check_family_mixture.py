"""CPU-only shared-family numerical and training-contract audit."""
import json,math
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t,norm
from scipy.special import logsumexp
from family_mixture_sequence import Network,expand_families
from global_cosine_sequence import Network as Baseline
from family_mixture_loss import components,objective,probability,point_statistics
from train_invariant import conditional_probability
from fold5_screening_802 import self_test

def main():
    torch.set_num_threads(2);self_test();torch.manual_seed(277)
    original=torch.randn(2,12,3,3,8,requires_grad=True)
    family=torch.randn(2,12,1,requires_grad=True)
    gate=torch.randn(2,12,3,requires_grad=True)
    raw=expand_families(original,family)
    y=torch.randn(2,12,3);default=torch.rand(2,12,3)>.65
    mask=torch.ones(2,12,dtype=torch.bool);mask[:,0]=False
    q=torch.rand(2,12,requires_grad=True)
    rr=raw[mask];yy=y[mask];dd=default[mask]
    sd=torch.nn.functional.softplus(rr[:,:,2])+.025
    mu=rr[:,:,1].detach().numpy();std=sd.detach().numpy();lw=rr[:,:,0].detach().log_softmax(-1).numpy();w=np.exp(lw);truth=yy.numpy()
    nll=[]
    for j in range(3):
        logp=norm.logpdf(truth[:,j,None],loc=mu[:,j],scale=std[:,j])
        if j==0:logp[:,:8]=t.logpdf(truth[:,0,None],3,loc=mu[:,0,:8],scale=std[:,0,:8]/np.sqrt(3))
        nll.append(-logsumexp(lw[:,j]+logp,axis=-1))
    nll=np.stack(nll,1)
    actual=-(rr[:,:,0].log_softmax(-1)+components(rr,yy)).logsumexp(-1)
    nll_error=float(np.max(abs(nll-actual.detach().numpy())));assert nll_error<2e-6
    bce=torch.nn.functional.binary_cross_entropy_with_logits(gate[mask],dd.float(),reduction='none').detach().numpy()
    expected=(q[mask].detach().numpy()*(bce+np.where(dd.numpy(),0,nll)).mean(-1)).mean()
    loss=objective((gate,raw),y,default,mask,weights=q,epoch=0)
    assert abs(float(loss.detach())-expected)<1e-6
    assert torch.allclose(objective((gate,raw),y,default,mask,weights=q*.2,epoch=0),loss*.2,atol=1e-7)
    changed=y.clone();changed[~mask]=1e6
    assert torch.equal(loss,objective((gate,raw),changed,default,mask,weights=q,epoch=0))
    loss.backward()
    assert q.grad is None and (original.grad[default]==0).all() and (original.grad[~mask]==0).all()
    assert (family.grad[default[:,:,0]]==0).all() and (family.grad[~mask]==0).all()
    assert family.grad.abs().sum()>0 and torch.isfinite(family.grad).all()
    edges=np.r_[-np.inf,np.linspace(-6,6,255),np.inf]
    with torch.no_grad():stats,cdf=point_statistics(gate[mask],rr,yy,dd,[edges]*3)
    masses=[];tails=[];entropies=[]
    for j in range(3):
        z=(edges[None,:,None]-mu[:,j,None,:])/std[:,j,None,:]
        fc=norm.cdf(z)
        if j==0:fc[:,:,:8]=t.cdf(z[:,:,:8]*np.sqrt(3),3)
        expected_mass=(np.diff(fc,axis=1)*w[:,j,None,:]).sum(-1);expected_mass/=expected_mass.sum(-1,keepdims=True)
        value=probability(j,torch.tensor(w[:,j]),torch.tensor(mu[:,j]),torch.tensor(std[:,j]),edges).numpy()
        error=float(np.max(abs(value-expected_mass)));assert error<1e-6;masses.append(error)
        z=(truth[:,j,None]-mu[:,j])/std[:,j];fc=norm.cdf(z)
        if j==0:fc[:,:8]=t.cdf(z[:,:8]*np.sqrt(3),3)
        tc=(fc*w[:,j]).sum(-1);tail=2*np.minimum(tc,1-tc);use=~dd[:,j].numpy()
        error=float(np.max(abs(tail[use]-stats['target_tail'][:,j].numpy()[use])));assert error<6e-7;tails.append(error)
        entropy=-(expected_mass*np.log(np.maximum(expected_mass,1e-30))).sum(-1)/np.log(256)
        error=float(np.max(abs(entropy-stats['conditional_entropy'][:,j].numpy())));assert error<2e-6;entropies.append(error)
        assert np.isfinite(value).all() and (value>=0).all() and np.max(abs(value.sum(-1)-1))<1e-6
    assert np.allclose(stats['por_family_t3'].numpy(),torch.sigmoid(family.detach()[mask][:,0]).numpy(),atol=2e-7)
    assert torch.isnan(stats['nll'][dd]).all() and torch.isnan(stats['target_tail'][dd]).all()
    assert torch.allclose(stats['nll'][~dd],actual.detach()[~dd],atol=1e-6)
    assert torch.isfinite(cdf).all() and torch.allclose(cdf[:,:,-1],torch.ones_like(cdf[:,:,-1]),atol=1e-6)
    limits=[]
    for logit in [-60.,60.]:
        expanded=expand_families(original.detach(),torch.full_like(family,logit))[mask]
        weights=expanded[:,:,0].softmax(-1);m=expanded[:,:,1];s=torch.nn.functional.softplus(expanded[:,:,2])+.025
        actual=probability(0,weights[:,0],m[:,0],s[:,0],edges).numpy()
        base_w=original.detach()[mask][:,0,0].softmax(-1).numpy()
        z=(edges[None,:,None]-m[:,0,None,:8].numpy())/s[:,0,None,:8].numpy()
        fc=t.cdf(z*np.sqrt(3),3) if logit>0 else norm.cdf(z)
        expected=(np.diff(fc,axis=1)*base_w[:,None]).sum(-1)
        error=float(np.max(abs(actual-expected)));assert error<1e-6;limits.append(error)
        for j in [1,2]:
            base_raw=original.detach()[mask]
            base_sd=torch.nn.functional.softplus(base_raw[:,:,2])+.025
            expected=conditional_probability(base_raw[:,j,0].softmax(-1),base_raw[:,j,1],base_sd[:,j],torch.tensor(edges,dtype=torch.float32))
            assert torch.allclose(probability(j,weights[:,j],m[:,j],s[:,j],edges),expected,atol=2e-7)
    torch.manual_seed(91);a=Baseline().eval();rng=torch.get_rng_state()
    torch.manual_seed(91);b=Network().eval();assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert sum(p.numel() for p in b.parameters())==661856
    x=torch.randn(2,256,206);depth=torch.zeros(2,256);valid=torch.ones(2,256,dtype=torch.bool);valid[:,:19]=False
    with torch.no_grad():
        aa=a(x,depth,valid,True,absolute_depth=depth);bb=b(x,depth,valid,True,absolute_depth=depth)
        for (ag,ar),(bg,br) in zip(aa,bb):
            assert torch.equal(ag,bg)
            assert torch.equal(br[...,1,:8],ar[...,1,:]) and torch.equal(br[...,1,8:],ar[...,1,:])
            assert torch.equal(br[...,2,:8],ar[...,2,:]) and torch.equal(br[...,2,8:],ar[...,2,:])
            assert torch.allclose(br[...,0,:8].softmax(-1),ar[...,0,:].softmax(-1),atol=1e-7)
            assert torch.allclose(br[...,0,:].softmax(-1)[...,:8].sum(-1),torch.full_like(bg,.5),atol=1e-7)
        for head in [b.main_head,b.aux_mid,b.aux_coarse]:head.family_gate.weight.normal_(0,.1)
        active=b(x,depth,valid,True,absolute_depth=depth)
        changed=x.clone();changed[~valid]+=1000
        padded=b(changed,depth,valid,True,absolute_depth=depth)
        assert all(torch.equal(t,u) for xx,zz in zip(active,padded) for t,u in zip(xx,zz))
        changed=x.clone();changed[1]+=100
        other=b(changed,depth,valid,True,absolute_depth=depth)
        assert all(torch.equal(t[0],u[0]) for xx,zz in zip(active,other) for t,u in zip(xx,zz))
    b.train();outputs=b(x,depth,valid,True,absolute_depth=depth);use=valid.clone();use[:,:64]=False;use[:,192:]=False
    model_loss=objective(outputs,torch.randn(2,256,3),torch.rand(2,256,3)>.5,use,True,torch.ones_like(depth),epoch=20)
    model_loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in b.parameters())
    assert all(head.family_gate.weight.grad.abs().sum()>0 for head in [b.main_head,b.aux_mid,b.aux_coarse])
    result={'passed':True,'scope':'CPU only; no GPU/training or accuracy result','parameters':661856,'added_parameters':355,'scipy_mixture_NLL_error':nll_error,'scipy_mass_errors':masses,'scipy_tail_errors':tails,'scipy_entropy_errors':entropies,'rho0_Gaussian_rho1_t3_limit_errors':limits,'nonPOR_distribution_unchanged':True,'old_parameters_and_RNG_preserved':True,'initial_rho_half_and_shared_mu_std_verified':True,'initial_POR_distribution_changed':True,'gate_gradients_mask_Q_padding_well_checks':True,'all_three_family_heads_CPU_backward':True,'strict_threshold':80.2}
    out=Path('outputs/pure_dl_family_mixture_sequence');out.mkdir(exist_ok=True);(out/'cpu_preflight.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
