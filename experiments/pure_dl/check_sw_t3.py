"""CPU-only independent family/mass/tail/entropy and training-contract checks."""
import json,math
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t,norm
from scipy.special import logsumexp
from por_sw_t3_loss import objective,components,point_statistics,probability
from por_t3_patch_loss import components as old_components
from global_cosine_sequence import Network
from fold5_screening_802 import self_test

def main():
    torch.set_num_threads(2);torch.manual_seed(1345);self_test()
    gate=torch.randn(2,16,3,requires_grad=True);raw=torch.randn(2,16,3,3,8,requires_grad=True)
    y=torch.randn(2,16,3);default=torch.rand(2,16,3)>.7;mask=torch.ones(2,16,dtype=torch.bool);mask[:,0]=False
    q=torch.rand(2,16,requires_grad=True)
    rr=raw[mask].float();target=y[mask];d=default[mask]
    sd=torch.nn.functional.softplus(rr[:,:,2])+.025
    mu=rr[:,:,1].detach().numpy();std=sd.detach().numpy();yy=target.numpy();lw=rr[:,:,0].detach().log_softmax(-1).numpy();w=np.exp(lw)
    independent=[]
    for j in range(3):
        logp=t.logpdf(yy[:,j,None],3,loc=mu[:,j],scale=std[:,j]/np.sqrt(3)) if j in(0,2) else norm.logpdf(yy[:,j,None],loc=mu[:,j],scale=std[:,j])
        independent.append(-logsumexp(lw[:,j]+logp,axis=-1))
    independent=np.stack(independent,1)
    old=old_components(rr,target);new=components(rr,target)
    assert torch.equal(new[:,:2],old[:,:2]),'POR/PERM numerical functions must be unchanged'
    new_nll=-(rr[:,:,0].log_softmax(-1)+new).logsumexp(-1)
    nll_error=float(np.max(abs(independent-new_nll.detach().numpy())));assert nll_error<2e-6
    bce=torch.nn.functional.binary_cross_entropy_with_logits(gate[mask],d.float(),reduction='none').detach().numpy()
    expected=(q[mask].detach().numpy()*(bce+np.where(d.numpy(),0,independent)).mean(-1)).mean()
    loss=objective((gate,raw),y,default,mask,weights=q,epoch=0)
    assert abs(float(loss.detach())-expected)<1e-6
    changed=y.clone();changed[~mask]=1e6
    assert torch.equal(loss,objective((gate,raw),changed,default,mask,weights=q,epoch=0))
    assert torch.allclose(objective((gate,raw),y,default,mask,weights=q*.2,epoch=0),loss*.2,atol=1e-7)
    loss.backward();assert q.grad is None and (raw.grad[default]==0).all() and (raw.grad[~mask]==0).all()
    assert torch.isfinite(raw.grad).all() and raw.grad[~default].abs().sum()>0
    edges=np.r_[-np.inf,np.linspace(-5,5,255),np.inf]
    stats,cdf=point_statistics(gate[mask],rr,target,d,[edges]*3)
    mass_errors=[];tail_errors=[];entropy_errors=[]
    for j in range(3):
        standardized=(edges[None,:,None]-mu[:,j,None,:])/std[:,j,None,:]
        fc=t.cdf(standardized*np.sqrt(3),3) if j in(0,2) else norm.cdf(standardized)
        expected_mass=(np.diff(fc,axis=1)*w[:,j,None,:]).sum(-1)
        expected_mass/=expected_mass.sum(-1,keepdims=True)
        actual=probability(j,torch.tensor(w[:,j]),torch.tensor(mu[:,j]),torch.tensor(std[:,j]),torch.tensor(edges,dtype=torch.float32)).detach().numpy()
        error=float(np.max(abs(actual-expected_mass)));assert error<1e-6;mass_errors.append(error)
        z=(yy[:,j,None]-mu[:,j])/std[:,j]
        true_cdf=(w[:,j]*(t.cdf(z*np.sqrt(3),3) if j in(0,2) else norm.cdf(z))).sum(-1)
        tails=2*np.minimum(true_cdf,1-true_cdf);use=~d[:,j].numpy()
        err=float(np.max(abs(tails[use]-stats['target_tail'][:,j].detach().numpy()[use])));assert err<6e-7;tail_errors.append(err)
        entropy=-(expected_mass*np.log(np.maximum(expected_mass,1e-30))).sum(-1)/np.log(256)
        err=float(np.max(abs(entropy-stats['conditional_entropy'][:,j].detach().numpy())));assert err<2e-6;entropy_errors.append(err)
        assert np.isfinite(actual).all() and (actual>=0).all() and np.max(abs(actual.sum(-1)-1))<1e-6
    assert np.allclose(t.var(3,scale=std/np.sqrt(3)),std**2,rtol=2e-7)
    assert torch.isnan(stats['nll'][d]).all() and torch.isnan(stats['target_tail'][d]).all()
    assert torch.allclose(stats['nll'][~d],new_nll[~d],atol=1e-6)
    assert torch.isfinite(cdf).all() and torch.allclose(cdf[:,:,-1],torch.ones_like(cdf[:,:,-1]),atol=1e-6)
    # Explicit zero/large target distances and positive small/large scales.
    for j in range(3):
        pm=probability(j,torch.ones(2,1),torch.zeros(2,1),torch.tensor([[.025],[30.]]),torch.tensor([-float('inf'),-1e6,0.,1e6,float('inf')]))
        assert torch.isfinite(pm).all() and (pm>=0).all() and torch.allclose(pm.sum(-1),torch.ones(2),atol=1e-6)
    model=Network();x=torch.randn(2,256,206);v=torch.ones(2,256,dtype=torch.bool);depth=torch.zeros(2,256)
    outputs=model(x,depth,v,True,absolute_depth=depth)
    m=v.clone();m[:,:64]=False;m[:,192:]=False
    l=objective(outputs,torch.randn(2,256,3),torch.rand(2,256,3)>.5,m,True,torch.ones_like(depth),epoch=20)
    l.backward();assert torch.isfinite(l) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    result={'passed':True,'scope':'CPU only, no GPU or training','parameters':sum(p.numel() for p in model.parameters()),'scipy_nll_error':nll_error,'scipy_mass_errors_POR_PERM_SW':mass_errors,'scipy_tail_errors_POR_PERM_SW':tail_errors,'scipy_entropy_errors_POR_PERM_SW':entropy_errors,'POR_PERM_exact_old_component_values':True,'variance_matched_std':True,'boundary_scales_finite':True,'default_mask_invalid_mask_and_detached_finalQ':True,'full_model_CPU_backward':True,'strict_threshold':80.2}
    out=Path('outputs/pure_dl_sw_t3_sequence');out.mkdir(exist_ok=True);(out/'cpu_preflight.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
