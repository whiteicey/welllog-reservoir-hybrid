"""Multiply each supervised row's complete loss by externally supplied quality weight.
The caller defines anomalous input channels and supplies w=0.5**k. This module
never infers weights from targets, prediction errors, or training dynamics.
Denominator is the original supervised row count, not the sum of weights.
"""
import math
import json
import torch
from torch.nn import functional as F
from relative_only_sequence import Network

def _head_loss(output,target,default,mask,row_weights):
    gate,raw=output
    assert mask.shape==target.shape[:2] and default.shape==target.shape and bool(mask.any())
    if gate.shape[1]!=target.shape[1]:
        repeat=target.shape[1]//gate.shape[1]
        assert repeat*gate.shape[1]==target.shape[1]
        gate=gate.repeat_interleave(repeat,1);raw=raw.repeat_interleave(repeat,1)
    gate=gate[mask].float();raw=raw[mask].float();d=default[mask].bool()
    # Select supervised rows first; exclude default labels before density arithmetic.
    y=target[mask].float().masked_fill(d,0)
    logw=raw[:,:,0].log_softmax(-1);mu=raw[:,:,1];sd=F.softplus(raw[:,:,2])+.025
    nll=-(logw-.5*((y[...,None]-mu)/sd).square()-sd.log()-.5*math.log(2*math.pi)).logsumexp(-1)
    per_row=F.binary_cross_entropy_with_logits(gate,d.float(),reduction='none').mean(-1)+nll.masked_fill(d,0).mean(-1)
    if row_weights is None:w=torch.ones_like(per_row)
    else:
        assert row_weights.shape==mask.shape
        w=row_weights[mask].detach().float()
        assert bool(torch.isfinite(w).all()) and bool(((w>0)&(w<=1)).all())
    return (per_row*w).mean()

def objective(outputs,target,default,mask,deep_supervision=False,row_weights=None):
    if not deep_supervision:return _head_loss(outputs,target,default,mask,row_weights)
    main,mid,coarse=outputs
    return (_head_loss(main,target,default,mask,row_weights)
            +.2*_head_loss(mid,target,default,mask,row_weights)
            +.1*_head_loss(coarse,target,default,mask,row_weights))

def self_test():
    torch.set_num_threads(2);torch.manual_seed(957)
    from population_likelihood_sequence import objective as reference
    gate=torch.randn(2,8,3,requires_grad=True);raw=torch.randn(2,8,3,3,8,requires_grad=True);y=torch.randn(2,8,3);d=torch.rand(2,8,3)>.45;mask=torch.ones(2,8,dtype=torch.bool);mask[:,0]=False
    ones=torch.ones(2,8,requires_grad=True);base=objective((gate,raw),y,d,mask,row_weights=ones)
    assert torch.allclose(base,reference((gate,raw),y,d,mask),atol=1e-6,rtol=1e-6)
    for k in range(4):assert torch.allclose(objective((gate,raw),y,d,mask,row_weights=ones*(.5**k)),base*(.5**k),atol=1e-6)
    weights=torch.tensor([[1.,.5,.25,.125,1.,.5,.25,.125]]).expand(2,-1).clone().requires_grad_()
    actual=objective((gate,raw),y,d,mask,row_weights=weights)
    rr=raw[mask];dist=torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=rr[:,:,0]),torch.distributions.Normal(rr[:,:,1],F.softplus(rr[:,:,2])+.025))
    bce=F.binary_cross_entropy_with_logits(gate[mask],d[mask].float(),reduction='none')
    independent=((bce-dist.log_prob(y[mask]).masked_fill(d[mask],0)).mean(-1)*weights[mask].detach()).sum()/int(mask.sum())
    assert torch.allclose(actual,independent,atol=1e-6)
    actual.backward();assert weights.grad is None and ones.grad is None
    assert bool((raw.grad[d]==0).all()) and bool((raw.grad[~mask]==0).all()) and bool((gate.grad[~mask]==0).all())
    altered=y.clone();altered[d|~mask[:,:,None]]=float('nan')
    assert torch.equal(actual,objective((gate,raw),altered,d,mask,row_weights=weights))
    default_all=torch.ones_like(d)
    all_default=objective((gate,raw),y,default_all,mask,row_weights=weights)
    expected_bce=(F.binary_cross_entropy_with_logits(gate[mask],torch.ones_like(gate[mask]),reduction='none').mean(-1)*weights[mask].detach()).mean()
    assert torch.equal(all_default,expected_bce)
    outputs=((gate,raw),(gate[:,::2],raw[:,::2]),(gate[:,::4],raw[:,::4]))
    total=objective(outputs,y,d,mask,True,weights)
    aux_expected=actual+.2*objective((gate[:,::2].repeat_interleave(2,1),raw[:,::2].repeat_interleave(2,1)),y,d,mask,row_weights=weights)+.1*objective((gate[:,::4].repeat_interleave(4,1),raw[:,::4].repeat_interleave(4,1)),y,d,mask,row_weights=weights)
    assert torch.equal(total,aux_expected)
    assert torch.allclose(objective(outputs,y,d,mask,True,ones),reference(outputs,y,d,mask,True),atol=1e-6)
    return {'passed':True,'independent_distribution_and_original_row_denominator':True,'ones_match_population_objective':True,'k_0_1_2_3_scaling':True,'default_and_invalid_density_grad_zero':True,'row_weight_detached':True,'default_gate_also_scaled':True,'same_auxiliary_row_weights':True,'no_anomaly_scope_inference':True}
if __name__=='__main__':print(json.dumps(self_test(),indent=2))
