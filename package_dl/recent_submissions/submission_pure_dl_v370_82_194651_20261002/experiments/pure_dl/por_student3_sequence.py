"""POR-only fixed-df=3 Student-t mixture; Gaussian PERM/SW and original heads.
The supplied std remains a standard deviation: Student-t scale=std/sqrt(3).
This changes the POR density family, not anomaly labels or observation weights.
"""
import math
import json
import torch
from torch.nn import functional as F
from relative_only_sequence import Network

_T_CONSTANT=math.lgamma(2)-math.lgamma(1.5)-.5*math.log(3*math.pi)

def head_loss(output,target,default,mask):
    gate,raw=output
    assert mask.shape==target.shape[:2] and default.shape==target.shape and bool(mask.any())
    if gate.shape[1]!=target.shape[1]:
        repeat=target.shape[1]//gate.shape[1];assert repeat*gate.shape[1]==target.shape[1]
        gate=gate.repeat_interleave(repeat,1);raw=raw.repeat_interleave(repeat,1)
    gate=gate[mask].float();raw=raw[mask].float();d=default[mask].bool()
    y=target[mask].float().masked_fill(d,0)
    logw=raw[:,:,0].log_softmax(-1);mu=raw[:,:,1];std=F.softplus(raw[:,:,2])+.025
    gaussian=-.5*((y[...,None]-mu)/std).square()-std.log()-.5*math.log(2*math.pi)
    scale=std[:,0]/math.sqrt(3);z=(y[:,0,None]-mu[:,0])/scale
    por=_T_CONSTANT-scale.log()-2*torch.log1p(z.square()/3)
    log_density=torch.cat([por[:,None,:],gaussian[:,1:,:]],1)
    nll=-(logw+log_density).logsumexp(-1)
    return F.binary_cross_entropy_with_logits(gate,d.float())+nll.masked_fill(d,0).mean()

def objective(outputs,target,default,mask,deep_supervision=False):
    if not deep_supervision:return head_loss(outputs,target,default,mask)
    main,mid,coarse=outputs
    return head_loss(main,target,default,mask)+.2*head_loss(mid,target,default,mask)+.1*head_loss(coarse,target,default,mask)

@torch.no_grad()
def por_probability(weight,mu,std,edges):
    """Mixture masses of intervals between edges; std is NOT Student-t scale."""
    weight=weight.float();mu=mu.float();std=std.float()
    edges=torch.as_tensor(edges,device=mu.device,dtype=torch.float32)
    assert weight.shape==mu.shape==std.shape and edges.ndim==1
    assert bool((std>0).all()) and bool(torch.isfinite(std).all())
    # z/sqrt(3)=(edge-mu)/std when Student-t scale=std/sqrt(3).
    phi=torch.atan((edges[None,:,None]-mu[:,None,:])/std[:,None,:])
    cdf=.5+(phi+torch.sin(phi)*torch.cos(phi))/math.pi
    cdf=torch.where(torch.isneginf(edges)[None,:,None],torch.zeros_like(cdf),cdf)
    cdf=torch.where(torch.isposinf(edges)[None,:,None],torch.ones_like(cdf),cdf)
    probability=(cdf.diff(dim=1)*weight[:,None,:]).sum(-1).clamp_min(0)
    return probability/probability.sum(-1,keepdim=True).clamp_min(1e-12)

def self_test():
    import numpy as np
    from scipy.stats import t
    torch.set_num_threads(2);torch.manual_seed(913)
    gate=torch.randn(2,8,3,requires_grad=True);raw=torch.randn(2,8,3,3,8,requires_grad=True)
    y=torch.randn(2,8,3);default=torch.rand(2,8,3)>.4;mask=torch.ones(2,8,dtype=torch.bool);mask[:,0]=False
    loss=objective((gate,raw),y,default,mask)
    rr=raw[mask];yy=y[mask];dd=default[mask];std=F.softplus(rr[:,:,2])+.025
    components=[torch.distributions.StudentT(3.,rr[:,0,1],std[:,0]/math.sqrt(3)),torch.distributions.Normal(rr[:,1,1],std[:,1]),torch.distributions.Normal(rr[:,2,1],std[:,2])]
    nll=torch.stack([-torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=rr[:,j,0]),components[j]).log_prob(yy[:,j]) for j in range(3)],1)
    independent=F.binary_cross_entropy_with_logits(gate[mask],dd.float())+nll.masked_fill(dd,0).mean()
    assert torch.allclose(loss,independent,atol=1e-6,rtol=1e-6)
    loss.backward();assert torch.isfinite(raw.grad).all() and bool((raw.grad[default]==0).all()) and bool((raw.grad[~mask]==0).all())
    altered=y.clone();altered[default|~mask[:,:,None]]=float('nan');assert torch.equal(loss,objective((gate,raw),altered,default,mask))
    all_default=torch.ones_like(default);assert torch.equal(objective((gate,raw),y,all_default,mask),F.binary_cross_entropy_with_logits(gate[mask],all_default[mask].float()))
    edges=torch.tensor([-float('inf'),-100.,-10.,-1.,0.,1.,10.,100.,float('inf')]);w=rr[:,0,0].detach().softmax(-1);mu=rr[:,0,1].detach();sd=std[:,0].detach()
    actual=por_probability(w,mu,sd,edges).numpy()
    cdf=t.cdf((edges.numpy()[None,:,None]-mu.numpy()[:,None,:])/(sd.numpy()[:,None,:]/np.sqrt(3)),df=3)
    expected=(np.diff(cdf,axis=1)*w.numpy()[:,None,:]).sum(-1);expected/=expected.sum(-1,keepdims=True)
    error=float(np.max(abs(actual-expected)));assert error<2e-7 and np.max(abs(actual.sum(-1)-1))<2e-7
    outputs=((gate,raw),(gate[:,::2],raw[:,::2]),(gate[:,::4],raw[:,::4]))
    aux=loss+.2*objective((gate[:,::2].repeat_interleave(2,1),raw[:,::2].repeat_interleave(2,1)),y,default,mask)+.1*objective((gate[:,::4].repeat_interleave(4,1),raw[:,::4].repeat_interleave(4,1)),y,default,mask)
    assert torch.equal(objective(outputs,y,default,mask,True),aux)
    return {'passed':True,'independent_studentT_and_gaussian_mixture_nll':True,'scipy_t_cdf_max_error':error,'infinite_edges_and_tails':True,'default_and_padding_density_gradients_zero':True,'default_targets_can_be_nan':True,'all_default_BCE_only':True,'auxiliary_repeat_and_weights':True,'POR_only_fixed_df':3,'std_to_scale_factor':1/math.sqrt(3)}
if __name__=='__main__':print(json.dumps(self_test(),indent=2))
