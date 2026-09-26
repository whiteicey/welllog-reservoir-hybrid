import math
import numpy as np
import torch


def conditional_probability(weight, mean, std, edges):
    z = (edges[None, :, None]-mean[:, None, :])/std[:, None, :]
    cdf = .5*(1+torch.erf(z/np.sqrt(2)))
    prob = ((cdf[:, 1:]-cdf[:, :-1]).clamp_min(0)*weight[:, None, :]).sum(-1)
    return prob/prob.sum(1, keepdim=True).clamp_min(1e-12)

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

def probability(j,weight,mu,sd,edges):
    return por_probability(weight,mu,sd,edges) if j==0 else conditional_probability(weight,mu,sd,edges)
