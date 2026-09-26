"""Input-conditional POR t3/Gaussian family mixture; PERM/SW Gaussian unchanged."""
import math
import torch
from torch.nn import functional as F

def components(raw,target):
    assert raw.shape[-1]==16
    sd=F.softplus(raw[:,:,2])+.025;mu=raw[:,:,1]
    gaussian=-.5*((target[...,None]-mu)/sd).square()-sd.log()-.5*math.log(2*math.pi)
    scale=sd[:,0,:8]/math.sqrt(3)
    student=math.lgamma(2)-math.lgamma(1.5)-.5*math.log(3*math.pi)-scale.log()-2*torch.log1p(((target[:,0,None]-mu[:,0,:8])/scale).square()/3)
    por=torch.cat([student,gaussian[:,0,8:]],-1)
    return torch.cat([por[:,None],gaussian[:,1:]],1)


@torch.no_grad()
def probability(j,weight,mu,sd,edges):
    # Normalize ONCE after summing all16 weighted components. Never renormalize
    # the family branches separately, which would discard the learned rho.
    weight=weight.float();mu=mu.float();sd=sd.float()
    edges=torch.as_tensor(edges,device=mu.device,dtype=torch.float32)
    assert weight.shape==mu.shape==sd.shape and weight.shape[-1]==16
    z=(edges[None,:,None]-mu[:,None,:])/sd[:,None,:]
    cdf=torch.special.ndtr(z)
    if j==0:
        phi=torch.atan(z[:,:,:8])
        student=.5+(phi+torch.sin(phi)*torch.cos(phi))/math.pi
        cdf=torch.cat([student,cdf[:,:,8:]],-1)
    cdf=torch.where(torch.isneginf(edges)[None,:,None],torch.zeros_like(cdf),cdf)
    cdf=torch.where(torch.isposinf(edges)[None,:,None],torch.ones_like(cdf),cdf)
    masses=(cdf.diff(dim=1)*weight[:,None,:]).sum(-1).clamp_min(0)
    return masses/masses.sum(-1,keepdim=True).clamp_min(1e-12)


def head_loss(output,target,default,mask,weights):
    gate,raw=output
    if gate.shape[1]!=target.shape[1]:
        repeat=target.shape[1]//gate.shape[1]
        gate,raw=gate.repeat_interleave(repeat,1),raw.repeat_interleave(repeat,1)
    g,r,d=gate[mask].float(),raw[mask].float(),default[mask]
    y=target[mask].masked_fill(d,0)
    nll=-(r[:,:,0].log_softmax(-1)+components(r,y)).logsumexp(-1)
    point=(F.binary_cross_entropy_with_logits(g,d.float(),reduction='none')+nll.masked_fill(d,0)).mean(-1)
    return (weights[mask].detach().float()*point).mean()

def objective(outputs,target,default,mask,deep_supervision=False,weights=None,*,epoch):
    assert weights is not None and weights.shape==mask.shape
    if not deep_supervision:return head_loss(outputs,target,default,mask,weights)
    main,mid,coarse=outputs;factor=max(0.,1.-epoch/40.)
    loss=head_loss(main,target,default,mask,weights)
    if factor:loss=loss+factor*(.2*head_loss(mid,target,default,mask,weights)+.1*head_loss(coarse,target,default,mask,weights))
    return loss

def point_statistics(gate,raw,target,default,density_edges):
    raw=raw.float();target=target.float();default=default.bool()
    lw=raw[:,:,0].log_softmax(-1);w=lw.exp();mu=raw[:,:,1];sd=F.softplus(raw[:,:,2])+.025
    nll=-(lw+components(raw,target)).logsumexp(-1)
    nll=nll.masked_fill(default,float('nan'))
    cp=torch.stack([probability(j,w[:,j],mu[:,j],sd[:,j],torch.as_tensor(density_edges[j],device=raw.device,dtype=torch.float32)) for j in range(3)],1)
    entropy=-(cp*cp.clamp_min(1e-30).log()).sum(-1)/math.log(cp.shape[-1])
    cdf=torch.special.ndtr((target[...,None]-mu)/sd)
    phi=torch.atan((target[:,0,None]-mu[:,0,:8])/sd[:,0,:8])
    por=torch.cat([.5+(phi+torch.sin(phi)*torch.cos(phi))/math.pi,cdf[:,0,8:]],-1)
    cdf=torch.cat([por[:,None],cdf[:,1:]],1)
    target_cdf=(w*cdf).sum(-1).clamp(0,1).masked_fill(default,float('nan'))
    pg=gate.float().sigmoid()
    return {'nll':nll,'conditional_entropy':entropy,'target_tail':2*torch.minimum(target_cdf,1-target_cdf),'gate_probability':pg,'por_family_t3':w[:,0,:8].sum(-1)},cp.cumsum(-1)
