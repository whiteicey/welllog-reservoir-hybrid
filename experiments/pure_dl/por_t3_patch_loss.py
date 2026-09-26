"""POR Student-t3 and PERM/SW Gaussian; final point Q and annealed auxiliaries."""
import math
import torch
from torch.nn import functional as F
from por_student3_sequence import por_probability
from train_invariant import conditional_probability

def components(raw,target):
    sd=F.softplus(raw[:,:,2])+.025;mu=raw[:,:,1]
    logp=-.5*((target[...,None]-mu)/sd).square()-sd.log()-.5*math.log(2*math.pi)
    scale=sd[:,0]/math.sqrt(3)
    por=math.lgamma(2)-math.lgamma(1.5)-.5*math.log(3*math.pi)-scale.log()-2*torch.log1p(((target[:,0,None]-mu[:,0])/scale).square()/3)
    return torch.cat([por[:,None],logp[:,1:]],1)

def probability(j,weight,mu,sd,edges):
    return por_probability(weight,mu,sd,edges) if j==0 else conditional_probability(weight,mu,sd,edges)

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
    phi=torch.atan((target[:,0,None]-mu[:,0])/sd[:,0])
    cdf=torch.cat([(.5+(phi+torch.sin(phi)*torch.cos(phi))/math.pi)[:,None],cdf[:,1:]],1)
    target_cdf=(w*cdf).sum(-1).clamp(0,1).masked_fill(default,float('nan'))
    pg=gate.float().sigmoid()
    return {'nll':nll,'conditional_entropy':entropy,'target_tail':2*torch.minimum(target_cdf,1-target_cdf),'gate_probability':pg},cp.cumsum(-1)
