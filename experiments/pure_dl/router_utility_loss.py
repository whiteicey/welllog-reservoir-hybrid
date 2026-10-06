"""Bounded eight-node official-kernel auxiliary for routing only."""
import math
import numpy as np
import torch
from torch.nn import functional as F

_NODES,_WEIGHTS=np.polynomial.legendre.leggauss(8)
NODES=(_NODES+1)/2
QUADRATURE_WEIGHTS=_WEIGHTS/2
BOUNDS=((0.,50.),(.01,10000.),(8.305,99.9))
DEFAULTS=(.1,.01,99.9)


@torch.no_grad()
def member_utilities(members, target_raw, mask, med, scale):
    target=target_raw[mask].detach().float()
    nodes=torch.as_tensor(NODES,device=target.device,dtype=torch.float32)
    quadrature=torch.as_tensor(QUADRATURE_WEIGHTS,device=target.device,dtype=torch.float32)
    med=torch.as_tensor(med,device=target.device,dtype=torch.float32)
    scale=torch.as_tensor(scale,device=target.device,dtype=torch.float32)
    intervals=[];atom=[]
    for channel,(lower,upper) in enumerate(BOUNDS):
        truth=target[:,channel]
        if channel==1:
            logtruth=truth.clamp_min(1e-5).log10()
            low=torch.pow(10.,logtruth[:,None]-nodes)
            high=torch.pow(10.,logtruth[:,None]+nodes)
            atom.append((1-(math.log10(DEFAULTS[channel])-logtruth).abs()).clamp_min(0))
        else:
            radius=(.08 if channel==0 else .05)*(truth.abs()+1e-5)
            low=truth[:,None]-radius[:,None]*nodes
            high=truth[:,None]+radius[:,None]*nodes
            atom.append((1-(DEFAULTS[channel]-truth).abs()/radius).clamp_min(0))
        outside=(high<lower)|(low>upper)
        clipped_low=low.clamp(lower,upper);clipped_high=high.clamp(lower,upper)
        transform=torch.log1p if channel==0 else (torch.log10 if channel==1 else torch.log)
        lo=(transform(clipped_low)-med[channel])/scale[channel]
        hi=(transform(clipped_high)-med[channel])/scale[channel]
        intervals.append((lo,hi,low<=lower,high>=upper,outside))
    atom=torch.stack(atom,-1)
    utilities=[]
    for member in members:
        gate,raw=member[0]
        gate=gate[mask].detach().float().sigmoid()
        raw=raw[mask].detach().float()
        weight=raw[:,:,0].softmax(-1)
        mean=raw[:,:,1];sd=F.softplus(raw[:,:,2])+.025
        continuous=[]
        for channel,(lo,hi,lower_atom,upper_atom,outside) in enumerate(intervals):
            a=(lo[:,:,None]-mean[:,channel,None,:])/sd[:,channel,None,:]
            b=(hi[:,:,None]-mean[:,channel,None,:])/sd[:,channel,None,:]
            if channel==0:
                def cdf(value):
                    phi=torch.atan(value)
                    return .5+(phi+torch.sin(phi)*torch.cos(phi))/math.pi
                cdf_low,cdf_high=cdf(a),cdf(b)
            else:
                cdf_low,cdf_high=torch.special.ndtr(a),torch.special.ndtr(b)
            # Include mass collapsed onto physical action bounds by clipping.
            cdf_low=torch.where(lower_atom[:,:,None],torch.zeros_like(cdf_low),cdf_low)
            cdf_high=torch.where(upper_atom[:,:,None],torch.ones_like(cdf_high),cdf_high)
            mass=(cdf_high-cdf_low).clamp(0,1).masked_fill(outside[:,:,None],0)
            value=((mass*weight[:,channel,None,:]).sum(-1)*quadrature).sum(-1)
            continuous.append(value)
        continuous=torch.stack(continuous,-1).clamp(0,1)
        utilities.append((gate*atom+(1-gate)*continuous).clamp(0,1))
    return torch.stack(utilities,-1).detach()


def utility_loss(members, routing_weights, target_raw, mask, weights, med, scale):
    utility=member_utilities(members,target_raw,mask,med,scale)
    expectation=(routing_weights[mask].float()*utility).sum(-1)
    return (weights[mask].detach().float()*(1-expectation).mean(-1)).mean()
