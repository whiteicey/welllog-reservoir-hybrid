"""Fixed conditional CDF signatures augment detached posterior routing."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from paired_posterior_router_sequence import Network as Baseline, posterior_descriptors, retained_indices, calibration_indices, VIEW_NAMES

CDF_POINTS=(-3.,-2.,-1.,0.,1.,2.,3.)


def cdf_descriptors(outputs, grid):
    signatures=[]
    for _,raw in outputs:
        raw=raw.detach().float()
        weight=raw[...,0,:].softmax(-1)
        mu=raw[...,1,:]
        sd=F.softplus(raw[...,2,:])+.025
        x=(grid.float()[None,None,None,:,None]-mu[...,None,:])/sd[...,None,:]
        cdf=torch.special.ndtr(x)
        por_x=x[:,:,0]
        # sd is actual t3 standard deviation. phi+sin(phi)*cos(phi)
        # equals atan(x)+x/(1+x*x), avoiding an overflowing square.
        phi=torch.atan(por_x)
        por=.5+(phi+torch.sin(phi)*torch.cos(phi))/math.pi
        cdf=torch.cat([por[:,:,None],cdf[:,:,1:]],dim=2)
        mixture=(cdf*weight[...,None,:]).sum(-1).clamp(0,1)
        signatures.append(mixture.flatten(-2))
    return torch.cat(signatures,-1)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices):
        super().__init__(features,initial_means,retain_indices=retain_indices,
                         numeric_indices=numeric_indices,conditioning_indices=conditioning_indices)
        self.register_buffer('cdf_points',torch.tensor(CDF_POINTS,dtype=torch.float32))
        old=self.router[0]
        with torch.random.fork_rng(devices=[]):
            expanded=nn.Linear(134,32,device=old.weight.device,dtype=old.weight.dtype)
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.weight[:,:92].copy_(old.weight)
                expanded.bias.copy_(old.bias)
        self.router[0]=expanded

    def route_logits(self, features, valid, main_outputs):
        original=features.index_select(-1,self.router_indices).float()
        moments=posterior_descriptors(main_outputs)
        signatures=cdf_descriptors(main_outputs,self.cdf_points)
        x=torch.cat([original,moments,signatures],-1).masked_fill(~valid[...,None],0)
        return self.router(x).reshape(*features.shape[:2],3,2)
