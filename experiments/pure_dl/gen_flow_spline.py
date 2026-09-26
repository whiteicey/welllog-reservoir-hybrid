"""Conditional rational-quadratic normalizing flow; equations from Durkan et al. 2019.
Independent implementation, eight bins, identity tails, conditional affine pretransform.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from blur_no_dropout_sequence import Network as Baseline
K=8
P=3*K+1

def knots(raw):
    w=.001+(1-K*.001)*raw[...,2:2+K].softmax(-1)
    h=.001+(1-K*.001)*raw[...,2+K:2+2*K].softmax(-1)
    x=F.pad(w.cumsum(-1),(1,0))*8-4
    y=F.pad(h.cumsum(-1),(1,0))*8-4
    # force endpoints exactly to avoid roundoff in tails
    x=torch.cat([x[...,:1],x[...,1:-1],torch.full_like(x[...,-1:],4)],-1)
    y=torch.cat([y[...,:1],y[...,1:-1],torch.full_like(y[...,-1:],4)],-1)
    d=F.pad(.001+F.softplus(raw[...,2+2*K:]),(1,1),value=1.)
    return x,y,d

def spline(value,raw,inverse=False):
    x,y,d=knots(raw)
    while x.ndim<value.ndim+1:x=x.unsqueeze(-2);y=y.unsqueeze(-2);d=d.unsqueeze(-2)
    shape=value.shape+(K+1,)
    x=x.expand(shape);y=y.expand(shape);d=d.expand(shape)
    inside=(value>=-4)&(value<=4)
    v=value.clamp(-4,4)
    idx=((v[...,None]>= (y if inverse else x)).sum(-1)-1).clamp(0,K-1)[...,None]
    get=lambda q:q.gather(-1,idx).squeeze(-1)
    x0=get(x);y0=get(y);w=get(x[...,1:]-x[...,:-1]);h=get(y[...,1:]-y[...,:-1]);a=get(d);b=get(d[...,1:]);s=h/w
    if inverse:
        dy=v-y0;c=-s*dy;bb=h*a-dy*(a+b-2*s);aa=dy*(a+b-2*s)+h*(s-a)
        theta=2*c/(-bb-torch.sqrt((bb.square()-4*aa*c).clamp_min(0)))
    else:theta=(v-x0)/w
    q=theta*(1-theta);den=s+(a+b-2*s)*q
    result=x0+theta*w if inverse else y0+h*(s*theta.square()+a*q)/den
    logdet=(s.square()*(b*theta.square()+2*s*q+a*(1-theta).square())).log()-2*den.log()
    return torch.where(inside,result,value),torch.where(inside,-logdet if inverse else logdet,torch.zeros_like(value))

def transform(value,raw,inverse=False):
    loc=raw[...,0];scale=F.softplus(raw[...,1])+.025
    while loc.ndim<value.ndim:loc=loc.unsqueeze(-1);scale=scale.unsqueeze(-1)
    if inverse:
        out,ld=spline(value,raw,True);return out*scale+loc,ld+scale.log()
    out,ld=spline((value-loc)/scale,raw);return out,ld-scale.log()

def probabilities(raw,edges):
    parts=[]
    for start in range(0,len(raw),512):
        r=raw[start:start+512]
        values=edges[None].expand(len(r),-1)
        z,_=transform(values,r)
        cdf=torch.special.ndtr(z)
        p=cdf.diff(dim=-1).clamp_min(0);parts.append(p/p.sum(-1,keepdim=True))
    return torch.cat(parts)

class FlowHead(nn.Module):
    def __init__(self,width):
        super().__init__();self.gate=nn.Conv1d(width,3,1);self.density=nn.Conv1d(width,3*P,1)
        nn.init.normal_(self.density.weight,0,.005);nn.init.zeros_(self.density.bias)
        with torch.no_grad():
            b=self.density.bias.view(3,P);b[:,1]=math.log(math.expm1(1-.025));b[:,2+2*K:]=math.log(math.expm1(1-.001))
    def forward(self,x):return self.gate(x).transpose(1,2),self.density(x).transpose(1,2).reshape(len(x),-1,3,P)
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        self.main_head=FlowHead(128);self.aux_mid=FlowHead(96);self.aux_coarse=FlowHead(128)
def head_loss(output,target,default,mask):
    gate,raw=output
    if gate.shape[1]!=target.shape[1]:
        repeat=target.shape[1]//gate.shape[1];gate=gate.repeat_interleave(repeat,1);raw=raw.repeat_interleave(repeat,1)
    gate=gate[mask].float();raw=raw[mask].float();d=default[mask]
    # Defaults excluded before transform: arbitrary default labels cannot affect density gradients.
    y=target[mask].masked_fill(d,0);z,ld=transform(y,raw)
    nll=.5*z.square()+.5*math.log(2*math.pi)-ld
    return F.binary_cross_entropy_with_logits(gate,d.float())+nll.masked_fill(d,0).mean()
def objective(outputs,target,default,mask,deep_supervision=False):
    if not deep_supervision:return head_loss(outputs,target,default,mask)
    return sum(w*head_loss(o,target,default,mask) for w,o in zip([1,.2,.1],outputs))
