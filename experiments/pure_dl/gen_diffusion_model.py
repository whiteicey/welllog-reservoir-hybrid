"""Conditional cosine VP diffusion with v prediction and fixed quantile DDIM inference."""
import torch
from torch import nn
from torch.nn import functional as F
from blur_no_point_sequence import Network as Baseline
CONTEXT=16
SAMPLES=64
STEPS=16
class ContextHead(nn.Module):
    def __init__(self,width):
        super().__init__();self.gate=nn.Conv1d(width,3,1);self.density=nn.Conv1d(width,3*CONTEXT,1)
    def forward(self,x):return self.gate(x).transpose(1,2),self.density(x).transpose(1,2).reshape(len(x),-1,3,CONTEXT)
class Velocity(nn.Module):
    def __init__(self):
        super().__init__();self.context=nn.Linear(CONTEXT,32);self.state_time=nn.Linear(6,32);self.hidden=nn.Linear(32,32);self.output=nn.Linear(32,1)
    def forward(self,z,t,context):
        while t.ndim<z.ndim:t=t.unsqueeze(-1)
        t=t.expand_as(z)
        a=torch.stack([z,t,torch.sin(2*torch.pi*t),torch.cos(2*torch.pi*t),torch.sin(4*torch.pi*t),torch.cos(4*torch.pi*t)],-1)
        c=self.context(context)
        while c.ndim<a.ndim:c=c.unsqueeze(-2)
        return self.output(F.silu(self.hidden(F.silu(c+self.state_time(a))))).squeeze(-1)
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means);self.main_head=ContextHead(128);self.aux_mid=ContextHead(96);self.aux_coarse=ContextHead(128);self.velocity=Velocity()
    def head_loss(self,output,target,default,mask,noise=None,time=None):
        gate,ctx=output
        if gate.shape[1]!=target.shape[1]:
            n=target.shape[1]//gate.shape[1];gate=gate.repeat_interleave(n,1);ctx=ctx.repeat_interleave(n,1)
        ctx=ctx[mask].float();gate=gate[mask].float();d=default[mask];y=target[mask].masked_fill(d,0)
        e=torch.randn_like(y) if noise is None else noise
        t=torch.rand_like(y) if time is None else time
        a=torch.cos(torch.pi*t/2);s=torch.sin(torch.pi*t/2)
        z=a*y+s*e;v=self.velocity(z,t,ctx)
        return F.binary_cross_entropy_with_logits(gate,d.float())+(v-(a*e-s*y)).square().masked_fill(d,0).mean()
    def objective(self,outputs,target,default,mask,deep_supervision=False):
        if not deep_supervision:return self.head_loss(outputs,target,default,mask)
        return sum(w*self.head_loss(o,target,default,mask) for w,o in zip([1,.2,.1],outputs))
    @torch.no_grad()
    def generate(self,context):
        q=(torch.arange(SAMPLES,device=context.device,dtype=torch.float32)+.5)/SAMPLES
        initial=torch.special.ndtri(q)
        z=initial.expand(context.shape[:-1]+(SAMPLES,)).clone()
        # Deterministic DDIM eta=0 with cosine VP schedule and v parameterization.
        for step in range(STEPS,0,-1):
            t=torch.full_like(z,step/STEPS);tn=torch.full_like(z,(step-1)/STEPS)
            a=torch.cos(torch.pi*t/2);s=torch.sin(torch.pi*t/2)
            v=self.velocity(z,t,context.float());clean=a*z-s*v;noise=s*z+a*v
            z=torch.cos(torch.pi*tn/2)*clean+torch.sin(torch.pi*tn/2)*noise
        return z

def probabilities(samples,edges):
    bins=torch.searchsorted(edges[1:-1].contiguous(),samples.contiguous())
    p=torch.zeros(samples.shape[:-1]+(len(edges)-1,),device=samples.device,dtype=torch.float32)
    p.scatter_add_(-1,bins,torch.full_like(samples,1/SAMPLES))
    return p
