"""PERM CRPS replaces only its NLL; retain mixture architecture and decoding."""
import math
import torch
from torch.nn import functional as F
from blur_no_dropout_sequence import Network
from gaussian_crps import mixture_crps, self_test as integral_test

def objective_for_scale(scale):
    def head(output,z,default,supervised):
        gate,raw=output
        if gate.shape[1]!=z.shape[1]:
            repeat=z.shape[1]//gate.shape[1];gate=gate.repeat_interleave(repeat,1);raw=raw.repeat_interleave(repeat,1)
        gate=gate[supervised].float();raw=raw[supervised].float();y=z[supervised];d=default[supervised].bool()
        logw=raw[:,:,0].log_softmax(-1);mu=raw[:,:,1];sd=F.softplus(raw[:,:,2])+.025
        terms=-(logw-.5*((y[:,:,None]-mu)/sd).square()-sd.log()-.5*math.log(2*math.pi)).logsumexp(-1)
        active=~d[:,1]
        if bool(active.any()):
            terms=terms.clone();terms[active,1]=mixture_crps(logw[active,1],mu[active,1],sd[active,1],y[active,1])*float(scale[1])
        return F.binary_cross_entropy_with_logits(gate,d.float())+terms.masked_fill(d,0).mean()
    def objective(outputs,z,default,supervised,deep_supervision=False):
        if not deep_supervision:return head(outputs,z,default,supervised)
        return sum(w*head(o,z,default,supervised) for w,o in zip([1,.2,.1],outputs))
    return objective

if __name__=='__main__':
    torch.set_num_threads(2);print(integral_test())
    raw=torch.randn(2,16,3,3,8,requires_grad=True);gate=torch.randn(2,16,3,requires_grad=True);y=torch.randn(2,16,3);d=torch.rand(2,16,3)>.5;m=torch.ones(2,16,dtype=torch.bool);m[:,0]=False
    objective=objective_for_scale([1,2,1]);value=objective((gate,raw),y,d,m)
    altered=y.clone();altered[d]=10000;altered[:,0]=float('nan')
    assert torch.equal(value,objective((gate,raw),altered,d,m));value.backward()
    assert torch.isfinite(raw.grad).all() and torch.equal(raw.grad[d],torch.zeros_like(raw.grad[d])) and torch.equal(raw.grad[:,0],torch.zeros_like(raw.grad[:,0]))
    assert raw.grad[:,:,1,0].abs().sum()>0 and raw.grad[:,:,1,1].abs().sum()>0 and raw.grad[:,:,1,2].abs().sum()>0
    print('PASS: independent CRPS integral, default/missing invariance and gradients to all PERM mixture parameter families')
