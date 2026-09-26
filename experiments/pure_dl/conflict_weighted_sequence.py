"""Downweight only continuously conflicting nondefault target supervision; gate loss unchanged."""
import math
import torch
from torch.nn import functional as F
from relative_only_sequence import Network

def head_loss(output,target,default,mask,weights):
    gate,raw=output
    if gate.shape[1]!=target.shape[1]:
        n=target.shape[1]//gate.shape[1];gate=gate.repeat_interleave(n,1);raw=raw.repeat_interleave(n,1)
    g=gate[mask].float();r=raw[mask].float();d=default[mask];y=target[mask].masked_fill(d,0)
    sd=F.softplus(r[:,:,2])+.025
    nll=-(r[:,:,0].log_softmax(-1)-.5*((y[:,:,None]-r[:,:,1])/sd).square()-sd.log()-.5*math.log(2*math.pi)).logsumexp(-1)
    q=weights[mask].detach().float()
    return F.binary_cross_entropy_with_logits(g,d.float())+(q*nll.masked_fill(d,0)).mean()

def objective(outputs,target,default,mask,deep_supervision=False,weights=None):
    assert weights is not None and weights.shape==target.shape
    if not deep_supervision:return head_loss(outputs,target,default,mask,weights)
    return sum(w*head_loss(o,target,default,mask,weights) for w,o in zip([1,.2,.1],outputs))

if __name__=='__main__':
    from population_likelihood_sequence import objective as original
    torch.manual_seed(37);torch.set_num_threads(2)
    g=torch.randn(2,16,3,requires_grad=True);r=torch.randn(2,16,3,3,8,requires_grad=True);y=torch.randn(2,16,3);d=torch.rand(2,16,3)>.5;m=torch.ones(2,16,dtype=torch.bool);m[:,0]=False
    one=torch.ones_like(y)
    assert torch.allclose(objective((g,r),y,d,m,weights=one),original((g,r),y,d,m),atol=1e-6)
    q=torch.full_like(y,.2,requires_grad=True);got=objective((g,r),y,d,m,weights=q)
    rr=r[m];dd=d[m];dist=torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=rr[:,:,0]),torch.distributions.Normal(rr[:,:,1],F.softplus(rr[:,:,2])+.025))
    expected=F.binary_cross_entropy_with_logits(g[m],dd.float())+(-dist.log_prob(y[m].masked_fill(dd,0))).masked_fill(dd,0).mean()*.2
    assert torch.allclose(got,expected,atol=1e-6)
    got.backward();assert q.grad is None and torch.isfinite(r.grad).all() and (r.grad[d]==0).all() and (r.grad[~m]==0).all()
    assert torch.allclose(objective((g,r),y,torch.ones_like(d),m,weights=q),F.binary_cross_entropy_with_logits(g[m],torch.ones_like(g[m])))
    print('PASS: unweighted equivalence, independent weighted NLL, unchanged gate, default masks and detached weights')
