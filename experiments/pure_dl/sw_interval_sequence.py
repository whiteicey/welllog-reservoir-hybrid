"""SW interval-averaged likelihood; unchanged POR/PERM point likelihood."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from population_likelihood_sequence import Network
NODES,WEIGHTS=np.polynomial.legendre.leggauss(32)

def sw_interval_nll(raw,z,scale):
    lo=math.log(.95)/scale;hi=math.log(1.05)/scale
    nodes=torch.as_tensor(NODES,device=raw.device,dtype=raw.dtype)
    lw=torch.as_tensor(np.log(WEIGHTS/2),device=raw.device,dtype=raw.dtype)
    locations=z[:,None]+(hi+lo)/2+nodes*(hi-lo)/2
    mu=raw[:,1];sd=F.softplus(raw[:,2])+.025
    lp=raw[:,0].log_softmax(-1)[:,None,:]-.5*((locations[:,:,None]-mu[:,None,:])/sd[:,None,:]).square()-sd[:,None,:].log()-.5*math.log(2*math.pi)
    return -(lp.logsumexp(-1)+lw).logsumexp(-1)

def objective_for_stats(median,scale):
    sw_scale=float(scale[2])
    def head_loss(output,z,default,supervised):
        gate,raw=output
        if gate.shape[1]!=z.shape[1]:
            repeat=z.shape[1]//gate.shape[1]
            gate=gate.repeat_interleave(repeat,1);raw=raw.repeat_interleave(repeat,1)
        gate=gate[supervised].float();raw=raw[supervised].float()
        truth=z[supervised];dd=default[supervised].bool()
        mu=raw[:,:,1];sd=F.softplus(raw[:,:,2])+.025
        nll=-(raw[:,:,0].log_softmax(-1)-.5*((truth[:,:,None]-mu)/sd).square()-sd.log()-.5*math.log(2*math.pi)).logsumexp(-1)
        active=~dd[:,2]
        if bool(active.any()):
            nll=nll.clone();nll[active,2]=sw_interval_nll(raw[active,2],truth[active,2],sw_scale)
        return F.binary_cross_entropy_with_logits(gate,dd.float())+nll.masked_fill(dd,0).mean()
    def objective(outputs,z,default,supervised,deep_supervision=False):
        if not deep_supervision:return head_loss(outputs,z,default,supervised)
        return sum(w*head_loss(o,z,default,supervised) for w,o in zip([1,.2,.1],outputs))
    return objective

if __name__=='__main__':
    from scipy.special import ndtr,softmax
    torch.set_num_threads(2);torch.manual_seed(53)
    raw=torch.randn(40,3,8,dtype=torch.float64,requires_grad=True);z=torch.randn(40,dtype=torch.float64)
    for scale in [.1,.5,1.]:
        got=sw_interval_nll(raw,z,scale)
        r=raw.detach().numpy();mu=r[:,1];sd=np.logaddexp(0,r[:,2])+.025
        lo=z.numpy()+math.log(.95)/scale;hi=z.numpy()+math.log(1.05)/scale
        density=((ndtr((hi[:,None]-mu)/sd)-ndtr((lo[:,None]-mu)/sd))*softmax(r[:,0],axis=-1)).sum(-1)/(hi-lo)
        assert np.max(abs(got.detach().numpy()+np.log(density)))<1e-7
    got.sum().backward();assert torch.isfinite(raw.grad).all()
    objective=objective_for_stats([0,0,0],[1,1,.5])
    r=torch.randn(2,32,3,3,8,requires_grad=True);g=torch.randn(2,32,3,requires_grad=True);y=torch.randn(2,32,3);d=torch.rand(2,32,3)>.5;m=torch.ones(2,32,dtype=torch.bool);m[:,0]=False
    loss=objective((g,r),y,d,m);altered=y.clone();altered[d]=1e4;altered[:,0]=-1e4
    assert torch.equal(loss,objective((g,r),altered,d,m));loss.backward()
    assert torch.equal(r.grad[d],torch.zeros_like(r.grad[d]));assert torch.equal(r.grad[:,0],torch.zeros_like(r.grad[:,0]))
    print('PASS: interval quadrature vs independent Gaussian CDF; finite gradients; default and missing masks')
