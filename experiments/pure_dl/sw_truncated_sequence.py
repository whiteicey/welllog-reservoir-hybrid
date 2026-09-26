"""Globally upper-truncated SW mixture; train/inference use identical support."""
import math
import torch
from torch.nn import functional as F
from depth_norm_joint_sequence import Network
def head_loss(output, target_z, target_default, supervised, upper):
    gate, raw = output
    length = target_z.shape[1]
    if gate.shape[1] != length:
        repeat = length//gate.shape[1]
        assert repeat*gate.shape[1] == length
        gate = gate.repeat_interleave(repeat, dim=1)
        raw = raw.repeat_interleave(repeat, dim=1)
    assert supervised.shape == target_z.shape[:2] and bool(supervised.any())
    # Select first: invalid/out-of-well labels have exactly no loss or gradients.
    gate = gate[supervised].float(); raw = raw[supervised].float()
    truth = target_z[supervised]; default = target_default[supervised].float()
    weight = raw[:, :, 0].log_softmax(-1)
    mu = raw[:, :, 1]; sd = F.softplus(raw[:, :, 2])+.025
    nll = -(weight-.5*((truth[:, :, None]-mu)/sd).square()-sd.log()-.5*math.log(2*math.pi)).logsumexp(-1)
    log_z=(weight[:,2]+torch.special.log_ndtr((upper-mu[:,2])/sd[:,2])).logsumexp(-1)
    nll=nll.clone();nll[:,2]=nll[:,2]+log_z
    return F.binary_cross_entropy_with_logits(gate, default)+nll.masked_fill(default.bool(),0).mean()



def objective_for_stats(median,scale):
    upper=(math.log(100.)-float(median[2]))/float(scale[2])
    def objective(outputs,z,default,supervised,deep_supervision=False):
        if not deep_supervision:return head_loss(outputs,z,default,supervised,upper)
        return sum(w*head_loss(o,z,default,supervised,upper) for w,o in zip([1,.2,.1],outputs))
    return objective

def sw_probability(weight,mu,sd,edges,upper):
    log_upper=torch.special.log_ndtr((upper-mu)/sd)
    adjusted=(weight.clamp_min(1e-30).log()+log_upper).softmax(-1)
    capped=edges.clamp_max(upper)
    log_cdf=torch.special.log_ndtr((capped[None,:,None]-mu[:,None,:])/sd[:,None,:])
    cdf=(log_cdf-log_upper[:,None,:]).clamp_max(0).exp()
    prob=((cdf[:,1:]-cdf[:,:-1]).clamp_min(0)*adjusted[:,None,:]).sum(-1)
    return prob/prob.sum(-1,keepdim=True).clamp_min(1e-12)

if __name__=='__main__':
    import numpy as np
    from scipy.stats import norm
    torch.set_num_threads(2);torch.manual_seed(98)
    w=torch.randn(5,8,dtype=torch.float64).softmax(-1);mu=torch.randn(5,8,dtype=torch.float64);sd=torch.rand(5,8,dtype=torch.float64)+.2
    edges=torch.tensor(np.r_[-np.inf,np.linspace(-3,3,255),np.inf]);upper=.8
    got=sw_probability(w,mu,sd,edges,upper)
    cdf=norm.cdf((np.minimum(edges.numpy(),upper)[None,:,None]-mu.numpy()[:,None,:])/sd.numpy()[:,None,:])
    expected=(np.diff(cdf,axis=1)*w.numpy()[:,None,:]).sum(-1);expected/=expected.sum(-1,keepdims=True)
    assert np.max(abs(got.numpy()-expected))<1e-12
    assert torch.allclose(got.sum(-1),torch.ones(5,dtype=torch.float64))
    extreme=sw_probability(w,mu+100,sd,edges,upper);assert torch.isfinite(extreme).all() and torch.allclose(extreme.sum(-1),torch.ones(5,dtype=torch.float64))
    r=torch.randn(2,32,3,3,8,requires_grad=True);g=torch.randn(2,32,3,requires_grad=True);y=torch.randn(2,32,3);d=torch.rand(2,32,3)>.5;m=torch.ones(2,32,dtype=torch.bool);m[:,0]=False
    loss=head_loss((g,r),y,d,m,upper);altered=y.clone();altered[d]=1e4;altered[:,0]=-1e4
    assert torch.equal(loss,head_loss((g,r),altered,d,m,upper));loss.backward()
    assert torch.isfinite(r.grad).all() and torch.equal(r.grad[d],torch.zeros_like(r.grad[d]))
    print('PASS: independent truncated CDF, normalized extreme tails, default/missing invariance and finite masked gradients')
