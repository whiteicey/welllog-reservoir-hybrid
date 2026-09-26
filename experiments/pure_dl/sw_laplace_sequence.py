import math
import torch
from torch.nn import functional as F
from blur_no_dropout_sequence import Network

def head_loss(output, target_z, target_default, supervised):
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
    b=sd[:,2]/math.sqrt(2)
    laplace=-(weight[:,2]-(truth[:,2,None]-mu[:,2]).abs()/b-(2*b).log()).logsumexp(-1)
    nll=nll.clone();nll[:,2]=laplace
    return F.binary_cross_entropy_with_logits(gate, default)+nll.masked_fill(default.bool(),0).mean()



def objective(outputs, target_z, target_default, supervised, deep_supervision=False):
    if not deep_supervision:
        return head_loss(outputs, target_z, target_default, supervised)
    main, mid, coarse = outputs
    return (head_loss(main, target_z, target_default, supervised)
            +.2*head_loss(mid, target_z, target_default, supervised)
            +.1*head_loss(coarse, target_z, target_default, supervised))




def sw_probability(weight,mu,std,edges):
    b=std/math.sqrt(2)
    z=(edges[None,:,None]-mu[:,None,:])/b[:,None,:]
    cdf=.5-.5*z.sign()*torch.expm1(-z.abs())
    mass=((cdf[:,1:]-cdf[:,:-1]).clamp_min(0)*weight[:,None,:]).sum(-1)
    return mass/mass.sum(-1,keepdim=True).clamp_min(1e-12)

if __name__=='__main__':
    import numpy as np
    from scipy.stats import laplace
    torch.set_num_threads(2);torch.manual_seed(79)
    w=torch.randn(5,8,dtype=torch.float64).softmax(-1);mu=torch.randn(5,8,dtype=torch.float64);sd=torch.rand(5,8,dtype=torch.float64)+.1;edges=torch.tensor(np.r_[-np.inf,np.linspace(-4,4,255),np.inf])
    got=sw_probability(w,mu,sd,edges)
    cdf=laplace.cdf(edges.numpy()[None,:,None],loc=mu.numpy()[:,None,:],scale=sd.numpy()[:,None,:]/np.sqrt(2))
    expected=(np.diff(cdf,axis=1)*w.numpy()[:,None,:]).sum(-1)
    assert np.max(abs(got.numpy()-expected))<1e-12
    raw=torch.randn(1,16,3,3,8,requires_grad=True);gate=torch.randn(1,16,3);y=torch.randn(1,16,3);d=torch.ones(1,16,3,dtype=torch.bool);d[:,:,2]=False;m=torch.ones(1,16,dtype=torch.bool);m[:,0]=False
    value=head_loss((gate,raw),y,d,m);r=raw[m][:,2]
    dist=torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=r[:,0]),torch.distributions.Laplace(r[:,1],(F.softplus(r[:,2])+.025)/math.sqrt(2)))
    expected=F.binary_cross_entropy_with_logits(gate[m],d[m].float())-dist.log_prob(y[m][:,2]).sum()/(int(m.sum())*3)
    assert torch.allclose(value,expected,atol=1e-6)
    altered=y.clone();altered[d]=1e4;altered[:,0]=float('nan');assert torch.equal(value,head_loss((gate,raw),altered,d,m))
    value.backward();assert torch.isfinite(raw.grad).all() and torch.equal(raw.grad[d],torch.zeros_like(raw.grad[d]))
    print('PASS: independent Laplace mixture likelihood/CDF and masked finite gradients')
