import math
import torch
from torch.nn import functional as F
from multiscale_sequence import Network

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
    nll = -(weight-((truth[:, :, None]-mu)/sd).abs()-sd.log()-math.log(2)).logsumexp(-1)
    terms = []
    for j in range(3):
        active = default[:, j] < .5
        terms.append(nll[active, j].mean() if bool(active.any()) else nll[:, j].sum()*0)
    return F.binary_cross_entropy_with_logits(gate, default)+sum(terms)/3


def objective(outputs, target_z, target_default, supervised, deep_supervision=False):
    if not deep_supervision:
        return head_loss(outputs, target_z, target_default, supervised)
    main, mid, coarse = outputs
    return (head_loss(main, target_z, target_default, supervised)
            +.2*head_loss(mid, target_z, target_default, supervised)
            +.1*head_loss(coarse, target_z, target_default, supervised))



def conditional_probability(weight,mean,scale,edges):
    z=(edges[None,:,None]-mean[:,None,:])/scale[:,None,:]
    tail=.5*torch.exp(-z.abs())
    cdf=torch.where(z<0,tail,1-tail)
    prob=((cdf[:,1:]-cdf[:,:-1]).clamp_min(0)*weight[:,None,:]).sum(-1)
    return prob/prob.sum(1,keepdim=True).clamp_min(1e-12)

if __name__=='__main__':
    torch.set_num_threads(2);torch.manual_seed(179)
    raw=torch.randn(2,8,3,3,8,requires_grad=True);gate=torch.randn(2,8,3,requires_grad=True)
    y=torch.randn(2,8,3);default=torch.zeros_like(y,dtype=torch.bool);mask=torch.ones(2,8,dtype=torch.bool);mask[:,0]=False
    v=head_loss((gate,raw),y,default,mask)
    rr=raw[mask];dist=torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=rr[:,:,0]),torch.distributions.Laplace(rr[:,:,1],F.softplus(rr[:,:,2])+.025))
    expected=F.binary_cross_entropy_with_logits(gate[mask],default[mask].float())-dist.log_prob(y[mask]).mean()
    assert torch.allclose(v,expected,atol=1e-6)
    altered=y.clone();altered[~mask]=1e6
    assert torch.equal(v,head_loss((gate,raw),altered,default,mask))
    v.backward();assert torch.isfinite(raw.grad).all() and (raw.grad[~mask]==0).all()
    edges=torch.tensor([-float('inf'),-2.,0.,2.,float('inf')]);w=torch.tensor([[.2,.8]]);mu=torch.tensor([[-1.,1.]]);sc=torch.tensor([[.3,1.1]])
    cp=conditional_probability(w,mu,sc,edges)
    cdf=torch.distributions.Laplace(mu[:,None],sc[:,None]).cdf(edges[None,:,None])
    assert torch.allclose(cp,((cdf[:,1:]-cdf[:,:-1])*w[:,None]).sum(-1),atol=1e-6)
    assert torch.allclose(cp.sum(-1),torch.ones(1))
    print('PASS: Laplace mixture NLL and CDF match PyTorch; masked labels/gradients isolated')
