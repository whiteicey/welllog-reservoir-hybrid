"""Remove the point MLP bypass; retain all sequence inputs and original decoder."""
import torch
from torch import nn
from blur_no_dropout_sequence import Network as Baseline
from population_likelihood_sequence import objective
class EmptyPoint(nn.Module):
    def forward(self,x):return x[:,:0,:]
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        self.point=EmptyPoint();old=self.combine[0]
        with torch.random.fork_rng(devices=[]):new=nn.Conv1d(64,128,1)
        with torch.no_grad():new.weight.copy_(old.weight[:,:64]);new.bias.copy_(old.bias)
        self.combine[0]=new
if __name__=='__main__':
    torch.set_num_threads(2);torch.manual_seed(42);a=Baseline(features=12).eval()
    torch.manual_seed(42);m=Network(features=12).eval()
    with torch.no_grad():a.point[-1].weight.zero_();a.point[-1].bias.zero_()
    x=torch.randn(2,256,12);d=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:32]=False
    with torch.no_grad():
        aa=a(x,d,v,absolute_depth=d);bb=m(x,d,v,absolute_depth=d)
        assert all(torch.allclose(q,w,atol=2e-6,rtol=2e-6) for q,w in zip(aa,bb))
        xx=x.clone();xx[1]+=100;cc=m(xx,d,v,absolute_depth=d)
        assert all(torch.equal(q[0],w[0]) for q,w in zip(bb,cc))
        xx=x.clone();xx[~v]+=100;dd=d.clone();dd[~v]+=100;cc=m(xx,dd,v,absolute_depth=dd)
        assert all(torch.equal(q,w) for q,w in zip(bb,cc))
    loss=objective(m(x,d,v,True,absolute_depth=d),torch.randn(2,256,3),torch.rand(2,256,3)>.5,v,True);loss.backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in m.parameters())
    assert not any(k.startswith('point.') for k in m.state_dict())
    print('PASS: zero-bypass baseline equivalence, well/padding isolation, finite remaining gradients')
