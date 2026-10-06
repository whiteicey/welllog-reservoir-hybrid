"""Remove absolute-depth stem input, preserving relative depth and all measured features."""
import torch
from torch import nn
from blur_ungated_sequence import Network as Baseline
from population_likelihood_sequence import objective
class RelativeOnlyStem(nn.Conv1d):
    def forward(self,x):return super().forward(x[:,:-1,:])
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means);old=self.stem
        with torch.random.fork_rng(devices=[]):new=RelativeOnlyStem(features+2,64,1)
        with torch.no_grad():new.weight.copy_(old.weight[:,:-1]);new.bias.copy_(old.bias)
        self.stem=new
if __name__=='__main__':
    torch.set_num_threads(2);torch.manual_seed(42);a=Baseline(features=12).eval()
    torch.manual_seed(42);m=Network(features=12).eval()
    x=torch.randn(2,256,12);d=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:32]=False
    with torch.no_grad():
        aa=a(x,d,v,absolute_depth=d);bb=m(x,d,v,absolute_depth=d)
        assert all(torch.allclose(q,w,atol=2e-6,rtol=2e-6) for q,w in zip(aa,bb))
        shifted=m(x,d,v,absolute_depth=d+1000)
        assert all(torch.equal(q,w) for q,w in zip(bb,shifted))
        xx=x.clone();xx[1]+=100;cc=m(xx,d,v,absolute_depth=d)
        assert all(torch.equal(q[0],w[0]) for q,w in zip(bb,cc))
        xx=x.clone();xx[~v]+=100;dd=d.clone();dd[~v]+=100;cc=m(xx,dd,v,absolute_depth=dd)
        assert all(torch.equal(q,w) for q,w in zip(bb,cc))
    loss=objective(m(x,d,v,True,absolute_depth=d),torch.randn(2,256,3),torch.rand(2,256,3)>.5,v,True);loss.backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in m.parameters())
    print('PASS: initialized baseline equivalence, explicit depth-offset invariance, isolation and finite gradients')
