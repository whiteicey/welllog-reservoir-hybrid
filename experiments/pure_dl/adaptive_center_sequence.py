"""Zero-initialized feature-wise window centering before the sequence encoder."""
import torch
from torch import nn
from relative_only_sequence import Network as Baseline
from population_likelihood_sequence import objective
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        self.center_strength=nn.Parameter(torch.zeros(features))
    def forward(self,features,relative_depth,valid,auxiliary=False,absolute_depth=None):
        mask=valid[:,:,None]
        mean=features.float().masked_fill(~mask,0).sum(1,keepdim=True)/mask.sum(1,keepdim=True).clamp_min(1)
        centered=features.float()-self.center_strength.tanh()[None,None,:]*mean
        return super().forward(centered.to(features.dtype),relative_depth,valid,auxiliary,absolute_depth=absolute_depth)
if __name__=='__main__':
    torch.set_num_threads(2);torch.manual_seed(42);a=Baseline(features=12).eval()
    torch.manual_seed(42);m=Network(features=12).eval()
    x=torch.randn(2,256,12);d=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:32]=False
    with torch.no_grad():
        aa=a(x,d,v,absolute_depth=d);bb=m(x,d,v,absolute_depth=d)
        assert all(torch.equal(q,w) for q,w in zip(aa,bb))
        m.center_strength.fill_(.3);bb=m(x,d,v,absolute_depth=d)
        xx=x.clone();xx[1]+=100;cc=m(xx,d,v,absolute_depth=d)
        assert all(torch.equal(q[0],w[0]) for q,w in zip(bb,cc))
        xx=x.clone();xx[~v]+=100;dd=d.clone();dd[~v]+=100;cc=m(xx,dd,v,absolute_depth=dd)
        assert all(torch.equal(q,w) for q,w in zip(bb,cc))
        shifted=m(x,d,v,absolute_depth=d+1000)
        assert all(torch.equal(q,w) for q,w in zip(bb,shifted))
    loss=objective(m(x,d,v,True,absolute_depth=d),torch.randn(2,256,3),torch.rand(2,256,3)>.5,v,True);loss.backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in m.parameters())
    assert m.center_strength.grad.abs().sum()>0
    print('PASS: exact initial baseline, nonzero-centering well/padding/depth isolation, learnable finite gradient')
