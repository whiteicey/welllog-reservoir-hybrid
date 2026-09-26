"""Low-rank multiplicative interactions at the three decoder merges."""
import torch
from torch import nn
from blur_no_dropout_sequence import Network as Baseline
from population_likelihood_sequence import objective
class InteractionMerge(nn.Module):
    def __init__(self,old,deep_channels):
        super().__init__();self.linear=old;self.split=deep_channels
        with torch.random.fork_rng(devices=[]):
            self.deep=nn.Conv1d(deep_channels,16,1)
            self.skip=nn.Conv1d(old.in_channels-deep_channels,16,1)
            self.output=nn.Conv1d(16,old.out_channels,1,bias=False)
        nn.init.zeros_(self.output.weight)
    def forward(self,x):
        a,b=x[:,:self.split],x[:,self.split:]
        return self.linear(x)+self.output(self.deep(a).tanh()*self.skip(b).tanh())
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        for name,deep in [('merge2',192),('merge1',128),('merge0',96)]:
            setattr(self,name,InteractionMerge(getattr(self,name),deep))
if __name__=='__main__':
    torch.set_num_threads(2);torch.manual_seed(42);a=Baseline(features=12).eval()
    torch.manual_seed(42);m=Network(features=12).eval()
    x=torch.randn(2,256,12);d=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:32]=False
    with torch.no_grad():
        aa=a(x,d,v,absolute_depth=d);bb=m(x,d,v,absolute_depth=d)
        assert all(torch.equal(q,w) for q,w in zip(aa,bb))
        for name in ['merge2','merge1','merge0']:getattr(m,name).output.weight.normal_(0,.01)
        bb=m(x,d,v,absolute_depth=d)
        xx=x.clone();xx[1]+=100;cc=m(xx,d,v,absolute_depth=d)
        assert all(torch.equal(q[0],w[0]) for q,w in zip(bb,cc))
        xx=x.clone();xx[~v]+=100;dd=d.clone();dd[~v]+=100;cc=m(xx,dd,v,absolute_depth=dd)
        assert all(torch.equal(q,w) for q,w in zip(bb,cc))
    loss=objective(m(x,d,v,True,absolute_depth=d),torch.randn(2,256,3),torch.rand(2,256,3)>.5,v,True);loss.backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in m.parameters())
    assert m.merge0.deep.weight.grad.abs().sum()>0 and m.merge0.skip.weight.grad.abs().sum()>0
    print('PASS: exact initial baseline, nonzero-interaction isolation, finite branch gradients')
