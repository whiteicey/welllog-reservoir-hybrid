"""Replace learned sigmoid gates with constant 0.5 and remove unused gate projections."""
import torch
from torch import nn
from torch.nn import functional as F
from blur_no_point_sequence import Network as Baseline
from multiscale_sequence import GatedBlock
from population_likelihood_sequence import objective
class UngatedBlock(nn.Module):
    def __init__(self,old):
        super().__init__();self.norm=old.norm;self.depthwise=old.depthwise;self.project=old.project;self.dropout=old.dropout;self.gain=old.gain
        c=old.expand.in_channels
        with torch.random.fork_rng(devices=[]):self.expand=nn.Conv1d(c,c,1)
        with torch.no_grad():self.expand.weight.copy_(old.expand.weight[:c]);self.expand.bias.copy_(old.expand.bias[:c])
    def forward(self,x):
        return x+self.gain*self.project(self.dropout(.5*F.gelu(self.expand(self.depthwise(self.norm(x))))))
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means);count=0
        for parent in list(self.modules()):
            for name,child in list(parent.named_children()):
                if isinstance(child,GatedBlock):setattr(parent,name,UngatedBlock(child));count+=1
        assert count==9
if __name__=='__main__':
    torch.set_num_threads(2);torch.manual_seed(42);a=Baseline(features=12).eval()
    torch.manual_seed(42);m=Network(features=12).eval()
    with torch.no_grad():
        for block in a.modules():
            if isinstance(block,GatedBlock):
                c=block.expand.in_channels;block.expand.weight[c:].zero_();block.expand.bias[c:].zero_()
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
    print('PASS: constant-half-gate equivalence, well/padding isolation, finite gradients')
