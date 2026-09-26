"""Expand nine depthwise kernels from seven to fifteen using zero-padded initialization."""
import torch
from torch import nn
from blur_no_dropout_sequence import Network as Baseline
from population_likelihood_sequence import objective
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        count=0
        for block in list(self.modules()):
            old=getattr(block,'depthwise',None)
            if isinstance(old,nn.Conv1d):
                assert old.kernel_size==(7,) and old.groups==old.in_channels
                # Preserve all other random initialization and exact original central taps.
                with torch.random.fork_rng(devices=[]):
                    new=nn.Conv1d(old.in_channels,old.out_channels,15,padding=7*old.dilation[0],dilation=old.dilation,groups=old.groups)
                with torch.no_grad():
                    new.weight.zero_();new.weight[:,:,4:11].copy_(old.weight);new.bias.copy_(old.bias)
                block.depthwise=new;count+=1
        assert count==9
if __name__=='__main__':
    torch.set_num_threads(2);torch.manual_seed(42);a=Baseline(features=12).eval()
    torch.manual_seed(42);m=Network(features=12).eval()
    x=torch.randn(2,256,12);d=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:32]=False
    with torch.no_grad():
        aa=a(x,d,v,absolute_depth=d);bb=m(x,d,v,absolute_depth=d)
        assert all(torch.allclose(q,w,atol=2e-5,rtol=2e-5) for q,w in zip(aa,bb))
        xx=x.clone();xx[1]+=100;cc=m(xx,d,v,absolute_depth=d)
        assert all(torch.equal(q[0],w[0]) for q,w in zip(bb,cc))
        xx=x.clone();xx[~v]+=100;dd=d.clone();dd[~v]+=100;cc=m(xx,dd,v,absolute_depth=dd)
        assert all(torch.equal(q,w) for q,w in zip(bb,cc))
    loss=objective(m(x,d,v,True,absolute_depth=d),torch.randn(2,256,3),torch.rand(2,256,3)>.5,v,True);loss.backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in m.parameters())
    assert m.enc0.depthwise.weight.grad[:,:,:4].abs().sum()>0
    print('PASS: baseline-equivalent initialization, well/padding isolation, finite outer-tap gradients')
