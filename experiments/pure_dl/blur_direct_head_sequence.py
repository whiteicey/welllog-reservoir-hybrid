"""Attach the mixture head directly to the 64-channel decoder, without combine MLP."""
import torch
from torch import nn
from blur_no_point_sequence import Network as Baseline
from population_likelihood_sequence import objective
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means);self.combine=nn.Identity()
        for name in ['gate','density']:
            old=getattr(self.main_head,name)
            with torch.random.fork_rng(devices=[]):new=nn.Conv1d(64,old.out_channels,1)
            with torch.no_grad():new.weight.copy_(old.weight[:,:64]);new.bias.copy_(old.bias)
            setattr(self.main_head,name,new)
if __name__=='__main__':
    torch.set_num_threads(2);torch.manual_seed(42);a=Baseline(features=12).eval()
    torch.manual_seed(42);m=Network(features=12).eval()
    # Encoder and decoder initial weights are unaffected; only head connection changes.
    assert torch.equal(a.enc0.depthwise.weight,m.enc0.depthwise.weight)
    assert torch.equal(a.main_head.density.bias,m.main_head.density.bias)
    x=torch.randn(2,256,12);d=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:32]=False
    with torch.no_grad():
        bb=m(x,d,v,absolute_depth=d)
        assert bb[0].shape==(2,256,3) and bb[1].shape==(2,256,3,3,8)
        xx=x.clone();xx[1]+=100;cc=m(xx,d,v,absolute_depth=d)
        assert all(torch.equal(q[0],w[0]) for q,w in zip(bb,cc))
        xx=x.clone();xx[~v]+=100;dd=d.clone();dd[~v]+=100;cc=m(xx,dd,v,absolute_depth=dd)
        assert all(torch.equal(q,w) for q,w in zip(bb,cc))
    loss=objective(m(x,d,v,True,absolute_depth=d),torch.randn(2,256,3),torch.rand(2,256,3)>.5,v,True);loss.backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in m.parameters())
    assert not any(k.startswith(('point.','combine.')) for k in m.state_dict())
    print('PASS: unchanged backbone initialization, output shapes, isolation and finite gradients')
