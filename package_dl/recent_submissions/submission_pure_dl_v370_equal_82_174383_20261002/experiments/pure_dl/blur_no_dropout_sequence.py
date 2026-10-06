"""Disable existing dropout on the verified three-stage blur model."""
import torch
from torch import nn
from blur_down_sequence import Network as Baseline
from population_likelihood_sequence import objective
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        for module in self.modules():
            if isinstance(module,nn.Dropout):module.p=0.
if __name__=='__main__':
    torch.set_num_threads(2);torch.manual_seed(42);base=Baseline(features=12).eval()
    torch.manual_seed(42);model=Network(features=12)
    assert all(torch.equal(v,base.state_dict()[k]) for k,v in model.state_dict().items())
    drops=[m for m in model.modules() if isinstance(m,nn.Dropout)];assert len(drops)==10 and all(m.p==0 for m in drops)
    x=torch.randn(2,256,12);d=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool)
    with torch.no_grad():
        a=model(x,d,v,absolute_depth=d);b=model(x,d,v,absolute_depth=d)
        assert all(torch.equal(aa,bb) for aa,bb in zip(a,b))
        assert all(torch.equal(aa,bb) for aa,bb in zip(a,base(x,d,v,absolute_depth=d)))
    loss=objective(model(x,d,v,True,absolute_depth=d),torch.randn(2,256,3),torch.rand(2,256,3)>.5,v,True);loss.backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in model.parameters())
    print('PASS: unchanged state and eval outputs, ten dropout modules disabled, deterministic training forward, finite gradients')
