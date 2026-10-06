"""Combine physical depth input with channel-only per-depth normalization."""
import torch
from absolute_depth_sequence import Network as DepthBaseline
from multiscale_sequence import GatedBlock
from depthwise_norm_sequence import DepthNorm
class Network(DepthBaseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        for module in self.modules():
            if isinstance(module,GatedBlock):
                old=module.norm
                module.norm=DepthNorm(old.num_channels,eps=old.eps)
                module.norm.load_state_dict(old.state_dict())
if __name__=='__main__':
    from multiscale_sequence import objective
    torch.set_num_threads(2);torch.manual_seed(211)
    model=Network(features=12).eval()
    assert sum(isinstance(m,DepthNorm) for m in model.modules())==9
    with torch.no_grad():model.stem.weight[:,-1].fill_(.1)
    x=torch.randn(2,256,12);d=torch.randn(2,256);a=torch.randn(2,256);valid=torch.ones(2,256,dtype=torch.bool);valid[:,:32]=False
    with torch.no_grad():
        expected=model(x,d,valid,absolute_depth=a)
        xx=x.clone();aa=a.clone();xx[~valid]+=1000;aa[~valid]+=1000
        assert all(torch.equal(u,v) for u,v in zip(expected,model(xx,d,valid,absolute_depth=aa)))
        xx=x.clone();aa=a.clone();xx[1]+=100;aa[1]+=100
        assert all(torch.equal(u[0],v[0]) for u,v in zip(expected,model(xx,d,valid,absolute_depth=aa)))
    loss=objective(model(x,d,valid,True,absolute_depth=a),torch.randn(2,256,3),torch.rand(2,256,3)>.5,valid,True)
    loss.backward();assert model.stem.weight.grad[:,-1].abs().sum()>0
    assert all(torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None)
    print('PASS: combined shape, nine depth-local norms, nonzero depth-branch padding/well isolation, finite gradients')
