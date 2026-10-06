"""Per-depth channel LayerNorm replaces nine spatial GroupNorm modules."""
import torch
from torch import nn
from multiscale_sequence import Network as Baseline, GatedBlock
class DepthNorm(nn.LayerNorm):
    def forward(self,x):
        return super().forward(x.transpose(1,2)).transpose(1,2)
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        for module in self.modules():
            if isinstance(module,GatedBlock):
                original=module.norm
                module.norm=DepthNorm(original.num_channels,eps=original.eps)
                module.norm.load_state_dict(original.state_dict())
if __name__=='__main__':
    from multiscale_sequence import objective
    torch.set_num_threads(2);torch.manual_seed(191)
    norm=DepthNorm(64);x=torch.randn(2,64,16);y=x.clone();y[:,:,8:]+=100
    assert torch.equal(norm(x)[:,:,:8],norm(y)[:,:,:8])
    torch.manual_seed(193);base=Baseline(features=12)
    torch.manual_seed(193);model=Network(features=12)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items())
    assert sum(isinstance(m,DepthNorm) for m in model.modules())==9
    model.eval();x=torch.randn(2,256,12);depth=torch.randn(2,256);valid=torch.ones(2,256,dtype=torch.bool);valid[:,:32]=False
    with torch.no_grad():
        expected=model(x,depth,valid)
        changed=x.clone();changed[~valid]+=1000
        assert all(torch.equal(a,b) for a,b in zip(expected,model(changed,depth,valid)))
        changed=x.clone();changed[1]+=100
        assert all(torch.equal(a[0],b[0]) for a,b in zip(expected,model(changed,depth,valid)))
    loss=objective(model(x,depth,valid,True),torch.randn(2,256,3),torch.rand(2,256,3)>.5,valid,True)
    loss.backward();assert all(torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None)
    print('PASS: nine depth-local norms; same initial parameters; input/well isolation; finite gradients')
