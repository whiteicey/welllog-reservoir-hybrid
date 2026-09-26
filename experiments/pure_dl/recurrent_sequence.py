"""Small bidirectional recurrent context at the 32-position bottleneck."""
import torch
from torch import nn
from multiscale_sequence import Network as Baseline
class RecurrentContext(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(192)
        self.gru=nn.GRU(192,64,batch_first=True,bidirectional=True)
        self.project=nn.Linear(128,192)
        self.gain=nn.Parameter(torch.full((1,192,1),.1))
    def forward(self,x):
        with torch.autocast(device_type=x.device.type,enabled=False):
            y,_=self.gru(self.norm(x.float().transpose(1,2)))
            y=self.project(y).transpose(1,2)
        return x+self.gain*y.to(x.dtype)
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        self.bottleneck.add_module('recurrent',RecurrentContext())
if __name__=='__main__':
    from multiscale_sequence import objective
    torch.set_num_threads(2);torch.manual_seed(199);base=Baseline(features=12).eval()
    torch.manual_seed(199);model=Network(features=12).eval()
    x=torch.randn(2,256,12);depth=torch.randn(2,256);valid=torch.ones(2,256,dtype=torch.bool);valid[:,:32]=False
    with torch.no_grad():
        model.bottleneck.recurrent.gain.zero_()
        assert all(torch.equal(a,b) for a,b in zip(base(x,depth,valid),model(x,depth,valid)))
        model.bottleneck.recurrent.gain.fill_(.1)
        expected=model(x,depth,valid)
        changed=x.clone();changed[~valid]+=1000
        assert all(torch.equal(a,b) for a,b in zip(expected,model(changed,depth,valid)))
        changed=x.clone();changed[1]+=100
        assert all(torch.equal(a[0],b[0]) for a,b in zip(expected,model(changed,depth,valid)))
    loss=objective(model(x,depth,valid,True),torch.randn(2,256,3),torch.rand(2,256,3)>.5,valid,True)
    loss.backward()
    assert all(torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None)
    assert model.bottleneck.recurrent.gru.weight_ih_l0.grad.abs().sum()>0
    assert model.bottleneck.recurrent.gru.weight_ih_l0_reverse.grad.abs().sum()>0
    print('PASS: zero-gain baseline equivalence; padding/well isolation; finite gradients through both directions')
