"""Low-rank multiplicative feature interactions alongside the existing point MLP."""
import torch
from torch import nn
from depth_norm_joint_sequence import Network as Baseline
class InteractionPoint(nn.Module):
    def __init__(self,original,features):
        super().__init__();self.original=original
        self.left=nn.Conv1d(features,64,1);self.right=nn.Conv1d(features,64,1)
        self.project=nn.Conv1d(64,64,1);self.gain=nn.Parameter(torch.full((1,64,1),.1))
    def forward(self,x):
        interaction=torch.tanh(self.left(x))*torch.tanh(self.right(x))
        return self.original(x)+self.gain*self.project(interaction)
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        self.point=InteractionPoint(self.point,features)
if __name__=='__main__':
    from multiscale_sequence import objective
    torch.set_num_threads(2);torch.manual_seed(223);base=Baseline(features=12).eval()
    torch.manual_seed(223);model=Network(features=12).eval()
    x=torch.randn(2,256,12);d=torch.randn(2,256);a=torch.randn(2,256);valid=torch.ones(2,256,dtype=torch.bool);valid[:,:32]=False
    with torch.no_grad():
        model.point.gain.zero_()
        assert all(torch.equal(u,v) for u,v in zip(base(x,d,valid,absolute_depth=a),model(x,d,valid,absolute_depth=a)))
        model.point.gain.fill_(.1)
        expected=model(x,d,valid,absolute_depth=a)
        xx=x.clone();xx[~valid]+=1000
        assert all(torch.equal(u,v) for u,v in zip(expected,model(xx,d,valid,absolute_depth=a)))
        xx=x.clone();xx[1]+=100
        assert all(torch.equal(u[0],v[0]) for u,v in zip(expected,model(xx,d,valid,absolute_depth=a)))
    loss=objective(model(x,d,valid,True,absolute_depth=a),torch.randn(2,256,3),torch.rand(2,256,3)>.5,valid,True)
    loss.backward();assert all(torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None)
    assert model.point.left.weight.grad.abs().sum()>0 and model.point.right.weight.grad.abs().sum()>0
    print('PASS: zero-gain baseline equivalence; padding/well isolation; both interaction factors receive finite gradients')
