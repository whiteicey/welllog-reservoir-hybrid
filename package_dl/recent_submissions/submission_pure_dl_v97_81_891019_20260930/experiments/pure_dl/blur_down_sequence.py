"""Fixed low-pass filter before each stride-two convolution."""
import torch
from torch import nn
from torch.nn import functional as F
from depth_norm_joint_sequence import Network as Baseline
from population_likelihood_sequence import objective
class BlurDown(nn.Module):
    def __init__(self,conv):
        super().__init__();self.conv=conv
        self.register_buffer('kernel',torch.tensor([.25,.5,.25]).reshape(1,1,3))
    def smooth(self,x):
        return F.conv1d(F.pad(x,(1,1),mode='replicate'),self.kernel.to(x.dtype).expand(x.shape[1],1,3),groups=x.shape[1])
    def forward(self,x):return self.conv(self.smooth(x))
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        for name in ['down1','down2','down3']:setattr(self,name,BlurDown(getattr(self,name)))
if __name__=='__main__':
    torch.set_num_threads(2);layer=BlurDown(nn.Conv1d(4,8,4,stride=2,padding=1));x=torch.randn(2,4,32)
    xx=F.pad(x,(1,1),mode='replicate');expected=.25*xx[:,:,:-2]+.5*xx[:,:,1:-1]+.25*xx[:,:,2:]
    assert torch.allclose(layer.smooth(x),expected,atol=1e-7)
    assert torch.equal(layer.smooth(torch.ones_like(x)),torch.ones_like(x))
    torch.manual_seed(12);base=Baseline(features=12);torch.manual_seed(12);model=Network(features=12).eval()
    assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in base.parameters())
    assert torch.equal(base.down1.weight,model.down1.conv.weight)
    x=torch.randn(2,256,12);d=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:32]=False
    with torch.no_grad():
        model.stem.weight[:,-1].fill_(.1);a=model(x,d,v,absolute_depth=d)
        xx=x.clone();xx[1]+=100;b=model(xx,d,v,absolute_depth=d);assert all(torch.equal(aa[0],bb[0]) for aa,bb in zip(a,b))
        xx=x.clone();xx[~v]+=100;dd=d.clone();dd[~v]+=100;b=model(xx,dd,v,absolute_depth=dd);assert all(torch.equal(aa,bb) for aa,bb in zip(a,b))
    loss=objective(model(x,d,v,True,absolute_depth=d),torch.randn(2,256,3),torch.rand(2,256,3)>.5,v,True);loss.backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in model.parameters())
    print('PASS: filter formula, constant preservation, unchanged trainable initialization/count, isolation and gradients')
