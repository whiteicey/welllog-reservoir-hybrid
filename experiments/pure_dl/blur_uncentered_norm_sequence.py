"""Per-depth channel RMS scaling with the existing learned scale and bias."""
import torch
from torch import nn
from blur_no_dropout_sequence import Network as Baseline
from depthwise_norm_sequence import DepthNorm
from population_likelihood_sequence import objective
class UncenteredNorm(nn.Module):
    def __init__(self,old):
        super().__init__();self.eps=old.eps
        self.weight=nn.Parameter(old.weight.detach().clone());self.bias=nn.Parameter(old.bias.detach().clone())
    def forward(self,x):
        value=x.float();scaled=value*torch.rsqrt(value.square().mean(dim=1,keepdim=True)+self.eps)
        return (scaled*self.weight[None,:,None]+self.bias[None,:,None]).to(x.dtype)
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        for block in self.modules():
            if hasattr(block,'norm') and isinstance(block.norm,DepthNorm):block.norm=UncenteredNorm(block.norm)
if __name__=='__main__':
    torch.set_num_threads(2);n=UncenteredNorm(DepthNorm(4));x=torch.randn(2,4,16)
    expected=x/torch.sqrt(x.square().mean(1,keepdim=True)+n.eps)
    assert torch.allclose(n(x),expected,atol=1e-7)
    changed=x.clone();changed[:,:,8:]+=10;assert torch.equal(n(x)[:,:,:8],n(changed)[:,:,:8])
    assert not torch.allclose(n(x),n(x+10))
    model=Network(features=12).eval();assert sum(isinstance(m,UncenteredNorm) for m in model.modules())==9
    x=torch.randn(2,256,12);d=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:32]=False
    with torch.no_grad():
        model.stem.weight[:,-1].fill_(.1);a=model(x,d,v,absolute_depth=d)
        xx=x.clone();xx[1]+=100;b=model(xx,d,v,absolute_depth=d);assert all(torch.equal(aa[0],bb[0]) for aa,bb in zip(a,b))
        xx=x.clone();xx[~v]+=100;dd=d.clone();dd[~v]+=100;b=model(xx,dd,v,absolute_depth=dd);assert all(torch.equal(aa,bb) for aa,bb in zip(a,b))
    loss=objective(model(x,d,v,True,absolute_depth=d),torch.randn(2,256,3),torch.rand(2,256,3)>.5,v,True);loss.backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in model.parameters())
    print('PASS: RMS formula, depth-locality, offset sensitivity, nine norms, padding/well isolation, finite gradients')
