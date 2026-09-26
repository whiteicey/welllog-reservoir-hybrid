"""Input-conditioned bounded sampling offsets in the first depthwise convolution."""
import torch
from torch import nn
from torch.nn import functional as F
from blur_no_dropout_sequence import Network as Baseline
from population_likelihood_sequence import objective
class OffsetDepthwise(nn.Module):
    def __init__(self,old):
        super().__init__();self.weight=old.weight;self.bias=old.bias
        self.dilation=old.dilation[0];self.k=old.kernel_size[0]
        self.offset=nn.Conv1d(old.in_channels,self.k,1)
        nn.init.zeros_(self.offset.weight);nn.init.zeros_(self.offset.bias)
    def forward(self,x):
        b,c,l=x.shape;shift=2*self.offset(x).tanh()
        with torch.autocast(device_type=x.device.type,enabled=False):
            positions=torch.arange(l,device=x.device,dtype=torch.float32)[None,None,:]+(torch.arange(self.k,device=x.device,dtype=torch.float32)-self.k//2)[None,:,None]*self.dilation+shift.float()
            gx=2*positions.reshape(b,1,self.k*l)/(l-1)-1
            grid=torch.stack([gx,torch.zeros_like(gx)],-1)
            sampled=F.grid_sample(x.float()[:,:,None,:],grid,mode='bilinear',padding_mode='zeros',align_corners=True).reshape(b,c,self.k,l)
            result=(sampled*self.weight.float().squeeze(1)[None,:,:,None]).sum(2)+self.bias.float()[None,:,None]
        return result.to(x.dtype)
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means);self.enc0.depthwise=OffsetDepthwise(self.enc0.depthwise)
if __name__=='__main__':
    torch.set_num_threads(2);old=nn.Conv1d(4,4,7,padding=3,groups=4);new=OffsetDepthwise(old);x=torch.randn(2,4,64)
    assert torch.allclose(old(x),new(x),atol=1e-5,rtol=1e-5)
    model=Network(features=12).eval();x=torch.randn(2,256,12);d=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:32]=False
    with torch.no_grad():
        model.enc0.depthwise.offset.weight.normal_(0,.01)
        model.stem.weight[:,-1].fill_(.1);a=model(x,d,v,absolute_depth=d)
        xx=x.clone();xx[1]+=100;b=model(xx,d,v,absolute_depth=d);assert all(torch.equal(aa[0],bb[0]) for aa,bb in zip(a,b))
        xx=x.clone();xx[~v]+=100;dd=d.clone();dd[~v]+=100;b=model(xx,dd,v,absolute_depth=dd);assert all(torch.equal(aa,bb) for aa,bb in zip(a,b))
    loss=objective(model(x,d,v,True,absolute_depth=d),torch.randn(2,256,3),torch.rand(2,256,3)>.5,v,True);loss.backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in model.parameters())
    assert model.enc0.depthwise.offset.weight.grad.abs().sum()>0
    print('PASS: zero-offset convolution equivalence, nonzero-offset padding/well isolation, finite learnable offset gradients')
