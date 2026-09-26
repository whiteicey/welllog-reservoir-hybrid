"""Mix input channels before depthwise filtering within all nine residual blocks."""
import torch
from torch import nn
from torch.nn import functional as F
from depth_norm_joint_sequence import Network as Baseline
from depthwise_norm_sequence import DepthNorm
from population_likelihood_sequence import objective
class ChannelFirstBlock(nn.Module):
    def __init__(self,channels,dilation):
        super().__init__()
        self.norm=DepthNorm(channels)
        self.expand=nn.Conv1d(channels,2*channels,1)
        self.depthwise=nn.Conv1d(2*channels,2*channels,7,padding=3*dilation,dilation=dilation,groups=2*channels)
        self.project=nn.Conv1d(channels,channels,1)
        self.dropout=nn.Dropout(.1)
        self.gain=nn.Parameter(torch.full((1,channels,1),.1))
    def forward(self,x):
        value,gate=self.depthwise(self.expand(self.norm(x))).chunk(2,1)
        return x+self.gain*self.project(self.dropout(F.gelu(value)*gate.sigmoid()))
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        for name in ['enc0','enc1','enc2','dec2','dec1','dec0']:
            old=getattr(self,name);setattr(self,name,ChannelFirstBlock(old.project.in_channels,old.depthwise.dilation[0]))
        for i,old in enumerate(self.bottleneck):self.bottleneck[i]=ChannelFirstBlock(old.project.in_channels,old.depthwise.dilation[0])
if __name__=='__main__':
    torch.set_num_threads(2);model=Network(features=12).eval()
    assert sum(isinstance(m,ChannelFirstBlock) for m in model.modules())==9
    x=torch.randn(2,256,12);d=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:32]=False
    with torch.no_grad():
        model.stem.weight[:,-1].fill_(.1)
        a=model(x,d,v,absolute_depth=d);xx=x.clone();xx[1]+=10;b=model(xx,d,v,absolute_depth=d)
        assert all(torch.equal(aa[0],bb[0]) for aa,bb in zip(a,b))
        xx=x.clone();xx[~v]+=1000;dd=d.clone();dd[~v]+=1000;b=model(xx,dd,v,absolute_depth=dd)
        assert all(torch.equal(aa,bb) for aa,bb in zip(a,b))
    loss=objective(model(x,d,v,True,absolute_depth=d),torch.randn(2,256,3),torch.rand(2,256,3)>.5,v,True);loss.backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in model.parameters())
    print('PASS: nine channel-first blocks, padded/well isolation, all parameters finite backward')
