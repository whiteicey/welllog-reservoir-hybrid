import torch
from torch import nn
from torch.nn import functional as F
from depth_norm_joint_sequence import Network as ConvExpert
from gated_point_network import Network as PointExpert
from population_likelihood_sequence import objective

def combine(first,second,router):
    gate=torch.stack([first[0].float(),second[0].float()],-1)
    routing=router.float().log_softmax(-1)
    nondefault=routing+F.logsigmoid(-gate)
    combined_gate=(routing+F.logsigmoid(gate)).logsumexp(-1)-nondefault.logsumexp(-1)
    conditional=nondefault.log_softmax(-1)
    r1=first[1].float();r2=second[1].float()
    logweight=(torch.stack([r1[:,:,:,0].log_softmax(-1),r2[:,:,:,0].log_softmax(-1)],-2)+conditional[...,None]).flatten(-2)
    mean=torch.stack([r1[:,:,:,1],r2[:,:,:,1]],-2).flatten(-2)
    scale=torch.stack([r1[:,:,:,2],r2[:,:,:,2]],-2).flatten(-2)
    return combined_gate,torch.stack([logweight,mean,scale],-2)
class Network(nn.Module):
    def __init__(self,features=206,initial_means=None):
        super().__init__();self.conv=ConvExpert(features,initial_means);self.point=PointExpert(features,initial_means)
        self.router=nn.Sequential(nn.Conv1d(features,32,1),nn.GELU(),nn.Conv1d(32,6,1))
        nn.init.zeros_(self.router[-1].weight);nn.init.zeros_(self.router[-1].bias)
    def forward(self,x,depth,valid,auxiliary=False,absolute_depth=None):
        conv=self.conv(x,depth,valid,auxiliary,absolute_depth=absolute_depth)
        point=self.point(x,depth,valid,auxiliary)
        features=x.transpose(1,2)*valid[:,None]
        def route(f):return self.router(f).transpose(1,2).reshape(len(x),-1,3,2)
        if not auxiliary:return combine(conv,point,route(features))
        def pool(r):return F.avg_pool1d(features,r,r)/F.avg_pool1d(valid[:,None].float(),r,r).clamp_min(1e-6)
        return tuple(combine(c,p,route(f)) for c,p,f in zip(conv,point,[features,pool(2),pool(4)]))
if __name__=='__main__':
    torch.set_num_threads(2);torch.manual_seed(257)
    a=(torch.randn(2,8,3),torch.randn(2,8,3,3,8));b=(torch.randn(2,8,3),torch.randn(2,8,3,3,8));r=torch.randn(2,8,3,2)
    g,raw=combine(a,b,r);q=r.softmax(-1)
    expected=q[...,0]*a[0].sigmoid()+q[...,1]*b[0].sigmoid()
    assert torch.allclose(g.sigmoid(),expected,atol=1e-6)
    assert torch.allclose(raw[:,:,:,0].exp().sum(-1),torch.ones_like(g),atol=1e-6)
    model=Network(features=12);x=torch.randn(2,256,12);d=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool);absolute=torch.randn(2,256)
    y=model(x,d,v,True,absolute_depth=absolute)
    assert y[0][1].shape==(2,256,3,3,16)
    loss=objective(y,torch.randn(2,256,3),torch.rand(2,256,3)>.5,v,True);loss.backward()
    assert all(torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None)
    assert model.router[-1].weight.grad.abs().sum()>0 and model.conv.stem.weight.grad.abs().sum()>0 and model.point.stem.weight.grad.abs().sum()>0
    print('PASS heterogeneous mixture gate/probability normalization, 16-component shape, trainable routing and both experts')
