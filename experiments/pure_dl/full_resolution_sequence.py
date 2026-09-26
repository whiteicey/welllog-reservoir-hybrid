"""Full-resolution dilated depth network; no stride or interpolation."""
import torch
from torch import nn
from multiscale_sequence import GatedBlock,DensityHead
from depthwise_norm_sequence import DepthNorm
from population_likelihood_sequence import objective
class Network(nn.Module):
    def __init__(self,features=206,initial_means=None):
        super().__init__()
        self.stem=nn.Conv1d(features+3,64,1)
        with torch.no_grad():self.stem.weight[:,-1].zero_()
        self.blocks=nn.ModuleList([GatedBlock(64,d) for d in [1,2,4,8,16,32]])
        for block in self.blocks:block.norm=DepthNorm(64)
        self.point=nn.Sequential(nn.Conv1d(features,128,1),nn.GELU(),nn.Conv1d(128,64,1))
        self.combine=nn.Sequential(nn.Conv1d(128,128,1),nn.GELU(),nn.Dropout(.1))
        self.mid_project=nn.Conv1d(64,96,1);self.coarse_project=nn.Conv1d(64,128,1)
        self.main_head=DensityHead(128,initial_means);self.aux_mid=DensityHead(96,initial_means);self.aux_coarse=DensityHead(128,initial_means)
    def forward(self,features,relative_depth,valid,auxiliary=False,absolute_depth=None):
        assert absolute_depth is not None
        mask=valid[:,None].to(features.dtype)
        point=features.transpose(1,2)*mask
        h=self.stem(torch.cat([point,relative_depth[:,None]*mask,mask,absolute_depth[:,None]*mask],1))
        for i,block in enumerate(self.blocks):
            h=block(h)
            if i==1:mid=h
            if i==3:coarse=h
        main=self.main_head(self.combine(torch.cat([h,self.point(point)],1)))
        if not auxiliary:return main
        return main,self.aux_mid(self.mid_project(mid)),self.aux_coarse(self.coarse_project(coarse))
if __name__=='__main__':
    torch.set_num_threads(2);model=Network(features=12).eval()
    x=torch.randn(2,256,12);d=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:32]=False
    with torch.no_grad():
        model.stem.weight[:,-1].fill_(.1)
        actual=model(x,d,v,absolute_depth=d)
        xx=x.clone();xx[~v]+=1000;dd=d.clone();dd[~v]+=1000
        assert all(torch.equal(a,b) for a,b in zip(actual,model(xx,dd,v,absolute_depth=dd)))
        xx=x.clone();xx[1]+=100
        assert all(torch.equal(a[0],b[0]) for a,b in zip(actual,model(xx,d,v,absolute_depth=d)))
    out=model(x,d,v,True,absolute_depth=d)
    assert all(o[1].shape==(2,256,3,3,8) for o in out)
    loss=objective(out,torch.randn(2,256,3),torch.rand(2,256,3)>.5,v,True);loss.backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in model.parameters())
    print('PASS: full-resolution three heads, padded/well isolation, all parameters finite gradients')
