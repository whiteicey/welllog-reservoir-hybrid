"""Zero-initialized full-resolution refinement on the 82.134 baseline."""
import torch
from torch import nn
from torch.nn import functional as F
from blur_no_dropout_sequence import Network as Baseline
from multiscale_sequence import GatedBlock
from depthwise_norm_sequence import DepthNorm
from population_likelihood_sequence import objective
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        self.refine=nn.Sequential(GatedBlock(64,1),GatedBlock(64,2))
        for block in self.refine:
            block.norm=DepthNorm(64);block.dropout.p=0.
            nn.init.zeros_(block.gain)
    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        # features(B,L,206), relative_depth/valid(B,L). Padded inputs are masked.
        mask = valid[:, None].to(features.dtype)
        point = features.transpose(1, 2)*mask
        assert absolute_depth is not None and absolute_depth.shape == valid.shape
        x = torch.cat([point, relative_depth[:, None]*mask, mask, absolute_depth[:, None]*mask], dim=1)
        e0 = self.enc0(self.stem(x))
        e1 = self.enc1(self.down1(e0))
        e2 = self.enc2(self.down2(e1))
        h = self.bottleneck(self.down3(e2))
        h = self.dec2(self.merge2(torch.cat([F.interpolate(h, size=e2.shape[-1], mode='linear', align_corners=False), e2], 1)))
        coarse = h
        h = self.dec1(self.merge1(torch.cat([F.interpolate(h, size=e1.shape[-1], mode='linear', align_corners=False), e1], 1)))
        mid = h
        h = self.dec0(self.merge0(torch.cat([F.interpolate(h, size=e0.shape[-1], mode='linear', align_corners=False), e0], 1)))
        h = self.refine(h)
        main = self.main_head(self.combine(torch.cat([h, self.point(point)], 1)))
        if not auxiliary:
            return main
        # Loss evaluates every original target against its scale's distribution;
        # it never averages target values into artificial smoothed labels.
        return main, self.aux_mid(mid), self.aux_coarse(coarse)

if __name__=='__main__':
    torch.set_num_threads(2);torch.manual_seed(143);base=Baseline(features=12).eval()
    torch.manual_seed(143);model=Network(features=12).eval()
    x=torch.randn(2,256,12);d=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:32]=False
    with torch.no_grad():
        a=base(x,d,v,absolute_depth=d);b=model(x,d,v,absolute_depth=d)
        assert all(torch.equal(aa,bb) for aa,bb in zip(a,b))
        for block in model.refine:block.gain.fill_(.1)
        a=model(x,d,v,absolute_depth=d);xx=x.clone();xx[1]+=100;b=model(xx,d,v,absolute_depth=d);assert all(torch.equal(aa[0],bb[0]) for aa,bb in zip(a,b))
        xx=x.clone();xx[~v]+=100;dd=d.clone();dd[~v]+=100;b=model(xx,dd,v,absolute_depth=dd);assert all(torch.equal(aa,bb) for aa,bb in zip(a,b))
        for block in model.refine:block.gain.zero_()
    loss=objective(model(x,d,v,True,absolute_depth=d),torch.randn(2,256,3),torch.rand(2,256,3)>.5,v,True);loss.backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in model.parameters())
    assert all(b.gain.grad.abs().sum()>0 for b in model.refine)
    print('PASS: initial baseline equality, nonzero-refinement isolation, finite gradients and learnable zero gains')
