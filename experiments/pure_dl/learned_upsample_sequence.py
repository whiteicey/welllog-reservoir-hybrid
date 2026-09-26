import torch
from torch import nn
from torch.nn import functional as F
from depth_norm_joint_sequence import Network as Baseline
from population_likelihood_sequence import objective
class LearnedUpsample(nn.Module):
    def __init__(self,channels):
        super().__init__();self.residual=nn.ConvTranspose1d(channels,channels,4,stride=2,padding=1,groups=channels,bias=False)
        nn.init.zeros_(self.residual.weight)
    def forward(self,x):
        return F.interpolate(x,scale_factor=2,mode='linear',align_corners=False)+self.residual(x)
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        self.up2=LearnedUpsample(192);self.up1=LearnedUpsample(128);self.up0=LearnedUpsample(96)
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
        h = self.dec2(self.merge2(torch.cat([self.up2(h), e2], 1)))
        coarse = h
        h = self.dec1(self.merge1(torch.cat([self.up1(h), e1], 1)))
        mid = h
        h = self.dec0(self.merge0(torch.cat([self.up0(h), e0], 1)))
        main = self.main_head(self.combine(torch.cat([h, self.point(point)], 1)))
        if not auxiliary:
            return main
        # Loss evaluates every original target against its scale's distribution;
        # it never averages target values into artificial smoothed labels.
        return main, self.aux_mid(mid), self.aux_coarse(coarse)

if __name__=='__main__':
    torch.set_num_threads(2);torch.manual_seed(283);base=Baseline(features=12).eval()
    torch.manual_seed(283);model=Network(features=12).eval()
    x=torch.randn(2,256,12);d=torch.randn(2,256);a=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:32]=False
    with torch.no_grad():
        expected=model(x,d,v,absolute_depth=a)
        assert all(torch.equal(u,w) for u,w in zip(base(x,d,v,absolute_depth=a),expected))
        changed=x.clone();changed[~v]+=1000
        assert all(torch.equal(u,w) for u,w in zip(expected,model(changed,d,v,absolute_depth=a)))
        changed=x.clone();changed[1]+=100
        assert all(torch.equal(u[0],w[0]) for u,w in zip(expected,model(changed,d,v,absolute_depth=a)))
    outputs=model(x,d,v,True,absolute_depth=a)
    loss=objective(outputs,torch.randn(2,256,3),torch.rand(2,256,3)>.5,v,True);loss.backward()
    assert all(torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None)
    for layer in [model.up2,model.up1,model.up0]:assert layer.residual.weight.grad.abs().sum()>0
    print('PASS exact initial baseline, padding/well isolation, all three learnable resampling gradients')
