"""Add training-fold-scaled physical depth alongside existing relative depth."""
import torch
from torch import nn
from torch.nn import functional as F
from multiscale_sequence import Network as Baseline
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        old=self.stem
        self.stem=nn.Conv1d(features+3,64,1)
        with torch.no_grad():
            self.stem.weight[:,:-1].copy_(old.weight)
            self.stem.weight[:,-1].zero_()
            self.stem.bias.copy_(old.bias)
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
        main = self.main_head(self.combine(torch.cat([h, self.point(point)], 1)))
        if not auxiliary:
            return main
        # Loss evaluates every original target against its scale's distribution;
        # it never averages target values into artificial smoothed labels.
        return main, self.aux_mid(mid), self.aux_coarse(coarse)

if __name__=='__main__':
    from multiscale_sequence import objective
    torch.set_num_threads(2);torch.manual_seed(197);base=Baseline(features=12).eval()
    torch.manual_seed(197);model=Network(features=12).eval()
    x=torch.randn(2,256,12);d=torch.randn(2,256);a=torch.randn(2,256);valid=torch.ones(2,256,dtype=torch.bool);valid[:,:32]=False
    with torch.no_grad():
        expected=model(x,d,valid,absolute_depth=a)
        assert all(torch.equal(u,v) for u,v in zip(base(x,d,valid),expected))
        changed=a.clone();changed[~valid]+=1000
        assert all(torch.equal(u,v) for u,v in zip(expected,model(x,d,valid,absolute_depth=changed)))
    loss=objective(model(x,d,valid,True,absolute_depth=a),torch.randn(2,256,3),torch.rand(2,256,3)>.5,valid,True)
    loss.backward();assert model.stem.weight.grad[:,-1].abs().sum()>0
    assert all(torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None)
    print('PASS: zero-init depth branch matches baseline, padded depth isolated, new depth weights receive finite gradients')
