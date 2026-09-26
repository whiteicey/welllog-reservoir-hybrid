import torch
from torch import nn
from torch.nn import functional as F
from depth_norm_joint_sequence import Network as Baseline
from population_likelihood_sequence import objective
CURVES=['GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT']
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        self.context=nn.Sequential(nn.Linear(27,64),nn.GELU(),nn.Linear(64,128))
        nn.init.zeros_(self.context[-1].weight);nn.init.zeros_(self.context[-1].bias)
    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None, well_context=None):
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
        assert well_context is not None and well_context.shape == (len(features),27)
        gamma,beta=self.context(well_context.float()).chunk(2,-1)
        point_hidden=self.point(point)
        point_hidden=point_hidden*(1+.1*gamma.tanh()[:,:,None])+.1*beta[:,:,None]
        main = self.main_head(self.combine(torch.cat([h, point_hidden], 1)))
        if not auxiliary:
            return main
        # Loss evaluates every original target against its scale's distribution;
        # it never averages target values into artificial smoothed labels.
        return main, self.aux_mid(mid), self.aux_coarse(coarse)

if __name__=='__main__':
    torch.set_num_threads(2);torch.manual_seed(239);base=Baseline(features=12).eval()
    torch.manual_seed(239);model=Network(features=12).eval()
    x=torch.randn(2,256,12);d=torch.randn(2,256);a=torch.randn(2,256);valid=torch.ones(2,256,dtype=torch.bool);valid[:,:32]=False;c=torch.randn(2,27)
    with torch.no_grad():
        assert all(torch.equal(u,v) for u,v in zip(base(x,d,valid,absolute_depth=a),model(x,d,valid,absolute_depth=a,well_context=c)))
        model.context[-1].weight.normal_(0,.01)
        before=model(x,d,valid,absolute_depth=a,well_context=c)
        cc=c.clone();cc[1]+=100
        assert all(torch.equal(u[0],v[0]) for u,v in zip(before,model(x,d,valid,absolute_depth=a,well_context=cc)))
    loss=objective(model(x,d,valid,True,absolute_depth=a,well_context=c),torch.randn(2,256,3),torch.rand(2,256,3)>.5,valid,True)
    loss.backward();assert model.context[0].weight.grad.abs().sum()>0
    assert all(torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None)
    print('PASS: zero-init baseline equivalence, context cross-well isolation, finite context gradients')
