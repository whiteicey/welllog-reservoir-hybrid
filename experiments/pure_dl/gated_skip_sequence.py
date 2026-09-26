"""Context-conditioned skip gates, initialized to the unchanged baseline."""
import torch
from torch import nn
from torch.nn import functional as F
from depth_norm_joint_sequence import Network as Baseline
from population_likelihood_sequence import objective
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        self.skip_gates=nn.ModuleList([nn.Conv1d(96+64,64,1),nn.Conv1d(128+96,96,1),nn.Conv1d(192+128,128,1)])
        for gate in self.skip_gates:
            nn.init.zeros_(gate.weight);nn.init.zeros_(gate.bias)
    def gated_skip(self,h,skip,gate):
        up=F.interpolate(h,size=skip.shape[-1],mode='linear',align_corners=False)
        weight=2*torch.sigmoid(gate(torch.cat([up,skip],1)))
        return torch.cat([up,skip*weight],1)
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
        h = self.dec2(self.merge2(self.gated_skip(h,e2,self.skip_gates[2])))
        coarse = h
        h = self.dec1(self.merge1(self.gated_skip(h,e1,self.skip_gates[1])))
        mid = h
        h = self.dec0(self.merge0(self.gated_skip(h,e0,self.skip_gates[0])))
        main = self.main_head(self.combine(torch.cat([h, self.point(point)], 1)))
        if not auxiliary:
            return main
        # Loss evaluates every original target against its scale's distribution;
        # it never averages target values into artificial smoothed labels.
        return main, self.aux_mid(mid), self.aux_coarse(coarse)

if __name__=='__main__':
    torch.set_num_threads(2);torch.manual_seed(123);base=Baseline(features=12).eval()
    torch.manual_seed(123);model=Network(features=12).eval()
    x=torch.randn(2,256,12);d=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:32]=False
    with torch.no_grad():
        expected=base(x,d,v,absolute_depth=d)
        actual=model(x,d,v,absolute_depth=d)
        assert all(torch.equal(a,b) for a,b in zip(expected,actual))
        for gate in model.skip_gates:gate.weight.normal_(0,.02)
        actual=model(x,d,v,absolute_depth=d)
        xx=x.clone();xx[~v]+=1000;dd=d.clone();dd[~v]+=1000
        assert all(torch.equal(a,b) for a,b in zip(actual,model(xx,dd,v,absolute_depth=dd)))
        xx=x.clone();xx[1]+=100
        assert all(torch.equal(a[0],b[0]) for a,b in zip(actual,model(xx,d,v,absolute_depth=d)))
    loss=objective(model(x,d,v,True,absolute_depth=d),torch.randn(2,256,3),torch.rand(2,256,3)>.5,v,True)
    loss.backward()
    assert all(g.weight.grad.abs().sum()>0 for g in model.skip_gates)
    assert all(torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None)
    print('PASS: initial baseline equality, nonzero-gate padding/well isolation, all gate gradients finite and nonzero')
