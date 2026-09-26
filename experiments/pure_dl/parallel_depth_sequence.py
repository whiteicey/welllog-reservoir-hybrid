"""Add a zero-initialized wider depthwise branch in each residual block."""
import torch
from torch import nn
from relative_only_sequence import Network as Baseline
from blur_ungated_sequence import UngatedBlock


class ParallelDepthwise(nn.Module):
    def __init__(self, local):
        super().__init__()
        self.local = local
        c, d = local.in_channels, local.dilation[0]
        with torch.random.fork_rng(devices=[]):
            self.wide = nn.Conv1d(c, c, 7, padding=6*d, dilation=2*d, groups=c, bias=False)
        nn.init.zeros_(self.wide.weight)

    def forward(self, x):
        return self.local(x) + self.wide(x)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None):
        super().__init__(features, initial_means)
        blocks = [m for m in self.modules() if isinstance(m, UngatedBlock)]
        assert len(blocks) == 9
        for m in blocks:
            m.depthwise = ParallelDepthwise(m.depthwise)


if __name__ == '__main__':
    from global_point_loss import objective
    torch.set_num_threads(2)
    torch.manual_seed(42); a = Baseline(features=12).eval()
    torch.manual_seed(42); b = Network(features=12).eval()
    x = torch.randn(2,256,12); d = torch.randn(2,256)
    v = torch.ones(2,256,dtype=torch.bool); v[:,:32] = False
    with torch.no_grad():
        aa = a(x,d,v,absolute_depth=d); bb = b(x,d,v,absolute_depth=d)
        assert all(torch.equal(q,w) for q,w in zip(aa,bb))
        altered=x.clone();altered[1]+=100
        assert all(torch.equal(q[0],w[0]) for q,w in zip(bb,b(altered,d,v,absolute_depth=d)))
        altered=x.clone();altered[~v]+=100
        assert all(torch.equal(q,w) for q,w in zip(bb,b(altered,d,v,absolute_depth=d)))
    loss=objective(b(x,d,v,True,absolute_depth=d),torch.randn(2,256,3),
                   torch.rand(2,256,3)>.5,v,True,torch.ones(2,256))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in b.parameters())
    assert all(m.wide.weight.grad.abs().sum()>0 for m in b.modules() if isinstance(m,ParallelDepthwise))
    print('PASS: exact initial identity, well/padding isolation, new branches receive finite nonzero gradients')
    print('PARAMETERS',sum(p.numel() for p in Network().parameters()))
