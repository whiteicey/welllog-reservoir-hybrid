"""Keep the two-level patch layout, double attention representation width to64."""
import torch
from torch import nn
from patch_hierarchy_sequence import Network as Baseline, SmallAttention


class WideHierarchy(nn.Module):
    def __init__(self):
        super().__init__();self.patch_size=16
        self.norm=nn.LayerNorm(64);self.input=nn.Linear(64,64,bias=False)
        self.local=SmallAttention(width=64,heads=4)
        self.global_patches=SmallAttention(width=64,heads=4)
        self.output=nn.Linear(64,64,bias=False);nn.init.zeros_(self.output.weight)

    def forward(self,x,valid):
        batch,channels,length=x.shape;assert length%16==0
        patches=length//16
        z=self.input(self.norm(x.transpose(1,2))).masked_fill(~valid[...,None],0)
        local=z.reshape(batch*patches,16,64);mask=valid.reshape(batch*patches,16)
        local=(local+self.local(local,mask)).masked_fill(~mask[...,None],0)
        token=(local.sum(1)/mask.sum(1,keepdim=True).clamp_min(1)).reshape(batch,patches,64)
        context=self.global_patches(token,mask.any(1).reshape(batch,patches))
        mixed=local.reshape(batch,patches,16,64)+context[:,:,None,:]
        return self.output(mixed.reshape(batch,length,64)).masked_fill(~valid[...,None],0).transpose(1,2)


class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        with torch.random.fork_rng(devices=[]):self.stem.hierarchy=WideHierarchy()


if __name__=='__main__':
    from global_point_loss import objective
    torch.set_num_threads(2);torch.manual_seed(42);a=Baseline(features=12).eval()
    torch.manual_seed(42);m=Network(features=12).eval()
    x=torch.randn(2,256,12);d=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:32]=False
    with torch.no_grad():
        assert all(torch.equal(q,w) for q,w in zip(a(x,d,v,absolute_depth=d),m(x,d,v,absolute_depth=d)))
        nn.init.normal_(m.stem.hierarchy.output.weight,std=.02)
        expected=m(x,d,v,absolute_depth=d)
        padded=x.clone();padded[~v]+=100
        assert all(torch.equal(q,w) for q,w in zip(expected,m(padded,d,v,absolute_depth=d)))
        other=x.clone();other[1]+=100
        assert all(torch.equal(q[0],w[0]) for q,w in zip(expected,m(other,d,v,absolute_depth=d)))
        assert torch.isfinite(m.stem.hierarchy(torch.randn(2,64,256),torch.zeros_like(v))).all()
    loss=objective(m(x,d,v,True,absolute_depth=d),torch.randn(2,256,3),torch.rand(2,256,3)>.5,v,True,torch.ones(2,256));loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    print('PASS initial identity, padding/well isolation, empty patches, finite gradients')
    print('PARAMETERS',sum(p.numel() for p in Network().parameters()))
