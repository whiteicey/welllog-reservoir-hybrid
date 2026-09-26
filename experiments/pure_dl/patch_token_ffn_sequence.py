"""A small residual token FFN after global patch attention; preserve32-wide baseline."""
import torch
from torch import nn
from patch_hierarchy_sequence import Network as Baseline


class TokenFFNHierarchy(nn.Module):
    def __init__(self,old):
        super().__init__();self.patch_size=16
        self.norm=old.norm;self.input=old.input;self.local=old.local
        self.global_patches=old.global_patches;self.output=old.output
        self.token_ffn=nn.Sequential(nn.LayerNorm(32),nn.Linear(32,64),nn.GELU(),nn.Linear(64,32))
        nn.init.zeros_(self.token_ffn[-1].weight);nn.init.zeros_(self.token_ffn[-1].bias)

    def forward(self,x,valid):
        batch,channels,length=x.shape;patches=length//16
        z=self.input(self.norm(x.transpose(1,2))).masked_fill(~valid[...,None],0)
        local=z.reshape(batch*patches,16,32);mask=valid.reshape(batch*patches,16)
        local=(local+self.local(local,mask)).masked_fill(~mask[...,None],0)
        token=(local.sum(1)/mask.sum(1,keepdim=True).clamp_min(1)).reshape(batch,patches,32)
        token_valid=mask.any(1).reshape(batch,patches)
        context=self.global_patches(token,token_valid)
        context=context+self.token_ffn(token+context).masked_fill(~token_valid[...,None],0)
        mixed=local.reshape(batch,patches,16,32)+context[:,:,None,:]
        return self.output(mixed.reshape(batch,length,32)).masked_fill(~valid[...,None],0).transpose(1,2)


class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        with torch.random.fork_rng(devices=[]):self.stem.hierarchy=TokenFFNHierarchy(self.stem.hierarchy)


if __name__=='__main__':
    from global_point_loss import objective
    torch.set_num_threads(2);torch.manual_seed(19);a=Baseline(features=12).eval()
    torch.manual_seed(19);m=Network(features=12).eval()
    x=torch.randn(2,256,12);d=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:32]=False
    with torch.no_grad():
        # Also check identity when the outer projection is active, so FFN-zero behavior is covered.
        nn.init.normal_(a.stem.hierarchy.output.weight,std=.02)
        m.stem.hierarchy.output.weight.copy_(a.stem.hierarchy.output.weight)
        assert all(torch.equal(q,w) for q,w in zip(a(x,d,v,absolute_depth=d),m(x,d,v,absolute_depth=d)))
        nn.init.normal_(m.stem.hierarchy.token_ffn[-1].weight,std=.02)
        expected=m(x,d,v,absolute_depth=d)
        changed=x.clone();changed[~v]+=100
        assert all(torch.equal(q,w) for q,w in zip(expected,m(changed,d,v,absolute_depth=d)))
        changed=x.clone();changed[1]+=100
        assert all(torch.equal(q[0],w[0]) for q,w in zip(expected,m(changed,d,v,absolute_depth=d)))
    loss=objective(m(x,d,v,True,absolute_depth=d),torch.randn(2,256,3),torch.rand(2,256,3)>.5,v,True,torch.ones(2,256));loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert m.stem.hierarchy.token_ffn[1].weight.grad.abs().sum()>0
    print('PASS zero-FFN identity with active outer branch, padding/well isolation and live finite FFN gradients')
