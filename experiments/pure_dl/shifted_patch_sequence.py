"""Shared local attention in ordinary and half-patch-shifted partitions; no circular wrap."""
import torch
from torch import nn
from torch.nn import functional as F
from patch_hierarchy_sequence import Network as Baseline


class ShiftedHierarchy(nn.Module):
    def __init__(self,old):
        super().__init__();self.patch_size=16
        self.norm=old.norm;self.input=old.input;self.local=old.local
        self.global_patches=old.global_patches;self.output=old.output

    def local_mix(self,z,valid):
        batch,length,width=z.shape;patches=length//16
        regular=self.local(z.reshape(batch*patches,16,width),valid.reshape(batch*patches,16)).reshape(batch,length,width)
        padded=F.pad(z,(0,0,8,8));mask=F.pad(valid,(8,8),value=False)
        shifted=self.local(padded.reshape(batch*(patches+1),16,width),mask.reshape(batch*(patches+1),16))
        shifted=shifted.reshape(batch,length+16,width)[:,8:8+length]
        return (z+.5*(regular+shifted)).masked_fill(~valid[...,None],0)

    def forward(self,x,valid):
        batch,channels,length=x.shape;assert length%16==0
        patches=length//16
        z=self.input(self.norm(x.transpose(1,2))).masked_fill(~valid[...,None],0)
        points=self.local_mix(z,valid)
        local=points.reshape(batch*patches,16,32);mask=valid.reshape(batch*patches,16)
        token=(local.sum(1)/mask.sum(1,keepdim=True).clamp_min(1)).reshape(batch,patches,32)
        context=self.global_patches(token,mask.any(1).reshape(batch,patches))
        mixed=points.reshape(batch,patches,16,32)+context[:,:,None,:]
        return self.output(mixed.reshape(batch,length,32)).masked_fill(~valid[...,None],0).transpose(1,2)


class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        self.stem.hierarchy=ShiftedHierarchy(self.stem.hierarchy)


if __name__=='__main__':
    from global_point_loss import objective
    torch.set_num_threads(2);torch.manual_seed(31);base=Baseline(features=12).eval()
    torch.manual_seed(31);model=Network(features=12).eval()
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items())
    x=torch.randn(2,256,12);d=torch.randn(2,256);valid=torch.ones(2,256,dtype=torch.bool);valid[:,:24]=False
    with torch.no_grad():
        assert all(torch.equal(q,w) for q,w in zip(base(x,d,valid,absolute_depth=d),model(x,d,valid,absolute_depth=d)))
        nn.init.normal_(model.stem.hierarchy.output.weight,std=.02)
        expected=model(x,d,valid,absolute_depth=d)
        changed=x.clone();changed[~valid]+=100
        assert all(torch.equal(q,w) for q,w in zip(expected,model(changed,d,valid,absolute_depth=d)))
        changed=x.clone();changed[1]+=100
        assert all(torch.equal(q[0],w[0]) for q,w in zip(expected,model(changed,d,valid,absolute_depth=d)))
    z=torch.randn(1,256,32,requires_grad=True);mask=torch.ones(1,256,dtype=torch.bool)
    local=model.stem.hierarchy.local_mix(z,mask)
    edge_grad=torch.autograd.grad(local[0,15].square().sum(),z,retain_graph=True)[0]
    assert edge_grad[0,16].abs().sum()>0
    end_grad=torch.autograd.grad(local[0,0].square().sum(),z)[0]
    assert (end_grad[0,-8:]==0).all()
    loss=objective(model(x,d,valid,True,absolute_depth=d),torch.randn(2,256,3),torch.rand(2,256,3)>.5,valid,True,torch.ones(2,256));loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print('PASS same initial state/prediction, padding/well isolation, local gradient across patch edge, no circular wrap and finite gradients')
