"""Auxiliary SW conditional-mean regression; original mixture likelihood retained."""
import torch
from torch.nn import functional as F
from population_likelihood_sequence import Network,objective as base_objective

def auxiliary(output,z,default,supervised):
    raw=output[1][supervised].float();truth=z[supervised][:,2]
    active=~default[supervised][:,2].bool()
    if not bool(active.any()):return raw.sum()*0
    r=raw[active,2];mean=(r[:,0].softmax(-1)*r[:,1]).sum(-1)
    return F.smooth_l1_loss(mean,truth[active],beta=.2,reduction='sum')/len(raw)

def objective(outputs,z,default,supervised,deep_supervision=False):
    main=outputs[0] if deep_supervision else outputs
    return base_objective(outputs,z,default,supervised,deep_supervision)+.2*auxiliary(main,z,default,supervised)

if __name__=='__main__':
    torch.set_num_threads(2);torch.manual_seed(91)
    r=torch.randn(2,16,3,3,8,requires_grad=True);g=torch.randn(2,16,3,requires_grad=True);z=torch.randn(2,16,3);d=torch.rand(2,16,3)>.5;m=torch.ones(2,16,dtype=torch.bool);m[:,0]=False
    value=auxiliary((g,r),z,d,m)
    rr=r[m][:,2];mean=(rr[:,0].softmax(-1)*rr[:,1]).sum(-1);error=(mean-z[m][:,2]).abs();expected=torch.where(error<.2,.5*error.square()/.2,error-.1);expected=expected.masked_fill(d[m][:,2],0).mean()
    assert torch.allclose(value,expected)
    zz=z.clone();zz[d]=1e4;zz[:,0]=-1e4
    assert torch.equal(value,auxiliary((g,r),zz,d,m));value.backward()
    assert torch.equal(r.grad[d],torch.zeros_like(r.grad[d])) and torch.equal(r.grad[:,:,:2],torch.zeros_like(r.grad[:,:,:2]))
    assert torch.equal(r.grad[:,0],torch.zeros_like(r.grad[:,0])) and g.grad is None
    assert torch.isfinite(r.grad).all()
    model=Network(features=12);x=torch.randn(2,256,12);depth=torch.randn(2,256);valid=torch.ones(2,256,dtype=torch.bool)
    loss=objective(model(x,depth,valid,True,absolute_depth=depth),torch.randn(2,256,3),torch.rand(2,256,3)>.5,valid,True);loss.backward()
    assert all(torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None)
    print('PASS: independent Huber formula, default/missing/other-target zero auxiliary gradients, finite model backward')
