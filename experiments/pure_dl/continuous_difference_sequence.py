"""Supervise nondefault adjacent-row differences; no state-transition objective."""
import torch
from torch.nn import functional as F
from population_likelihood_sequence import Network, objective as base_objective

def difference_loss(output, target_z, target_default, supervised):
    _, raw=output
    raw=raw.float()
    mean=(raw[:,:,:,0].softmax(-1)*raw[:,:,:,1]).sum(-1)
    active=supervised[:,:,None] & ~target_default.bool()
    pair=active[:,1:] & active[:,:-1]
    # Select valid pairs before arithmetic: missing labels must not affect loss.
    pred=(mean[:,1:]-mean[:,:-1])[pair]
    truth=(target_z[:,1:]-target_z[:,:-1])[pair]
    if not bool(pair.any()):
        return mean.sum()*0
    # Population normalization preserves nondefault prevalence weighting.
    return F.smooth_l1_loss(pred,truth,reduction='sum',beta=.1)/max(int((supervised[:,1:] & supervised[:,:-1]).sum())*3,1)

def objective(outputs, target_z, target_default, supervised, deep_supervision=False):
    main=outputs[0] if deep_supervision else outputs
    return base_objective(outputs,target_z,target_default,supervised,deep_supervision)+.1*difference_loss(main,target_z,target_default,supervised)

if __name__=='__main__':
    torch.set_num_threads(2)
    raw=torch.randn(2,8,3,3,8,requires_grad=True)
    gate=torch.randn(2,8,3,requires_grad=True)
    target=torch.randn(2,8,3)
    default=torch.zeros(2,8,3,dtype=torch.bool);default[:,3]=True
    supervised=torch.ones(2,8,dtype=torch.bool);supervised[:,0]=False
    a=difference_loss((gate,raw),target,default,supervised)
    altered=target.clone();altered[:,3]=10000;altered[:,0]=-10000
    assert torch.equal(a,difference_loss((gate,raw),altered,default,supervised))
    a.backward();assert torch.equal(raw.grad[:,3],torch.zeros_like(raw.grad[:,3]));assert torch.equal(raw.grad[:,0],torch.zeros_like(raw.grad[:,0]));assert gate.grad is None
    mean=(raw.detach()[:,:,:,0].softmax(-1)*raw.detach()[:,:,:,1]).sum(-1)
    assert difference_loss((gate,raw),mean,default,supervised).item()==0
    assert difference_loss((gate,raw),target,torch.ones_like(default),supervised).item()==0
    model=Network(features=12);x=torch.randn(2,256,12);d=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool)
    loss=objective(model(x,d,v,True,absolute_depth=d),torch.randn(2,256,3),torch.rand(2,256,3)>.5,v,True)
    loss.backward();assert all(torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None)
    print('PASS: excluded default/missing endpoints, no gate auxiliary gradient, exact differences zero, all-default zero, finite model backward')
