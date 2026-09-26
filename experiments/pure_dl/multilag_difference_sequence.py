"""Supervise nondefault adjacent-row differences; no state-transition objective."""
import torch
from torch.nn import functional as F
from population_likelihood_sequence import Network, objective as base_objective

def difference_loss(output, target_z, target_default, supervised):
    _, raw=output
    raw=raw.float()
    mean=(raw[:,:,:,0].softmax(-1)*raw[:,:,:,1]).sum(-1)
    active=supervised[:,:,None] & ~target_default.bool()
    terms=[]
    for lag in (1,4,16):
        if mean.shape[1]<=lag:
            continue
        pair=active.unfold(1,lag+1,1).all(-1)
        support=supervised.unfold(1,lag+1,1).all(-1)
        pred=(mean[:,lag:]-mean[:,:-lag])[pair]
        truth=(target_z[:,lag:]-target_z[:,:-lag])[pair]
        if bool(pair.any()):
            terms.append(F.smooth_l1_loss(pred,truth,reduction='sum',beta=.1)/max(int(support.sum())*3,1))
        else:
            terms.append(mean.sum()*0)
    return sum(terms)/len(terms)

def objective(outputs, target_z, target_default, supervised, deep_supervision=False):
    main=outputs[0] if deep_supervision else outputs
    return base_objective(outputs,target_z,target_default,supervised,deep_supervision)+.1*difference_loss(main,target_z,target_default,supervised)

if __name__=='__main__':
    torch.set_num_threads(2)
    raw=torch.randn(2,32,3,3,8,requires_grad=True)
    gate=torch.randn(2,32,3,requires_grad=True)
    target=torch.randn(2,32,3)
    default=torch.zeros(2,32,3,dtype=torch.bool);default[:,3]=True
    supervised=torch.ones(2,32,dtype=torch.bool);supervised[:,0]=False
    a=difference_loss((gate,raw),target,default,supervised)
    altered=target.clone();altered[:,3]=10000;altered[:,0]=-10000
    assert torch.equal(a,difference_loss((gate,raw),altered,default,supervised))
    a.backward();assert torch.equal(raw.grad[:,3],torch.zeros_like(raw.grad[:,3]));assert torch.equal(raw.grad[:,0],torch.zeros_like(raw.grad[:,0]));assert gate.grad is None
    mean=(raw.detach()[:,:,:,0].softmax(-1)*raw.detach()[:,:,:,1]).sum(-1)
    assert difference_loss((gate,raw),mean,default,supervised).item()==0
    assert difference_loss((gate,raw),target,torch.ones_like(default),supervised).item()==0
    # Independently enumerate every interval, including an interior default.
    active=supervised[:,:,None] & ~default
    ref=[]
    mean=(raw[:,:,:,0].softmax(-1)*raw[:,:,:,1]).sum(-1)
    for lag in (1,4,16):
        terms=[]; count=0
        for b in range(2):
            for i in range(32-lag):
                if bool(supervised[b,i:i+lag+1].all()):
                    count+=3
                    for j in range(3):
                        if bool(active[b,i:i+lag+1,j].all()):
                            terms.append(F.smooth_l1_loss(mean[b,i+lag,j]-mean[b,i,j],target[b,i+lag,j]-target[b,i,j],beta=.1,reduction='sum'))
        ref.append(sum(terms)/max(count,1))
    assert torch.allclose(a,sum(ref)/3,atol=1e-6)
    model=Network(features=12);x=torch.randn(2,256,12);d=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool)
    loss=objective(model(x,d,v,True,absolute_depth=d),torch.randn(2,256,3),torch.rand(2,256,3)>.5,v,True)
    loss.backward();assert all(torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None)
    print('PASS: excluded default/missing endpoints, no gate auxiliary gradient, exact differences zero, all-default zero, finite model backward')
