"""Retire POR/PERM auxiliary tasks while preserving SW auxiliary supervision."""
import math
import torch
from torch.nn import functional as F
from global_point_loss import head_loss as main_loss

def channel_head_loss(output,target,default,mask,weights,factors):
    gate,raw=output
    if gate.shape[1]!=target.shape[1]:
        repeat=target.shape[1]//gate.shape[1]
        gate,raw=gate.repeat_interleave(repeat,1),raw.repeat_interleave(repeat,1)
    g,r,d=gate[mask].float(),raw[mask].float(),default[mask]
    y=target[mask].masked_fill(d,0)
    sd=F.softplus(r[:,:,2])+.025
    nll=-(r[:,:,0].log_softmax(-1)-.5*((y[:,:,None]-r[:,:,1])/sd).square()-sd.log()-.5*math.log(2*math.pi)).logsumexp(-1)
    channels=F.binary_cross_entropy_with_logits(g,d.float(),reduction='none')+nll.masked_fill(d,0)
    # Fixed /3: retaining SW does not amplify its coefficient after other tasks retire.
    per_point=(channels*r.new_tensor(factors)).mean(-1)
    return (weights[mask].detach().float()*per_point).mean()

def objective(outputs,target,default,mask,deep_supervision=False,weights=None,*,epoch):
    assert weights is not None and weights.shape==mask.shape
    if not deep_supervision:return main_loss(outputs,target,default,mask,weights)
    main,mid,coarse=outputs
    factor=max(0.,1.-epoch/40.)
    return (main_loss(main,target,default,mask,weights)
            +.2*channel_head_loss(mid,target,default,mask,weights,[factor,factor,1.])
            +.1*channel_head_loss(coarse,target,default,mask,weights,[factor,factor,1.]))

if __name__=='__main__':
    from global_point_loss import objective as original
    torch.manual_seed(88);torch.set_num_threads(2)
    outputs=tuple((torch.randn(2,n,3,requires_grad=True),torch.randn(2,n,3,3,8,requires_grad=True)) for n in [16,8,4])
    y=torch.randn(2,16,3);d=torch.rand(2,16,3)>.7;m=torch.ones(2,16,dtype=torch.bool);m[:,:2]=False
    q=torch.ones(2,16,requires_grad=True)
    assert torch.allclose(objective(outputs,y,d,m,True,q,epoch=0),original(outputs,y,d,m,True,q),atol=1e-6)
    loss=objective(outputs,y,d,m,True,q,epoch=40)
    assert torch.allclose(objective(outputs,y,d,m,True,q*.2,epoch=40),loss*.2)
    loss.backward();assert q.grad is None
    for gate,raw in outputs[1:]:
        assert gate.grad[:,:,:2].abs().sum()==0 and raw.grad[:,:,:2].abs().sum()==0
        assert gate.grad[:,:,2].abs().sum()>0 and raw.grad[:,:,2].abs().sum()>0
    assert outputs[0][1].grad[~m].abs().sum()==0
    assert torch.isfinite(loss)
    print('PASS: initial equivalence, POR/PERM aux retirement, SW retained, final Q weighting and mask')
