"""Train-only upper-quartile well-risk emphasis on the verified ungated network."""
import math
import torch
from torch.nn import functional as F
from blur_ungated_sequence import Network

def row_loss(output,target,default,mask):
    gate,raw=output
    if gate.shape[1]!=target.shape[1]:
        n=target.shape[1]//gate.shape[1];gate=gate.repeat_interleave(n,1);raw=raw.repeat_interleave(n,1)
    gate=gate[mask].float();raw=raw[mask].float();d=default[mask];y=target[mask].masked_fill(d,0)
    weight=raw[:,:,0].log_softmax(-1);mu=raw[:,:,1];sd=F.softplus(raw[:,:,2])+.025
    nll=-(weight-.5*((y[:,:,None]-mu)/sd).square()-sd.log()-.5*math.log(2*math.pi)).logsumexp(-1)
    return (F.binary_cross_entropy_with_logits(gate,d.float(),reduction='none')+nll.masked_fill(d,0)).mean(-1)

def objective(outputs,target,default,mask,deep_supervision=False,well_ids=None):
    assert well_ids is not None and well_ids.shape==(len(target),)
    if deep_supervision:
        losses=sum(w*row_loss(o,target,default,mask) for w,o in zip([1.,.2,.1],outputs))
    else:losses=row_loss(outputs,target,default,mask)
    ids=well_ids[:,None].expand_as(mask)[mask]
    _,group=torch.unique(ids,return_inverse=True)
    count=torch.bincount(group).to(losses.dtype)
    risk=torch.zeros_like(count).scatter_add_(0,group,losses)/count
    k=max(1,math.ceil(len(risk)/4))
    return .9*losses.mean()+.1*risk.topk(k).values.mean()

if __name__=='__main__':
    torch.set_num_threads(2);torch.manual_seed(42)
    g=torch.randn(4,8,3,requires_grad=True);r=torch.randn(4,8,3,3,8,requires_grad=True)
    y=torch.randn(4,8,3);d=torch.rand(4,8,3)>.4;m=torch.ones(4,8,dtype=torch.bool);m[:,0]=False;m[1,1:3]=False
    ids=torch.tensor([4,4,9,12]);loss=objective((g,r),y,d,m,well_ids=ids)
    rr=r[m];dd=d[m];yy=y[m].masked_fill(dd,0)
    dist=torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=rr[:,:,0]),torch.distributions.Normal(rr[:,:,1],F.softplus(rr[:,:,2])+.025))
    rows=(F.binary_cross_entropy_with_logits(g[m],dd.float(),reduction='none')-dist.log_prob(yy).masked_fill(dd,0)).mean(-1)
    row_ids=ids[:,None].expand_as(m)[m]
    risk=torch.stack([rows[row_ids==i].mean() for i in ids.unique()])
    expected=.9*rows.mean()+.1*risk.max()
    assert torch.allclose(loss,expected,atol=1e-6)
    changed=y.clone();changed[d|(~m[:,:,None])]=1000
    assert torch.equal(loss,objective((g,r),changed,d,m,well_ids=ids))
    loss.backward();assert torch.isfinite(r.grad).all() and (r.grad[d]==0).all() and (r.grad[~m]==0).all()
    model=Network(features=12);x=torch.randn(4,256,12);depth=torch.randn(4,256);mask=torch.ones(4,256,dtype=torch.bool)
    value=objective(model(x,depth,mask,True,absolute_depth=depth),torch.randn(4,256,3),torch.rand(4,256,3)>.5,mask,True,ids)
    value.backward();assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in model.parameters())
    print('PASS: independent grouped-risk formula, repeated-well aggregation, default/padding masks and model gradients')
