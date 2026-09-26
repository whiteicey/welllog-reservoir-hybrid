import math
import numpy as np
import torch
from torch.nn import functional as F
from population_likelihood_sequence import objective as base_objective
NODES,WEIGHTS=np.polynomial.hermite.hermgauss(7)

def inverse(x,j):
    if j==0:return torch.expm1(x.clamp(0,math.log(51)))
    if j==1:return torch.pow(10.,x.clamp(-2,4))
    return x.clamp(math.log(8.305),math.log(99.9)).exp()
def utility(pred,truth,j):
    if j==1:return (1-torch.log10((pred/truth.clamp_min(1e-5)).clamp_min(1e-5)).abs()).clamp_min(0)
    tolerance=.08 if j==0 else .05
    return (1-(pred-truth).abs()/(tolerance*(truth.abs()+1e-5))).clamp_min(0)
def auxiliary(raw,z,default,mask,median,scale):
    r=raw[mask].float();zz=z[mask].float();dd=default[mask]
    nodes=torch.as_tensor(NODES,device=r.device,dtype=torch.float32)
    weights=torch.as_tensor(WEIGHTS/np.sqrt(np.pi),device=r.device,dtype=torch.float32)
    result=r.sum()*0
    for j,importance in enumerate([.3,.35,.35]):
        active=~dd[:,j]
        if not active.any():continue
        rr=r[active,j];w=rr[:,0].softmax(-1);mu=rr[:,1];sd=F.softplus(rr[:,2])+.025
        samples=(mu[:,:,None]+math.sqrt(2)*sd[:,:,None]*nodes)*scale[j]+median[j]
        predictions=inverse(samples,j)
        target=zz[active,j]*scale[j]+median[j]
        # Targets are not clipped; only predictions obey the existing output bounds.
        truth=torch.expm1(target) if j==0 else (torch.pow(10.,target) if j==1 else target.exp())
        u=utility(predictions,truth[:,None,None],j)
        expected=(u*weights).sum(-1).mul(w).sum(-1)
        result=result+importance*(1-expected).sum()/len(r)
    return result

def objective_for_stats(median,scale):
    def objective(outputs,z,default,mask,deep_supervision=False):
        main=outputs[0] if deep_supervision else outputs
        return base_objective(outputs,z,default,mask,deep_supervision)+.2*auxiliary(main[1],z,default,mask,median,scale)
    return objective
if __name__=='__main__':
    torch.set_num_threads(2);torch.manual_seed(271)
    for j in range(3):
        truth=torch.tensor([10.,20.,30.])
        assert torch.equal(utility(truth,truth,j),torch.ones(3))
        assert ((utility(truth*2,truth,j)>=0)&(utility(truth*2,truth,j)<=1)).all()
    r=torch.randn(2,8,3,3,8,requires_grad=True);z=torch.randn(2,8,3);d=torch.rand(2,8,3)>.5;m=torch.ones(2,8,dtype=torch.bool);m[:,0]=False
    med=[2.,0.,4.3];scale=[.3,.7,.15]
    value=auxiliary(r,z,d,m,med,scale);changed=z.clone();changed[d|(~m[:,:,None])]=1000
    assert torch.equal(value,auxiliary(r,changed,d,m,med,scale))
    value.backward();assert torch.isfinite(r.grad).all() and (r.grad[d]==0).all() and (r.grad[~m]==0).all()
    assert abs(float(np.sum(WEIGHTS/np.sqrt(np.pi)))-1)<1e-12
    print('PASS utility bounds/exact prediction, default and padding exclusion, finite gradients, quadrature normalization')
