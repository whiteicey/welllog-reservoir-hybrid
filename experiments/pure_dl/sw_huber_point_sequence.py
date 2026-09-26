"""SW scalar log-Huber head; POR/PERM retain Gaussian mixtures."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from blur_no_dropout_sequence import Network as Baseline
class PointSWHead(nn.Module):
    def __init__(self,old):
        super().__init__();self.gate=old.gate;self.density=old.density
        self.sw_point=nn.Conv1d(old.gate.in_channels,1,1)
        nn.init.zeros_(self.sw_point.weight);nn.init.zeros_(self.sw_point.bias)
    def forward(self,x):
        gate=self.gate(x).transpose(1,2)
        raw=self.density(x).transpose(1,2).reshape(len(x),-1,3,3,8)
        point=self.sw_point(x).transpose(1,2)
        placeholder=torch.stack([torch.zeros_like(point).expand(-1,-1,8),point.expand(-1,-1,8),torch.zeros_like(point).expand(-1,-1,8)],2)
        return gate,torch.cat([raw[:,:,:2],placeholder[:,:,None]],2)
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        for name in ['main_head','aux_mid','aux_coarse']:setattr(self,name,PointSWHead(getattr(self,name)))
def objective_for_scale(scale):
    def head(output,z,default,mask):
        gate,raw=output
        if gate.shape[1]!=z.shape[1]:
            repeat=z.shape[1]//gate.shape[1];gate=gate.repeat_interleave(repeat,1);raw=raw.repeat_interleave(repeat,1)
        gate=gate[mask].float();raw=raw[mask].float();y=z[mask];d=default[mask].bool()
        mu=raw[:,:,1];sd=F.softplus(raw[:,:,2])+.025
        loss=-(raw[:,:,0].log_softmax(-1)-.5*((y[:,:,None]-mu)/sd).square()-sd.log()-.5*math.log(2*math.pi)).logsumexp(-1)
        loss=loss.clone();loss[:,2]=F.smooth_l1_loss((mu[:,2,0]-y[:,2])*float(scale[2]),torch.zeros_like(y[:,2]),beta=math.log(1.05),reduction='none')
        return F.binary_cross_entropy_with_logits(gate,d.float())+loss.masked_fill(d,0).mean()
    def objective(outputs,z,default,mask,deep_supervision=False):
        if not deep_supervision:return head(outputs,z,default,mask)
        return sum(w*head(o,z,default,mask) for w,o in zip([1,.2,.1],outputs))
    return objective
def decode_sw(gate,point_z,median,scale):
    point=(point_z*scale+median).clamp(math.log(8.305),math.log(99.9)).exp()
    default=torch.full_like(point,99.9)
    distance=(default-point).abs()
    default_score=gate+(1-gate)*(1-distance/(.05*(point.abs()+1e-5))).clamp_min(0)
    point_score=(1-gate)+gate*(1-distance/(.05*(default.abs()+1e-5))).clamp_min(0)
    return torch.where(point_score>default_score,point,default),point

if __name__=='__main__':
    import numpy as np
    import train_invariant as base
    torch.set_num_threads(2);model=Network(features=12)
    x=torch.randn(2,256,12);depth=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool);y=torch.randn(2,256,3);d=torch.rand(2,256,3)>.5
    objective=objective_for_scale([1,1,.5]);loss=objective(model(x,depth,v,True,absolute_depth=depth),y,d,v,True);loss.backward()
    assert model.main_head.sw_point.weight.grad.abs().sum()>0
    assert all(torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None)
    r=torch.randn(1,6,3,3,8,requires_grad=True);g=torch.randn(1,6,3);y=torch.randn(1,6,3);d=torch.ones(1,6,3,dtype=torch.bool);d[:,:,2]=False;v=torch.ones(1,6,dtype=torch.bool)
    value=objective((g,r),y,d,v);e=.5*(r[:,:,2,1,0]-y[:,:,2]).abs();beta=math.log(1.05)
    expected=F.binary_cross_entropy_with_logits(g,d.float())+torch.where(e<beta,.5*e.square()/beta,e-.5*beta).sum()/18
    assert torch.allclose(value,expected);value.backward();assert torch.equal(r.grad[d],torch.zeros_like(r.grad[d]))
    gate=torch.linspace(.01,.99,41);zz=torch.linspace(math.log(10),math.log(99.8),41)
    chosen,point=decode_sw(gate,zz,0.,1.)
    for g,q,c in zip(gate.numpy(),point.numpy(),chosen.numpy()):
        action=np.r_[np.linspace(8.305,99.9,1000),q,99.9]
        score=base.kernel_matrix(action,np.array([99.9,q]),'SW')@np.array([g,1-g])
        actual=base.kernel_matrix(np.array([c]),np.array([99.9,q]),'SW')@np.array([g,1-g])
        assert actual[0]+1e-5>=score.max()
    print('PASS: independent Huber formula, default masking, point gradients, original SW utility-grid decoder check')
