"""PERM scalar log-MAE head; POR/SW retain Gaussian mixtures."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from blur_no_dropout_sequence import Network as Baseline
class PointPermHead(nn.Module):
    def __init__(self,old):
        super().__init__();self.gate=old.gate;self.density=old.density
        self.perm_point=nn.Conv1d(old.gate.in_channels,1,1)
        nn.init.zeros_(self.perm_point.weight);nn.init.zeros_(self.perm_point.bias)
    def forward(self,x):
        gate=self.gate(x).transpose(1,2)
        raw=self.density(x).transpose(1,2).reshape(len(x),-1,3,3,8)
        point=self.perm_point(x).transpose(1,2)
        placeholder=torch.stack([torch.zeros_like(point).expand(-1,-1,8),point.expand(-1,-1,8),torch.zeros_like(point).expand(-1,-1,8)],2)
        return gate,torch.cat([raw[:,:,:1],placeholder[:,:,None],raw[:,:,2:]],2)
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        for name in ['main_head','aux_mid','aux_coarse']:setattr(self,name,PointPermHead(getattr(self,name)))
def objective_for_scale(scale):
    def head(output,z,default,mask):
        gate,raw=output
        if gate.shape[1]!=z.shape[1]:
            repeat=z.shape[1]//gate.shape[1];gate=gate.repeat_interleave(repeat,1);raw=raw.repeat_interleave(repeat,1)
        gate=gate[mask].float();raw=raw[mask].float();y=z[mask];d=default[mask].bool()
        mu=raw[:,:,1];sd=F.softplus(raw[:,:,2])+.025
        loss=-(raw[:,:,0].log_softmax(-1)-.5*((y[:,:,None]-mu)/sd).square()-sd.log()-.5*math.log(2*math.pi)).logsumexp(-1)
        loss=loss.clone();loss[:,1]=(mu[:,1,0]-y[:,1]).abs()*float(scale[1])
        return F.binary_cross_entropy_with_logits(gate,d.float())+loss.masked_fill(d,0).mean()
    def objective(outputs,z,default,mask,deep_supervision=False):
        if not deep_supervision:return head(outputs,z,default,mask)
        return sum(w*head(o,z,default,mask) for w,o in zip([1,.2,.1],outputs))
    return objective
def decode_perm(gate,point_z,median,scale):
    point=torch.pow(10.,(point_z*scale+median).clamp(-2,4))
    return torch.where(gate>.5,torch.full_like(point,.01),point),point
if __name__=='__main__':
    torch.set_num_threads(2);model=Network(features=12)
    x=torch.randn(2,256,12);depth=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool);y=torch.randn(2,256,3);d=torch.rand(2,256,3)>.5
    out=model(x,depth,v,True,absolute_depth=depth);objective=objective_for_scale([1,2,1]);loss=objective(out,y,d,v,True);loss.backward()
    assert model.main_head.perm_point.weight.grad.abs().sum()>0
    assert all(torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None)
    r=torch.randn(1,6,3,3,8,requires_grad=True);g=torch.randn(1,6,3);y=torch.randn(1,6,3);d=torch.ones(1,6,3,dtype=torch.bool);d[:,:,1]=False;v=torch.ones(1,6,dtype=torch.bool)
    value=objective((g,r),y,d,v)
    expected=F.binary_cross_entropy_with_logits(g,d.float())+2*(r[:,:,1,1,0]-y[:,:,1]).abs().sum()/18
    assert torch.allclose(value,expected);value.backward();assert torch.equal(r.grad[d],torch.zeros_like(r.grad[d]))
    gate=torch.tensor([.1,.5,.9]);z=torch.tensor([0.,1.,2.]);chosen,point=decode_perm(gate,z,0.,1.)
    actions=torch.logspace(-2,4,2000)
    utility=(1-(actions[:,None].log10()-torch.stack([torch.full_like(point,-2),point.log10()],1)[:,None,:]).abs()).clamp_min(0)
    expected_max=(utility*torch.stack([gate,1-gate],1)[:,None,:]).sum(-1).max(-1).values
    chosen_score=gate*(1-(chosen.log10()+2).abs()).clamp_min(0)+(1-gate)*(1-(chosen.log10()-point.log10()).abs()).clamp_min(0)
    assert (chosen_score+1e-6>=expected_max).all()
    print('PASS: PERM log-MAE formula, default mask gradients, scalar-head gradients, fixed two-atom utility decoding')
