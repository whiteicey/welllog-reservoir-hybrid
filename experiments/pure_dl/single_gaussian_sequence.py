import torch
from torch import nn
from depth_norm_joint_sequence import Network as Baseline
from population_likelihood_sequence import objective,head_loss
class SingleHead(nn.Module):
    def __init__(self,original):
        super().__init__();self.gate=original.gate
        width=original.density.in_channels;self.density=nn.Conv1d(width,6,1)
        with torch.no_grad():
            self.density.weight.copy_(original.density.weight.view(3,3,8,width,1)[:,1:].mean(2).reshape(6,width,1))
            self.density.bias.copy_(original.density.bias.view(3,3,8)[:,1:].mean(2).reshape(6))
    def forward(self,x):
        value=self.density(x).transpose(1,2).reshape(len(x),-1,3,2,1)
        raw=torch.cat([torch.zeros_like(value[:,:,:,:1]),value],dim=3)
        return self.gate(x).transpose(1,2),raw
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        self.main_head=SingleHead(self.main_head);self.aux_mid=SingleHead(self.aux_mid);self.aux_coarse=SingleHead(self.aux_coarse)
if __name__=='__main__':
    from torch.nn import functional as F
    torch.set_num_threads(2);torch.manual_seed(269)
    model=Network(features=12);x=torch.randn(2,256,12);d=torch.randn(2,256);a=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool)
    outputs=model(x,d,v,True,absolute_depth=a);y=torch.randn(2,256,3);default=torch.rand(2,256,3)>.5
    g,r=outputs[0];assert r.shape==(2,256,3,3,1)
    normal=torch.distributions.Normal(r[:,:,:,1,0].float(),F.softplus(r[:,:,:,2,0].float())+.025)
    expected=F.binary_cross_entropy_with_logits(g.float(),default.float())-normal.log_prob(y).masked_fill(default,0).mean()
    assert torch.allclose(head_loss(outputs[0],y,default,v),expected,atol=1e-6)
    loss=objective(outputs,y,default,v,True);loss.backward()
    assert all(torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None)
    assert model.main_head.density.weight.grad.abs().sum()>0
    print('PASS: single-component shape, exact Normal NLL equivalence, finite gradients')
