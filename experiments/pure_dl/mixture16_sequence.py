"""Sixteen-component density heads on the unchanged joint depth network."""
import math
import torch
from torch import nn
from depth_norm_joint_sequence import Network as Baseline
from population_likelihood_sequence import objective
class DensityHead(nn.Module):
    def __init__(self, width, initial_means=None):
        super().__init__()
        self.gate = nn.Conv1d(width, 3, 1)
        self.density = nn.Conv1d(width, 144, 1)
        nn.init.normal_(self.density.weight, 0, .005)
        nn.init.zeros_(self.density.bias)
        with torch.no_grad():
            bias = self.density.bias.view(3, 3, 16)
            if initial_means is not None:
                bias[:, 1].copy_(torch.as_tensor(initial_means))
            bias[:, 2].fill_(math.log(math.expm1(.5-.025)))

    def forward(self, x):
        return self.gate(x).transpose(1, 2), self.density(x).transpose(1, 2).reshape(len(x), -1, 3, 3, 16)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None):
        super().__init__(features, None)
        self.main_head=DensityHead(128, initial_means)
        self.aux_mid=DensityHead(96, initial_means)
        self.aux_coarse=DensityHead(128, initial_means)

if __name__=='__main__':
    torch.set_num_threads(2)
    model=Network(features=12)
    x=torch.randn(2,256,12); d=torch.randn(2,256); v=torch.ones(2,256,dtype=torch.bool)
    y=torch.randn(2,256,3); default=torch.rand(2,256,3)>.5
    out=model(x,d,v,True,absolute_depth=d)
    assert out[0][1].shape==(2,256,3,3,16)
    loss=objective(out,y,default,v,True); loss.backward()
    assert all(torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None)
    from population_likelihood_sequence import head_loss
    gate=torch.randn(2,3,requires_grad=True); raw=torch.randn(2,3,3,16,requires_grad=True)
    z=torch.randn(2,3); mask=torch.tensor([[True,False,True],[False,True,False]])
    value=head_loss((gate[None],raw[None]),z[None],mask[None],torch.ones(1,2,dtype=torch.bool))
    dist=torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=raw[:,:,0]),torch.distributions.Normal(raw[:,:,1],torch.nn.functional.softplus(raw[:,:,2])+.025))
    expected=torch.nn.functional.binary_cross_entropy_with_logits(gate,mask.float())-dist.log_prob(z).masked_fill(mask,0).mean()
    assert torch.allclose(value,expected,atol=1e-6)
    value.backward(); assert torch.equal(raw.grad[mask],torch.zeros_like(raw.grad[mask]))
    print('PASS: 16-component shapes, finite backward, independent likelihood, masked default gradients')
