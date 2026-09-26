"""Categorical conditional target distributions with two-bin soft labels.

Not an action-utility classifier: outputs represent target probabilities, to be
decoded with the fixed official expected-utility rule. Grids are train-fold-only.
"""
import json
import torch
from torch import nn
from torch.nn import functional as F
from multiscale_sequence import Network as Baseline

BINS=256


class CategoricalHead(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.gate=original.gate
        self.density=nn.Conv1d(original.gate.in_channels,3*BINS,1)
        nn.init.normal_(self.density.weight,0,.005);nn.init.zeros_(self.density.bias)

    def forward(self,x):
        return self.gate(x).transpose(1,2),self.density(x).transpose(1,2).reshape(len(x),-1,3,BINS)


class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        self.main_head=CategoricalHead(self.main_head)
        self.aux_mid=CategoricalHead(self.aux_mid)
        self.aux_coarse=CategoricalHead(self.aux_coarse)


def soft_label_loss(logits,truth,centers):
    assert centers.shape==logits.shape[-2:]
    step=(centers[:,-1]-centers[:,0])/(centers.shape[-1]-1)
    assert bool((step>0).all())
    position=((truth-centers[:,0])/step).clamp(0,centers.shape[-1]-1)
    lower=position.long();upper=(lower+1).clamp_max(centers.shape[-1]-1)
    fraction=position-lower
    logp=logits.float().log_softmax(-1)
    return -((1-fraction)*logp.gather(-1,lower[...,None]).squeeze(-1)
             +fraction*logp.gather(-1,upper[...,None]).squeeze(-1))


def head_loss(output,truth,defaults,supervised,centers_z):
    gate,logits=output
    if gate.shape[1]!=truth.shape[1]:
        repeat=truth.shape[1]//gate.shape[1]
        assert repeat*gate.shape[1]==truth.shape[1]
        gate=gate.repeat_interleave(repeat,1);logits=logits.repeat_interleave(repeat,1)
    gate=gate[supervised].float();logits=logits[supervised]
    label=truth[supervised];default=defaults[supervised]
    losses=soft_label_loss(logits,label,centers_z)
    terms=[]
    for j in range(3):
        active=~default[:,j].bool()
        terms.append(losses[active,j].mean() if active.any() else logits[:,j].sum()*0)
    return F.binary_cross_entropy_with_logits(gate,default.float())+sum(terms)/3


def objective(outputs,target_z,target_default,supervised,deep_supervision=False,*,centers_z):
    if not deep_supervision:return head_loss(outputs,target_z,target_default,supervised,centers_z)
    return sum(w*head_loss(o,target_z,target_default,supervised,centers_z) for w,o in zip((1.,.2,.1),outputs))


def self_test():
    torch.set_num_threads(2);torch.manual_seed(123)
    centers=torch.tensor([[0.,1.,2.]]).expand(3,-1)
    logits=torch.randn(2,3,3,requires_grad=True);truth=torch.tensor([[0.,1.,2.],[.5,1.5,1.]])
    actual=soft_label_loss(logits,truth,centers)
    lp=logits.log_softmax(-1)
    assert torch.allclose(actual[0],-lp[0,torch.arange(3),torch.arange(3)])
    assert torch.allclose(actual[1,0],-.5*(lp[1,0,0]+lp[1,0,1]))
    model=Network(features=12).eval();x=torch.randn(2,256,12);d=torch.randn(2,256)
    valid=torch.ones(2,256,dtype=torch.bool);valid[:,:32]=False
    output=model(x,d,valid,True);z=torch.randn(2,256,3);default=torch.rand(2,256,3)>.4
    grid=torch.linspace(-4,4,BINS)[None].expand(3,-1)
    loss=objective(output,z,default,valid,True,centers_z=grid)
    altered=z.clone();altered[~valid]=float('nan')
    assert torch.equal(loss,objective(output,altered,default,valid,True,centers_z=grid))
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert output[0][1].shape==(2,256,3,BINS)
    assert torch.allclose(output[0][1].softmax(-1).sum(-1),torch.ones(2,256,3),atol=1e-6)
    return {'passed':True,'exact_center_and_midpoint_labels':True,'masked_labels_excluded':True,
            'finite_gradients':True,'normalized_discrete_probabilities':True,
            'parameters':sum(p.numel() for p in Network().parameters()),'gpu_training_started':False}


if __name__=='__main__':print(json.dumps(self_test(),indent=2))
