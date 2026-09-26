"""Joint POR/PERM Gaussian mixture likelihood; SW remains independently modeled."""
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from multiscale_sequence import Network as Baseline


class PairedHead(nn.Module):
    def __init__(self,original):
        super().__init__();self.gate=original.gate
        self.density=nn.Conv1d(original.gate.in_channels,64,1)
        with torch.no_grad():
            for new,old in [(self.density.weight,original.density.weight),(self.density.bias,original.density.bias)]:
                new[:8].copy_((old[:8]+old[24:32])*.5)
                new[8:16].copy_(old[8:16]);new[16:24].copy_(old[32:40])
                new[24:32].copy_(old[16:24]);new[32:40].copy_(old[40:48])
                new[40:].copy_(old[48:])

    def forward(self,x):
        value=self.density(x).transpose(1,2)
        por=torch.stack([value[...,:8],value[...,8:16],value[...,24:32]],-2)
        perm=torch.stack([value[...,:8],value[...,16:24],value[...,32:40]],-2)
        sw=value[...,40:].reshape(len(x),-1,3,8)
        return self.gate(x).transpose(1,2),torch.stack([por,perm,sw],-3)


class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        self.main_head=PairedHead(self.main_head)
        self.aux_mid=PairedHead(self.aux_mid);self.aux_coarse=PairedHead(self.aux_coarse)


def joint_pair_nll(log_weights,means,scales,truth,active):
    safe=torch.where(active,truth,torch.zeros_like(truth))
    component=-.5*((safe[...,None]-means)/scales).square()-scales.log()-.5*math.log(2*math.pi)
    component=torch.where(active[...,None],component,torch.zeros_like(component))
    return -(log_weights+component.sum(-2)).logsumexp(-1)


def head_loss(output,truth,defaults,supervised):
    gate,raw=output
    if gate.shape[1]!=truth.shape[1]:
        repeat=truth.shape[1]//gate.shape[1];assert repeat*gate.shape[1]==truth.shape[1]
        gate=gate.repeat_interleave(repeat,1);raw=raw.repeat_interleave(repeat,1)
    gate=gate[supervised].float();raw=raw[supervised].float()
    y=truth[supervised];active=~defaults[supervised].bool()
    logw=raw[:,:,0].log_softmax(-1);mu=raw[:,:,1];sd=F.softplus(raw[:,:,2])+.025
    pair=joint_pair_nll(logw[:,0],mu[:,:2],sd[:,:2],y[:,:2],active[:,:2])
    count=active[:,:2].sum(-1);observed=count>0
    pair_mean=(pair[observed]/count[observed]).mean() if observed.any() else raw.sum()*0
    sw=joint_pair_nll(logw[:,2],mu[:,2:3],sd[:,2:3],y[:,2:3],active[:,2:3])
    sw_mean=sw[active[:,2]].mean() if active[:,2].any() else raw.sum()*0
    return F.binary_cross_entropy_with_logits(gate,(~active).float())+(2*pair_mean+sw_mean)/3


def objective(outputs,target_z,target_default,supervised,deep_supervision=False):
    if not deep_supervision:return head_loss(outputs,target_z,target_default,supervised)
    return sum(w*head_loss(o,target_z,target_default,supervised) for w,o in zip((1.,.2,.1),outputs))


def self_test():
    torch.set_num_threads(2);torch.manual_seed(153)
    logw=torch.randn(4,8).log_softmax(-1);mu=torch.randn(4,2,8);sd=torch.rand(4,2,8)+.1;y=torch.randn(4,2)
    active=torch.tensor([[True,False]]).expand(4,-1)
    actual=joint_pair_nll(logw,mu,sd,y,active)
    reference=-(logw-.5*((y[:,0,None]-mu[:,0])/sd[:,0]).square()-sd[:,0].log()-.5*math.log(2*math.pi)).logsumexp(-1)
    assert torch.allclose(actual,reference)
    changed=y.clone();changed[:,1]=1e8
    assert torch.equal(actual,joint_pair_nll(logw,mu,sd,changed,active))
    assert torch.allclose(joint_pair_nll(logw,mu,sd,y,torch.zeros_like(active)),torch.zeros(4),atol=1e-6)
    model=Network(features=12);x=torch.randn(2,16,12);d=torch.randn(2,16);valid=torch.ones(2,16,dtype=torch.bool);valid[:,:3]=False
    out=model(x,d,valid,True)
    assert torch.equal(out[0][1][:,:,0,0],out[0][1][:,:,1,0])
    target=torch.randn(2,16,3);default=torch.rand(2,16,3)>.4
    loss=objective(out,target,default,valid,True)
    altered=target.clone();altered[~valid]=float('nan')
    assert torch.equal(loss,objective(out,altered,default,valid,True))
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    return {'passed':True,'missing_pair_target_integrates_to_correct_marginal':True,
            'default_target_value_excluded':True,'shared_POR_PERM_mixture_weights':True,
            'masked_labels_excluded':True,'finite_gradients':True,'parameters':sum(p.numel() for p in Network().parameters())}


if __name__=='__main__':print(json.dumps(self_test(),indent=2))
