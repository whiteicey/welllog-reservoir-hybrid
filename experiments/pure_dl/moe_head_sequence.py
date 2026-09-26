"""Four shared nonlinear experts with a separate input-dependent gate per target."""
import json
import torch
from torch import nn
from multiscale_sequence import Network as Baseline, objective


class MixtureHead(nn.Module):
    def __init__(self, original):
        super().__init__()
        width=original.gate.in_channels; self.width=width
        self.experts=nn.Sequential(nn.Conv1d(4*width,4*64,1,groups=4),nn.GELU(),
                                  nn.Conv1d(4*64,4*width,1,groups=4))
        self.router=nn.Conv1d(width,3*4,1)
        nn.init.normal_(self.router.weight,0,.01);nn.init.zeros_(self.router.bias)
        self.gain=nn.Parameter(torch.full((1,3,width,1),.1))
        self.gate=nn.Conv1d(3*width,3,1,groups=3)
        self.density=nn.Conv1d(3*width,72,1,groups=3)
        with torch.no_grad():
            self.gate.weight.copy_(original.gate.weight);self.gate.bias.copy_(original.gate.bias)
            self.density.weight.copy_(original.density.weight);self.density.bias.copy_(original.density.bias)

    def forward(self,x):
        batch,_,length=x.shape
        experts=self.experts(x.repeat(1,4,1)).reshape(batch,4,self.width,length)
        weights=self.router(x).reshape(batch,3,4,length).softmax(2)
        mixed=torch.einsum('btel,becl->btcl',weights,experts)
        features=(x[:,None]+self.gain*mixed).reshape(batch,3*self.width,length)
        return self.gate(features).transpose(1,2),self.density(features).transpose(1,2).reshape(batch,length,3,3,8)


class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        self.main_head=MixtureHead(self.main_head)
        self.aux_mid=MixtureHead(self.aux_mid)
        self.aux_coarse=MixtureHead(self.aux_coarse)


def self_test():
    torch.set_num_threads(2)
    torch.manual_seed(118);base=Baseline(features=12).eval()
    torch.manual_seed(118);model=Network(features=12).eval()
    for key,value in base.state_dict().items():
        if not key.startswith(('main_head.','aux_mid.','aux_coarse.')):
            assert torch.equal(value,model.state_dict()[key])
    x=torch.randn(2,256,12);depth=torch.randn(2,256);valid=torch.ones(2,256,dtype=torch.bool);valid[0,:32]=False
    with torch.no_grad():
        for head in (model.main_head,model.aux_mid,model.aux_coarse):head.gain.zero_()
        for a,b in zip(base(x,depth,valid,True),model(x,depth,valid,True)):
            assert all(torch.allclose(u,v,atol=1e-6) for u,v in zip(a,b))
        for head in (model.main_head,model.aux_mid,model.aux_coarse):head.gain.fill_(.1)
        expected=model(x,depth,valid)
        changed=x.clone();changed[~valid]+=1000
        assert all(torch.equal(a,b) for a,b in zip(expected,model(changed,depth,valid)))
        changed=x.clone();changed[1]+=100
        assert all(torch.equal(a[0],b[0]) for a,b in zip(expected,model(changed,depth,valid)))
    loss=objective(model(x,depth,valid,True),torch.randn(2,256,3),torch.rand(2,256,3)>.4,valid,True)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    for head in (model.main_head,model.aux_mid,model.aux_coarse):
        assert head.router.weight.grad.abs().sum()>0 and head.experts[0].weight.grad.abs().sum()>0
    return {'passed':True,'unchanged_backbone_initialization':True,'zero_gain_recovers_baseline':True,
            'masked_inputs_and_other_wells_isolated':True,'gates_and_experts_receive_gradients':True,
            'parameters':sum(p.numel() for p in Network().parameters())}


if __name__=='__main__':print(json.dumps(self_test(),indent=2))
