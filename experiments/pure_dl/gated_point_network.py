"""Pointwise multiplicative gated MLP; no new depth convolution or PLE encoding."""
import json
import torch
from torch import nn
from torch.nn import functional as F
from multiscale_sequence import DensityHead, objective


class GatedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(192)
        self.expand=nn.Linear(192,512)
        self.project=nn.Linear(256,192)
        self.dropout=nn.Dropout(.05)
        self.gain=nn.Parameter(torch.full((192,),.1))

    def forward(self,x):
        value,gate=self.expand(self.norm(x)).chunk(2,-1)
        return x+self.gain*self.project(self.dropout(value*F.silu(gate)))


class Network(nn.Module):
    def __init__(self,features=206,initial_means=None):
        super().__init__()
        self.stem=nn.Linear(features,192)
        self.blocks=nn.Sequential(*[GatedBlock() for _ in range(4)])
        self.output=nn.Sequential(nn.LayerNorm(192),nn.Linear(192,128),nn.GELU())
        self.main_head=DensityHead(128,initial_means)
        self.aux_mid=DensityHead(128,initial_means)
        self.aux_coarse=DensityHead(128,initial_means)

    def forward(self,features,relative_depth,valid,auxiliary=False):
        point=features*valid[:,:,None]
        hidden=self.output(self.blocks(self.stem(point))).transpose(1,2)*valid[:,None]
        main=self.main_head(hidden)
        if not auxiliary:return main
        def pool(ratio):
            return F.avg_pool1d(hidden,ratio,ratio)/F.avg_pool1d(valid[:,None].float(),ratio,ratio).clamp_min(1e-6)
        return main,self.aux_mid(pool(2)),self.aux_coarse(pool(4))


def self_test():
    torch.set_num_threads(2);torch.manual_seed(163)
    model=Network(features=12).eval();x=torch.randn(2,16,12);d=torch.randn(2,16)
    valid=torch.ones(2,16,dtype=torch.bool);valid[0,:3]=False
    original=model(x,d,valid,True)
    with torch.no_grad():
        changed=x.clone();changed[~valid]+=1000
        for a,b in zip(original,model(changed,d,valid,True)):
            assert all(torch.equal(v,w) for v,w in zip(a,b))
        changed=x.clone();changed[1]+=100
        assert all(torch.equal(a[0],b[0]) for a,b in zip(original[0],model(changed,d,valid)))
        changed=x.clone();changed[0,8]+=100
        assert all(torch.equal(a[0,:8],b[0,:8]) for a,b in zip(original[0],model(changed,d,valid)))
    loss=objective(original,torch.randn(2,16,3),torch.rand(2,16,3)>.4,valid,True)
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert all(block.expand.weight.grad.abs().sum()>0 for block in model.blocks)
    return {'passed':True,'padding_and_well_isolation':True,'pointwise_main_prediction':True,
            'gated_blocks_receive_finite_gradients':True,'parameters':sum(p.numel() for p in Network().parameters())}


if __name__=='__main__':print(json.dumps(self_test(),indent=2))
