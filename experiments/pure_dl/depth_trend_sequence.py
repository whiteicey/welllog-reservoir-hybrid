"""A learned shared relative-depth trend plus local Gaussian-mixture residuals."""
import json
import torch
from torch import nn
from torch.nn import functional as F
from multiscale_sequence import Network as Baseline, objective


class Network(Baseline):
    def __init__(self,features=206,initial_means=None,depth_index=0,depth_median=0.,depth_scale=1.):
        super().__init__(features,initial_means)
        self.depth_index=depth_index
        self.register_buffer('depth_median',torch.tensor(float(depth_median)))
        self.register_buffer('depth_scale',torch.tensor(float(depth_scale)))
        self.register_buffer('trend_knots',torch.arange(16,dtype=torch.float32)/16)
        self.register_buffer('trend_strength',torch.tensor(1.))
        self.trend_slopes=nn.Parameter(torch.full((2,16),-3.))
        self.sw_trend=nn.Parameter(torch.zeros(16))
        self.trend_offset=nn.Parameter(torch.zeros(3))

    def trend(self,relative):
        basis=((relative[...,None]-self.trend_knots)*16).clamp(0,1)
        midpoint=((.5-self.trend_knots)*16).clamp(0,1)
        weights=torch.cat([-F.softplus(self.trend_slopes),self.sw_trend[None]],dim=0)
        return self.trend_strength*(torch.einsum('blk,jk->blj',basis-midpoint,weights)+self.trend_offset)

    def forward(self,features,relative_depth,valid,auxiliary=False):
        outputs=super().forward(features,relative_depth,valid,auxiliary)
        u=(features[:,:,self.depth_index].float()*self.depth_scale+self.depth_median).clamp(0,1)*valid
        def add(output):
            gate,raw=output
            position=u
            if raw.shape[1]!=u.shape[1]:
                ratio=u.shape[1]//raw.shape[1]
                numerator=F.avg_pool1d(u[:,None],ratio,ratio)
                denominator=F.avg_pool1d(valid[:,None].float(),ratio,ratio)
                position=(numerator/denominator.clamp_min(1e-6)).squeeze(1)
            trend=self.trend(position)
            return gate,torch.stack([raw[...,0,:],raw[...,1,:]+trend[...,None],raw[...,2,:]],dim=-2)
        return tuple(add(o) for o in outputs) if auxiliary else add(outputs)


def self_test():
    torch.set_num_threads(2)
    torch.manual_seed(131);base=Baseline(features=12).eval()
    torch.manual_seed(131);model=Network(features=12).eval()
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items())
    x=torch.randn(2,256,12);x[:,:,0]=torch.linspace(0,1,256);d=torch.randn(2,256)
    valid=torch.ones(2,256,dtype=torch.bool);valid[0,:32]=False
    with torch.no_grad():
        model.trend_strength.zero_()
        for a,b in zip(base(x,d,valid,True),model(x,d,valid,True)):
            assert all(torch.equal(v,w) for v,w in zip(a,b))
        model.trend_strength.fill_(1)
        trend=model.trend(torch.linspace(0,1,257)[None])
        assert (torch.diff(trend[:,:,:2],dim=1)<=1e-7).all()
        expected=model(x,d,valid)
        changed=x.clone();changed[~valid]+=1000
        actual=model(changed,d,valid)
        # Ignore padded output positions; real positions must be invariant.
        assert all(torch.equal(a[valid],b[valid]) for a,b in zip(expected,actual))
        for a,b in zip(model(x,d,valid,True),model(changed,d,valid,True)):
            repeat=256//a[0].shape[1]
            assert all(torch.equal(v.repeat_interleave(repeat,1)[valid],w.repeat_interleave(repeat,1)[valid]) for v,w in zip(a,b))
        changed=x.clone();changed[1]+=100
        assert all(torch.equal(a[0],b[0]) for a,b in zip(expected,model(changed,d,valid)))
    loss=objective(model(x,d,valid,True),torch.randn(2,256,3),torch.rand(2,256,3)>.4,valid,True)
    loss.backward()
    assert torch.isfinite(model.trend_slopes.grad).all() and model.trend_slopes.grad.abs().sum()>0
    return {'passed':True,'same_backbone_initialization':True,'disabled_trend_equals_baseline':True,
            'POR_PERM_trend_branch_nonincreasing':True,'valid_output_padding_and_well_isolation':True,
            'trend_gradients_finite_nonzero':True,'parameters':sum(p.numel() for p in Network().parameters())}


if __name__=='__main__':print(json.dumps(self_test(),indent=2))
