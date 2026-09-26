"""Train-only equal-well average trend, frozen before residual density training."""
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from multiscale_sequence import Network as Baseline, objective


class GlobalTrend(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(1,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,3))

    def forward(self,u):return self.layers((2*u-1)[...,None])


def average_profile(u,labels,defaults,wells,bins=64):
    index=np.clip((u*bins).astype(np.int64),0,bins-1)
    total=np.zeros((bins,3));counts=np.zeros((bins,3),np.int64)
    for well in np.unique(wells):
        for j in range(3):
            use=(wells==well)&~defaults[:,j]
            n=np.bincount(index[use],minlength=bins)
            sums=np.bincount(index[use],weights=labels[use,j],minlength=bins)
            present=n>0
            total[present,j]+=sums[present]/n[present];counts[present,j]+=1
    profile=total/np.maximum(counts,1)
    return (np.arange(bins,dtype=np.float32)+.5)/bins,profile.astype(np.float32),counts


def fit_trend(u,labels,defaults,wells):
    centers,profile,counts=average_profile(u,labels,defaults,wells)
    # Local RNG scope ensures fitting this stage does not change residual-model seed.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261019)
        model=GlobalTrend();opt=torch.optim.Adam(model.parameters(),lr=.01)
        xx=torch.tensor(centers);yy=torch.tensor(profile);mask=torch.tensor(counts>0)
        for _ in range(600):
            opt.zero_grad(set_to_none=True)
            loss=(model(xx)-yy).square()[mask].mean();loss.backward();opt.step()
    model.eval().requires_grad_(False)
    return model,{'centers':centers.tolist(),'profile_z':profile.tolist(),'contributing_wells':counts.tolist(),
                  'fit_mse':float(loss.detach()),'steps':600,'fit_seed':20261019,
                  'well_ids':np.unique(wells).tolist(),'averaging':'within-well bin means, then equal-weight mean across available training wells'}


class Network(Baseline):
    def __init__(self,features=206,initial_means=None,depth_index=0,depth_median=0.,depth_scale=1.,trend_state=None):
        super().__init__(features,initial_means)
        self.depth_index=depth_index
        self.register_buffer('depth_median',torch.tensor(float(depth_median)))
        self.register_buffer('depth_scale',torch.tensor(float(depth_scale)))
        self.global_trend=GlobalTrend()
        if trend_state is not None:self.global_trend.load_state_dict(trend_state)
        self.global_trend.requires_grad_(False)

    def forward(self,features,relative_depth,valid,auxiliary=False):
        outputs=super().forward(features,relative_depth,valid,auxiliary)
        u=(features[:,:,self.depth_index].float()*self.depth_scale+self.depth_median).clamp(0,1)*valid
        def add(output):
            gate,raw=output;position=u
            if raw.shape[1]!=u.shape[1]:
                ratio=u.shape[1]//raw.shape[1]
                position=(F.avg_pool1d(u[:,None],ratio,ratio)/F.avg_pool1d(valid[:,None].float(),ratio,ratio).clamp_min(1e-6)).squeeze(1)
            trend=self.global_trend(position)
            return gate,torch.stack([raw[...,0,:],raw[...,1,:]+trend[...,None],raw[...,2,:]],dim=-2)
        return tuple(add(o) for o in outputs) if auxiliary else add(outputs)


def self_test():
    torch.set_num_threads(2)
    u=np.full(101,.5);labels=np.r_[np.zeros((100,3)),np.full((1,3),10.)]
    wells=np.r_[np.zeros(100,int),[1]];default=np.zeros((101,3),bool)
    _,profile,counts=average_profile(u,labels,default,wells)
    assert np.array_equal(profile[32],np.full(3,5.)) and (counts[32]==2).all()
    default[-1]=True
    assert (average_profile(u,labels,default,wells)[1][32]==0).all()
    model=Network(features=12).eval();x=torch.randn(2,16,12);d=torch.randn(2,16);valid=torch.ones(2,16,dtype=torch.bool);valid[0,:3]=False
    before={k:v.clone() for k,v in model.global_trend.state_dict().items()}
    output=model(x,d,valid,True)
    loss=objective(output,torch.randn(2,16,3),torch.rand(2,16,3)>.4,valid,True);loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert all(p.grad is None and not p.requires_grad for p in model.global_trend.parameters())
    assert all(torch.equal(v,model.global_trend.state_dict()[k]) for k,v in before.items())
    with torch.no_grad():
        changed=x.clone();changed[~valid]+=1000
        for a,b in zip(output,model(changed,d,valid,True)):
            repeat=16//a[0].shape[1]
            assert all(torch.equal(v.repeat_interleave(repeat,1)[valid],w.repeat_interleave(repeat,1)[valid]) for v,w in zip(a,b))
    return {'passed':True,'equal_well_not_equal_row_average':True,'defaults_excluded_from_trend':True,
            'trend_frozen_during_residual_training':True,'masked_padding_invariant':True,
            'parameters':sum(p.numel() for p in Network().parameters())}


if __name__=='__main__':print(json.dumps(self_test(),indent=2))
