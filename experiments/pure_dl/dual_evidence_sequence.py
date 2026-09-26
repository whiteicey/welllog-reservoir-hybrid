"""Untrained mechanism prototype: separate feature and sequence evidence paths.

The feature path reads existing per-row descriptors, which include engineered
context features; it is not a strictly raw-point-only or independent teacher.
Inference remains the sequence path. Consensus is evidence, not proof of bad labels.
"""
import math
import torch
from torch import nn
from relative_only_sequence import Network as Baseline
from global_point_loss import objective as weighted_objective


class Network(Baseline):
    def __init__(self, features=206, initial_means=None):
        super().__init__(features, initial_means)
        with torch.random.fork_rng(devices=[]):
            self.feature_expert=nn.Sequential(nn.LayerNorm(features),nn.Linear(features,64),
                nn.GELU(),nn.Linear(64,64),nn.GELU(),nn.Linear(64,75))

    def feature_prediction(self, features, valid):
        z=self.feature_expert(features.float().masked_fill(~valid[...,None],0))
        return z[...,:3],z[...,3:].reshape(*z.shape[:-1],3,3,8)

    def forward(self, features, relative_depth, valid, auxiliary=False,
                absolute_depth=None, diagnostic=False):
        sequence=super().forward(features,relative_depth,valid,auxiliary,
                                 absolute_depth=absolute_depth)
        if auxiliary or diagnostic:
            feature=self.feature_prediction(features,valid)
            return (sequence,feature)
        return sequence


def objective(outputs,target,default,mask,deep_supervision=False,weights=None):
    sequence,feature=outputs
    primary=weighted_objective(sequence,target,default,mask,deep_supervision,weights)
    # Feature expert remains independently supervised, not silenced by main-path judgments.
    secondary=weighted_objective(feature,target,default,mask,False,torch.ones_like(weights))
    return primary + .1*secondary


def distribution_agreement(p,q):
    """Normalized Jensen-Shannon divergence in [0,1]; zero means agreement."""
    p=p.float().clamp_min(0);q=q.float().clamp_min(0)
    p=p/p.sum(-1,keepdim=True).clamp_min(1e-30)
    q=q/q.sum(-1,keepdim=True).clamp_min(1e-30)
    mid=(p+q)/2
    return .5*(torch.special.xlogy(p,p/mid.clamp_min(1e-30)).sum(-1)
               +torch.special.xlogy(q,q/mid.clamp_min(1e-30)).sum(-1))/math.log(2)


if __name__=='__main__':
    torch.set_num_threads(2)
    torch.manual_seed(13);base=Baseline(features=12).eval()
    torch.manual_seed(13);model=Network(features=12).eval()
    x=torch.randn(2,128,12);d=torch.randn(2,128);v=torch.ones(2,128,dtype=torch.bool)
    with torch.no_grad():
        assert all(torch.equal(a,b) for a,b in zip(base(x,d,v,absolute_depth=d),model(x,d,v,absolute_depth=d)))
        original=model.feature_prediction(x,v)
        changed=x.clone();changed[:,1:]+=10
        assert all(torch.equal(a[:,0],b[:,0]) for a,b in zip(original,model.feature_prediction(changed,v)))
    target=torch.randn(2,128,3);default=torch.rand(2,128,3)>.5
    feature=model.feature_prediction(x,v)
    weighted_objective(feature,target,default,v,False,torch.ones(2,128)).backward()
    assert all(p.grad is None for name,p in model.named_parameters() if not name.startswith('feature_expert.'))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.feature_expert.parameters())
    p=torch.tensor([[1.,0,0],[.2,.3,.5]])
    assert torch.allclose(distribution_agreement(p,p),torch.zeros(2))
    assert torch.allclose(distribution_agreement(p[:1],torch.tensor([[0.,0,1.]])),torch.ones(1))
    assert torch.equal(distribution_agreement(p,p.flip(-1)),distribution_agreement(p.flip(-1),p))
    model.zero_grad();outputs=model(x,d,v,True,absolute_depth=d)
    loss=objective(outputs,target,default,v,True,torch.ones(2,128));loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print('PASS unchanged main inference, feature token isolation, disjoint gradients, JS identity/extremes/symmetry, joint finite gradients')
    print('PARAMETERS',sum(p.numel() for p in Network().parameters()))
