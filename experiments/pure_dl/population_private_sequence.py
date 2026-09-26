import torch
from depth_norm_joint_sequence import Network as Baseline
from private_head_sequence import PrivateHead
from population_likelihood_sequence import objective
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        self.main_head=PrivateHead(self.main_head)
        self.aux_mid=PrivateHead(self.aux_mid)
        self.aux_coarse=PrivateHead(self.aux_coarse)
if __name__=='__main__':
    torch.set_num_threads(2);torch.manual_seed(233);base=Baseline(features=12).eval()
    torch.manual_seed(233);model=Network(features=12).eval()
    x=torch.randn(2,256,12);d=torch.randn(2,256);a=torch.randn(2,256);valid=torch.ones(2,256,dtype=torch.bool);valid[:,:32]=False
    with torch.no_grad():
        for h in [model.main_head,model.aux_mid,model.aux_coarse]:h.gain.zero_()
        assert all(torch.allclose(u,v,atol=1e-6) for u,v in zip(base(x,d,valid,absolute_depth=a),model(x,d,valid,absolute_depth=a)))
        for h in [model.main_head,model.aux_mid,model.aux_coarse]:h.gain.fill_(.1)
        before=model(x,d,valid,absolute_depth=a)
        model.main_head.second.bias[:128].add_(1)
        after=model(x,d,valid,absolute_depth=a)
        assert all(torch.equal(u[:,:,1:],v[:,:,1:]) for u,v in zip(before,after))
    loss=objective(model(x,d,valid,True,absolute_depth=a),torch.randn(2,256,3),torch.rand(2,256,3)>.5,valid,True)
    loss.backward()
    for h in [model.main_head,model.aux_mid,model.aux_coarse]:
        assert h.first.weight.grad.abs().sum()>0
    assert all(torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None)
    print('PASS: baseline recovery, target-private isolation, finite population-likelihood gradients')
