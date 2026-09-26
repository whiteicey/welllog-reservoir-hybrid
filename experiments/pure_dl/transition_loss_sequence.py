import torch
from torch.nn import functional as F
from population_likelihood_sequence import Network, objective as population_objective

def transition_loss(gate,defaults,supervised):
    pair=supervised[:,1:]&supervised[:,:-1]
    if not pair.any():return gate.sum()*0
    probability=gate.float().sigmoid()
    left=probability[:,:-1][pair];right=probability[:,1:][pair]
    changed=left*(1-right)+(1-left)*right
    target=(defaults[:,:-1]!=defaults[:,1:])[pair].float()
    return F.binary_cross_entropy(changed.clamp(1e-6,1-1e-6),target)

def objective(outputs,target_z,target_default,supervised,deep_supervision=False):
    main=outputs[0] if deep_supervision else outputs
    return population_objective(outputs,target_z,target_default,supervised,deep_supervision)+.1*transition_loss(main[0],target_default,supervised)

if __name__=='__main__':
    torch.set_num_threads(2);torch.manual_seed(241)
    g=torch.randn(2,8,3,requires_grad=True);d=torch.rand(2,8,3)>.5;m=torch.ones(2,8,dtype=torch.bool);m[:,0]=False
    loss=transition_loss(g,d,m)
    assert torch.allclose(loss,transition_loss(g.flip(1),d.flip(1),m.flip(1)))
    dd=d.clone();dd[~m]=~dd[~m]
    assert torch.equal(loss,transition_loss(g,dd,m))
    loss.backward();assert torch.isfinite(g.grad).all() and (g.grad[~m]==0).all()
    assert transition_loss(g,d,torch.zeros_like(m))==0
    confident=torch.tensor([[[8.,8.,8.],[8.,8.,8.]]]);stable=torch.ones_like(confident,dtype=torch.bool)
    mask=torch.ones(1,2,dtype=torch.bool)
    assert transition_loss(confident,stable,mask)<transition_loss(torch.zeros_like(confident),stable,mask)
    print('PASS: pair reversal invariance, masked labels/gradients isolation, empty pairs, stable-state confidence')
