"""Only the first downsampling stage uses the previously verified fixed blur."""
from depth_norm_joint_sequence import Network as Baseline
from blur_down_sequence import BlurDown
from population_likelihood_sequence import objective
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        self.down1=BlurDown(self.down1)
if __name__=='__main__':
    import torch
    torch.set_num_threads(2);torch.manual_seed(17);base=Baseline(features=12)
    torch.manual_seed(17);model=Network(features=12)
    assert sum(isinstance(m,BlurDown) for m in model.modules())==1
    assert torch.equal(base.down1.weight,model.down1.conv.weight)
    assert torch.equal(base.down2.weight,model.down2.weight) and torch.equal(base.down3.weight,model.down3.weight)
    x=torch.randn(2,256,12);d=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool)
    loss=objective(model(x,d,v,True,absolute_depth=d),torch.randn(2,256,3),torch.rand(2,256,3)>.5,v,True);loss.backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in model.parameters())
    print('PASS: exactly first-stage blur, unchanged trainable initialization, all gradients finite')
