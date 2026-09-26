"""Double gated-MLP hidden width within the no-dropout blurred U-Net."""
import torch
from torch import nn
from blur_no_dropout_sequence import Network as Baseline
from multiscale_sequence import GatedBlock
from population_likelihood_sequence import objective
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        for block in self.modules():
            if isinstance(block,GatedBlock):
                c=block.project.out_channels
                block.expand=nn.Conv1d(c,4*c,1)
                block.project=nn.Conv1d(2*c,c,1)
if __name__=='__main__':
    torch.set_num_threads(2);model=Network(features=12).eval()
    blocks=[m for m in model.modules() if isinstance(m,GatedBlock)];assert len(blocks)==9
    assert all(b.expand.out_channels==4*b.project.out_channels and b.project.in_channels==2*b.project.out_channels and b.dropout.p==0 for b in blocks)
    x=torch.randn(2,256,12);d=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:32]=False
    with torch.no_grad():
        model.stem.weight[:,-1].fill_(.1);a=model(x,d,v,absolute_depth=d)
        xx=x.clone();xx[1]+=100;b=model(xx,d,v,absolute_depth=d);assert all(torch.equal(aa[0],bb[0]) for aa,bb in zip(a,b))
        xx=x.clone();xx[~v]+=100;dd=d.clone();dd[~v]+=100;b=model(xx,dd,v,absolute_depth=dd);assert all(torch.equal(aa,bb) for aa,bb in zip(a,b))
    loss=objective(model(x,d,v,True,absolute_depth=d),torch.randn(2,256,3),torch.rand(2,256,3)>.5,v,True);loss.backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in model.parameters())
    print('PASS: nine widened gated MLP blocks, no dropout, padded/well isolation, all gradients finite')
