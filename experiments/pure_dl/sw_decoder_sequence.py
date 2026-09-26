"""Shared encoder with a separate three-stage SW decoder, trained from scratch."""
import copy
import torch
from torch.nn import functional as F
from depth_norm_joint_sequence import Network as Baseline
from population_likelihood_sequence import objective
class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        for name in ['merge2','dec2','merge1','dec1','merge0','dec0','combine','main_head','aux_mid','aux_coarse']:
            setattr(self,'sw_'+name,copy.deepcopy(getattr(self,name)))
    def decode(self,h,e0,e1,e2,point,prefix,auxiliary):
        def layer(name):return getattr(self,prefix+name)
        h=layer('dec2')(layer('merge2')(torch.cat([F.interpolate(h,size=e2.shape[-1],mode='linear',align_corners=False),e2],1)))
        coarse=h
        h=layer('dec1')(layer('merge1')(torch.cat([F.interpolate(h,size=e1.shape[-1],mode='linear',align_corners=False),e1],1)))
        mid=h
        h=layer('dec0')(layer('merge0')(torch.cat([F.interpolate(h,size=e0.shape[-1],mode='linear',align_corners=False),e0],1)))
        main=layer('main_head')(layer('combine')(torch.cat([h,point],1)))
        return (main,layer('aux_mid')(mid),layer('aux_coarse')(coarse)) if auxiliary else main
    @staticmethod
    def join(shared,sw):
        return torch.cat([shared[0][:,:,:2],sw[0][:,:,2:]],2),torch.cat([shared[1][:,:,:2],sw[1][:,:,2:]],2)
    def forward(self,features,relative_depth,valid,auxiliary=False,absolute_depth=None):
        mask=valid[:,None].to(features.dtype);point=features.transpose(1,2)*mask
        assert absolute_depth is not None
        x=torch.cat([point,relative_depth[:,None]*mask,mask,absolute_depth[:,None]*mask],1)
        e0=self.enc0(self.stem(x));e1=self.enc1(self.down1(e0));e2=self.enc2(self.down2(e1));h=self.bottleneck(self.down3(e2));point=self.point(point)
        shared=self.decode(h,e0,e1,e2,point,'',auxiliary);sw=self.decode(h,e0,e1,e2,point,'sw_',auxiliary)
        return tuple(self.join(a,b) for a,b in zip(shared,sw)) if auxiliary else self.join(shared,sw)
if __name__=='__main__':
    torch.set_num_threads(2);torch.manual_seed(34);base=Baseline(features=12).eval()
    torch.manual_seed(34);model=Network(features=12).eval()
    x=torch.randn(2,256,12);d=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:32]=False
    with torch.no_grad():
        assert all(torch.equal(a,b) for a,b in zip(base(x,d,v,absolute_depth=d),model(x,d,v,absolute_depth=d)))
        a=model(x,d,v,absolute_depth=d);xx=x.clone();xx[1]+=10;b=model(xx,d,v,absolute_depth=d)
        assert all(torch.equal(aa[0],bb[0]) for aa,bb in zip(a,b))
        xx=x.clone();xx[~v]+=1000;b=model(xx,d,v,absolute_depth=d)
        assert all(torch.equal(aa,bb) for aa,bb in zip(a,b))
    out=model(x,d,v,absolute_depth=d);out[1][:,:,2].sum().backward()
    assert model.sw_dec0.project.weight.grad.abs().sum()>0
    assert model.dec0.project.weight.grad.abs().sum()==0
    model.zero_grad(set_to_none=True)
    out=model(x,d,v,True,absolute_depth=d)
    loss=objective(out,torch.randn(2,256,3),torch.rand(2,256,3)>.5,v,True);loss.backward()
    assert all(torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None)
    assert model.dec0.project.weight.grad.abs().sum()>0 and model.sw_dec0.project.weight.grad.abs().sum()>0
    print('PASS: fresh-initialization equivalence, decoder SW gradient isolation, padded/well isolation and finite backward')
