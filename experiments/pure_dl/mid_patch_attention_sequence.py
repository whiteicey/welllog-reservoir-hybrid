"""Move local/global attention after enc1, at half spatial resolution."""
import torch
from torch import nn
from torch.nn import functional as F
from relative_only_sequence import Network as Baseline
from patch_hierarchy_sequence import SmallAttention


class MidHierarchy(nn.Module):
    def __init__(self):
        super().__init__();self.norm=nn.LayerNorm(96);self.input=nn.Linear(96,32,bias=False)
        self.local=SmallAttention(width=32,heads=4,maximum_tokens=8)
        self.global_patches=SmallAttention(width=32,heads=4,maximum_tokens=16)
        self.output=nn.Linear(32,96,bias=False);nn.init.zeros_(self.output.weight)

    def forward(self,x,valid):
        batch,channels,length=x.shape;assert length%8==0
        patches=length//8
        z=self.input(self.norm(x.transpose(1,2))).masked_fill(~valid[...,None],0)
        local=z.reshape(batch*patches,8,32);mask=valid.reshape(batch*patches,8)
        local=(local+self.local(local,mask)).masked_fill(~mask[...,None],0)
        token=(local.sum(1)/mask.sum(1,keepdim=True).clamp_min(1)).reshape(batch,patches,32)
        context=self.global_patches(token,mask.any(1).reshape(batch,patches))
        mixed=local.reshape(batch,patches,8,32)+context[:,:,None,:]
        return self.output(mixed.reshape(batch,length,32)).masked_fill(~valid[...,None],0).transpose(1,2)


class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        with torch.random.fork_rng(devices=[]):self.mid=MidHierarchy()

    def forward(self,features,relative_depth,valid,auxiliary=False,absolute_depth=None):
        mask=valid[:,None].to(features.dtype);point=features.transpose(1,2)*mask
        x=torch.cat([point,relative_depth[:,None]*mask,mask,absolute_depth[:,None]*mask],dim=1)
        e0=self.enc0(self.stem(x));e1=self.enc1(self.down1(e0))
        mid_valid=F.max_pool1d(valid[:,None].float(),2,2)[:,0]>0
        e1=e1+self.mid(e1,mid_valid)
        e2=self.enc2(self.down2(e1));h=self.bottleneck(self.down3(e2))
        h=self.dec2(self.merge2(torch.cat([F.interpolate(h,size=e2.shape[-1],mode='linear',align_corners=False),e2],1)));coarse=h
        h=self.dec1(self.merge1(torch.cat([F.interpolate(h,size=e1.shape[-1],mode='linear',align_corners=False),e1],1)));mid=h
        h=self.dec0(self.merge0(torch.cat([F.interpolate(h,size=e0.shape[-1],mode='linear',align_corners=False),e0],1)))
        main=self.main_head(self.combine(torch.cat([h,self.point(point)],1)))
        return (main,self.aux_mid(mid),self.aux_coarse(coarse)) if auxiliary else main


if __name__=='__main__':
    from global_point_loss import objective
    torch.set_num_threads(2);torch.manual_seed(51);base=Baseline(features=12).eval()
    torch.manual_seed(51);m=Network(features=12).eval()
    x=torch.randn(2,256,12);d=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:31]=False
    with torch.no_grad():
        assert all(torch.equal(q,w) for q,w in zip(base(x,d,v,absolute_depth=d),m(x,d,v,absolute_depth=d)))
        nn.init.normal_(m.mid.output.weight,std=.02)
        expected=m(x,d,v,absolute_depth=d)
        changed=x.clone();changed[~v]+=100;dd=d.clone();dd[~v]+=100
        assert all(torch.equal(q,w) for q,w in zip(expected,m(changed,dd,v,absolute_depth=dd)))
        changed=x.clone();changed[1]+=100
        assert all(torch.equal(q[0],w[0]) for q,w in zip(expected,m(changed,d,v,absolute_depth=d)))
        assert (m.mid(torch.randn(2,96,128),torch.zeros(2,128,dtype=torch.bool))==0).all()
    loss=objective(m(x,d,v,True,absolute_depth=d),torch.randn(2,256,3),torch.rand(2,256,3)>.5,v,True,torch.ones(2,256));loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    print('PASS zero-branch baseline identity, odd padding boundaries, well isolation, empty context and finite gradients')
