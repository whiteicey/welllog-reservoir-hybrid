"""Individual point queries read other patch tokens; no all-points global attention."""
import math
import torch
from torch import nn
from patch_hierarchy_sequence import Network as Baseline, PatchHierarchy


class PointPatchRead(PatchHierarchy):
    def __init__(self,old):
        nn.Module.__init__(self)
        self.patch_size=old.patch_size
        self.norm=old.norm;self.input=old.input;self.local=old.local
        self.global_patches=old.global_patches;self.output=old.output
        self.read_norm=nn.LayerNorm(32);self.memory_norm=nn.LayerNorm(32)
        self.query=nn.Linear(32,32,bias=False);self.kv=nn.Linear(32,64,bias=False)
        self.read_project=nn.Linear(32,32,bias=False)
        self.read_bias=nn.Parameter(torch.zeros(4,31))

    def read(self,points,tokens,point_valid,token_valid):
        batch,length,width=points.shape;patches=tokens.shape[1]
        q=self.query(self.read_norm(points)).reshape(batch,length,4,8).transpose(1,2)
        memory=self.memory_norm(tokens).masked_fill(~token_valid[...,None],0)
        k,v=self.kv(memory).reshape(batch,patches,2,4,8).permute(2,0,3,1,4)
        own=torch.arange(length,device=points.device)//16
        ids=torch.arange(patches,device=points.device)
        allowed=token_valid[:,None,:]&(ids[None,None,:]!=own[None,:,None])
        scores=(q@k.transpose(-1,-2)).float()/math.sqrt(8)
        scores=scores+self.read_bias[:,own[:,None]-ids[None,:]+15]
        scores=scores.masked_fill(~allowed[:,None],-1e4)
        value=(scores.softmax(-1).to(v.dtype)@v).transpose(1,2).reshape(batch,length,32)
        result=self.read_project(value)
        return result.masked_fill(~(allowed.any(-1)&point_valid)[...,None],0)

    def forward(self,x,valid):
        batch,channels,length=x.shape;assert length%16==0 and length<=256
        patches=length//16
        z=self.input(self.norm(x.transpose(1,2))).masked_fill(~valid[...,None],0)
        local=z.reshape(batch*patches,16,32);mask=valid.reshape(batch*patches,16)
        local=(local+self.local(local,mask)).masked_fill(~mask[...,None],0)
        token=(local.sum(1)/mask.sum(1,keepdim=True).clamp_min(1)).reshape(batch,patches,32)
        token_valid=mask.any(1).reshape(batch,patches)
        memory=token+self.global_patches(token,token_valid)
        points=local.reshape(batch,length,32)
        mixed=points+self.read(points,memory,valid,token_valid)
        return self.output(mixed).masked_fill(~valid[...,None],0).transpose(1,2)


class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        with torch.random.fork_rng(devices=[]):
            self.stem.hierarchy=PointPatchRead(self.stem.hierarchy)


if __name__=='__main__':
    from global_point_loss import objective
    torch.set_num_threads(2);torch.manual_seed(23);a=Baseline(features=12).eval()
    torch.manual_seed(23);m=Network(features=12).eval()
    x=torch.randn(2,256,12);d=torch.randn(2,256);valid=torch.ones(2,256,dtype=torch.bool);valid[:,:32]=False
    with torch.no_grad():
        assert all(torch.equal(q,w) for q,w in zip(a(x,d,valid,absolute_depth=d),m(x,d,valid,absolute_depth=d)))
        nn.init.normal_(m.stem.hierarchy.output.weight,std=.02)
        expected=m(x,d,valid,absolute_depth=d)
        padded=x.clone();padded[~valid]+=100
        assert all(torch.equal(q,w) for q,w in zip(expected,m(padded,d,valid,absolute_depth=d)))
        other=x.clone();other[1]+=100
        assert all(torch.equal(q[0],w[0]) for q,w in zip(expected,m(other,d,valid,absolute_depth=d)))
        h=m.stem.hierarchy;p=torch.randn(1,32,32);tokens=torch.randn(1,2,32);pv=torch.ones(1,32,dtype=torch.bool);tv=torch.tensor([[True,False]])
        read=h.read(p,tokens,pv,tv)
        assert (read[:,:16]==0).all() and torch.isfinite(read).all()
        assert h.read(p,tokens,pv,torch.ones_like(tv)).abs().sum()>0
    loss=objective(m(x,d,valid,True,absolute_depth=d),torch.randn(2,256,3),torch.rand(2,256,3)>.5,valid,True,torch.ones(2,256));loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    print('PASS initial identity, padding/well isolation, own-patch exclusion, empty external patch set, finite gradients')
    print('PARAMETERS',sum(p.numel() for p in Network().parameters()))
