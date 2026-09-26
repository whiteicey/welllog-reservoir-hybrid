"""Local point attention plus global patch-token attention, with masked pooling."""
import math
import torch
from torch import nn
from relative_only_sequence import Network as Baseline, RelativeOnlyStem


class SmallAttention(nn.Module):
    def __init__(self, width=32, heads=4, maximum_tokens=16):
        super().__init__();self.heads=heads;self.maximum_tokens=maximum_tokens
        self.norm=nn.LayerNorm(width)
        self.qkv=nn.Linear(width,width*3,bias=False)
        self.project=nn.Linear(width,width,bias=False)
        self.relative_bias=nn.Parameter(torch.zeros(heads,2*maximum_tokens-1))

    def forward(self,x,valid):
        batch,length,width=x.shape;assert length<=self.maximum_tokens
        z=self.norm(x).masked_fill(~valid[...,None],0)
        q,k,v=self.qkv(z).reshape(batch,length,3,self.heads,width//self.heads).permute(2,0,3,1,4)
        scores=(q@k.transpose(-2,-1)).float()/math.sqrt(width//self.heads)
        pos=torch.arange(length,device=x.device)
        bias=self.relative_bias[:,pos[:,None]-pos[None,:]+self.maximum_tokens-1]
        scores=(scores+bias).masked_fill(~valid[:,None,None,:],-1e4)
        attended=(scores.softmax(-1).to(v.dtype)@v).transpose(1,2).reshape(batch,length,width)
        return self.project(attended).masked_fill(~valid[...,None],0)


class PatchHierarchy(nn.Module):
    def __init__(self):
        super().__init__();self.patch_size=16
        self.norm=nn.LayerNorm(64);self.input=nn.Linear(64,32,bias=False)
        self.local=SmallAttention();self.global_patches=SmallAttention()
        self.output=nn.Linear(32,64,bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self,x,valid):
        batch,channels,length=x.shape;assert length%16==0
        patches=length//16
        z=self.input(self.norm(x.transpose(1,2))).masked_fill(~valid[...,None],0)
        local=z.reshape(batch*patches,16,32);mask=valid.reshape(batch*patches,16)
        local=(local+self.local(local,mask)).masked_fill(~mask[...,None],0)
        token=local.sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        token=token.reshape(batch,patches,32);token_valid=mask.any(1).reshape(batch,patches)
        global_context=self.global_patches(token,token_valid)
        mixed=local.reshape(batch,patches,16,32)+global_context[:,:,None,:]
        return self.output(mixed.reshape(batch,length,32)).masked_fill(~valid[...,None],0).transpose(1,2)


class PatchStem(RelativeOnlyStem):
    def __init__(self,old):
        super().__init__(old.in_channels,old.out_channels,1)
        with torch.no_grad():self.weight.copy_(old.weight);self.bias.copy_(old.bias)
        self.hierarchy=PatchHierarchy()

    def forward(self,x):
        stem=super().forward(x)
        return stem+self.hierarchy(stem,x[:,-2,:]>0)


class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        with torch.random.fork_rng(devices=[]):self.stem=PatchStem(self.stem)


if __name__=='__main__':
    from global_point_loss import objective
    torch.set_num_threads(2);torch.manual_seed(37);a=Baseline(features=12).eval()
    torch.manual_seed(37);m=Network(features=12).eval()
    x=torch.randn(2,256,12);d=torch.randn(2,256);v=torch.ones(2,256,dtype=torch.bool);v[:,:32]=False
    with torch.no_grad():
        expected=a(x,d,v,absolute_depth=d)
        assert all(torch.equal(q,w) for q,w in zip(expected,m(x,d,v,absolute_depth=d)))
        nn.init.normal_(m.stem.hierarchy.output.weight,std=.02)
        expected=m(x,d,v,absolute_depth=d)
        changed=x.clone();changed[~v]+=100
        assert all(torch.equal(q,w) for q,w in zip(expected,m(changed,d,v,absolute_depth=d)))
        changed=x.clone();changed[1]+=100
        assert all(torch.equal(q[0],w[0]) for q,w in zip(expected,m(changed,d,v,absolute_depth=d)))
        # Global tokens really transmit information between patches.
        h=m.stem.hierarchy;features=torch.randn(1,64,256);mask=torch.ones(1,256,dtype=torch.bool)
        first=h(features,mask);features[:,:,128:144]+=torch.randn(1,64,16)
        assert not torch.equal(first[:,:,:16],h(features,mask)[:,:,:16])
        assert torch.isfinite(h(features,torch.zeros_like(mask))).all()
        assert (h(features,torch.zeros_like(mask))==0).all()
    loss=objective(m(x,d,v,True,absolute_depth=d),torch.randn(2,256,3),torch.rand(2,256,3)>.5,v,True,torch.ones(2,256))
    loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    print('PASS initial identity, padding/well isolation, real inter-patch communication, empty patches and finite gradients')
    print('PARAMETERS',sum(p.numel() for p in Network().parameters()))
