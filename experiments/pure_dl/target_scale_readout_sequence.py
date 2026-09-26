"""Target-identity queries read three decoder scales; only main density changes."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from global_cosine_sequence import Network as Baseline


class TargetScaleReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.projections=nn.ModuleList([nn.Linear(c,32,bias=False) for c in [128,96,128]])
        self.norms=nn.ModuleList([nn.LayerNorm(32) for _ in range(3)])
        self.query=nn.Linear(128,32,bias=False)
        self.key=nn.Linear(32,32,bias=False)
        self.target_embedding=nn.Parameter(torch.randn(3,32)*.02)
        self.output=nn.Linear(32,128,bias=False)
        nn.init.zeros_(self.output.weight)

    def route(self,fine,mid,coarse,valid):
        # The target embedding is just POR/PERM/SW identity, never a label.
        length=fine.shape[-1]
        assert valid.shape==(fine.shape[0],length)
        values=[]
        for scale,(feature,projection,norm) in enumerate(zip([fine,mid,coarse],self.projections,self.norms)):
            factor=2**scale
            source_valid=(F.max_pool1d(valid[:,None].float(),factor,factor)>0) if factor>1 else valid[:,None]
            assert source_valid.shape[-1]==feature.shape[-1]
            aligned=feature.masked_fill(~source_valid,0)
            if scale:aligned=F.interpolate(aligned,size=length,mode='linear',align_corners=False)
            values.append(norm(projection(aligned.transpose(1,2))).masked_fill(~valid[...,None],0))
        values=torch.stack(values,2)  # B,L,3 scales,32
        key=self.key(values)
        query=self.query(fine.transpose(1,2)).unsqueeze(2)+self.target_embedding[None,None]
        scores=torch.einsum('bltd,blsd->blts',query.float(),key.float())/math.sqrt(32.)
        weights=scores.softmax(-1).masked_fill(~valid[:,:,None,None],0)
        context=torch.einsum('blts,blsd->bltd',weights.to(values.dtype),values)
        return self.output(context),weights

    def forward(self,fine,mid,coarse,valid):
        return self.route(fine,mid,coarse,valid)[0]


class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        with torch.random.fork_rng(devices=[]):self.target_scale=TargetScaleReadout()

    def forward(self,features,relative_depth,valid,auxiliary=False,absolute_depth=None):
        mask=valid[:,None].to(features.dtype)
        point=features.transpose(1,2)*mask
        assert absolute_depth is not None and absolute_depth.shape==valid.shape
        x=torch.cat([point,relative_depth[:,None]*mask,mask,absolute_depth[:,None]*mask],1)
        e0=self.enc0(self.stem(x));e1=self.enc1(self.down1(e0));e2=self.enc2(self.down2(e1))
        h=self.bottleneck(self.down3(e2))
        h=self.dec2(self.merge2(torch.cat([F.interpolate(h,size=e2.shape[-1],mode='linear',align_corners=False),e2],1)))
        coarse=h
        h=self.dec1(self.merge1(torch.cat([F.interpolate(h,size=e1.shape[-1],mode='linear',align_corners=False),e1],1)))
        mid=h
        h=self.dec0(self.merge0(torch.cat([F.interpolate(h,size=e0.shape[-1],mode='linear',align_corners=False),e0],1)))
        fine=self.combine(torch.cat([h,self.point(point)],1))
        gate,raw=self.main_head(fine)
        residual=self.target_scale(fine,mid,coarse,valid)
        # Reuse the original target's24 density output rows; no new private head.
        density_rows=self.main_head.density.weight[:,:,0].reshape(3,24,128)
        delta=torch.einsum('bltc,tkc->bltk',residual,density_rows).reshape_as(raw)
        main=(gate,raw+delta)
        return (main,self.aux_mid(mid),self.aux_coarse(coarse)) if auxiliary else main
