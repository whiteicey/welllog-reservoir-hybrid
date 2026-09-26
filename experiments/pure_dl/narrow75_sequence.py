"""Global-cosine SOTA backbone narrowed0.75x; fixed original distribution/loss."""
import torch
from torch import nn
from torch.nn import functional as F
from relative_only_sequence import RelativeOnlyStem
from blur_no_point_sequence import EmptyPoint
from depthwise_norm_sequence import DepthNorm
from blur_down_sequence import BlurDown
from patch_hierarchy_sequence import SmallAttention as DotAttention
from cosine_patch_sequence import SmallAttention as CosineAttention
from multiscale_sequence import DensityHead


class Block(nn.Module):
    def __init__(self,width,dilation=1):
        super().__init__()
        self.norm=DepthNorm(width,eps=1e-5)
        self.depthwise=nn.Conv1d(width,width,7,padding=3*dilation,dilation=dilation,groups=width)
        self.expand=nn.Conv1d(width,width,1)
        self.project=nn.Conv1d(width,width,1)
        self.dropout=nn.Dropout(0.)
        self.gain=nn.Parameter(torch.full((1,width,1),.1))
    def forward(self,x):
        return x+self.gain*self.project(self.dropout(.5*F.gelu(self.expand(self.depthwise(self.norm(x))))))


class Hierarchy(nn.Module):
    def __init__(self):
        super().__init__();self.patch_size=16
        self.norm=nn.LayerNorm(48);self.input=nn.Linear(48,24,bias=False)
        self.local=DotAttention(width=24,heads=4,maximum_tokens=16)
        self.global_patches=CosineAttention(width=24,heads=4,maximum_tokens=16)
        self.output=nn.Linear(24,48,bias=False);nn.init.zeros_(self.output.weight)
    def forward(self,x,valid):
        batch,channels,length=x.shape;assert length%16==0
        patches=length//16
        z=self.input(self.norm(x.transpose(1,2))).masked_fill(~valid[...,None],0)
        local=z.reshape(batch*patches,16,24);mask=valid.reshape(batch*patches,16)
        local=(local+self.local(local,mask)).masked_fill(~mask[...,None],0)
        token=local.sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        token=token.reshape(batch,patches,24);token_valid=mask.any(1).reshape(batch,patches)
        context=self.global_patches(token,token_valid)
        mixed=local.reshape(batch,patches,16,24)+context[:,:,None,:]
        return self.output(mixed.reshape(batch,length,24)).masked_fill(~valid[...,None],0).transpose(1,2)


class Stem(RelativeOnlyStem):
    def __init__(self,features):
        super().__init__(features+2,48,1)
        self.hierarchy=Hierarchy()
    def forward(self,x):
        stem=super().forward(x)
        return stem+self.hierarchy(stem,x[:,-2,:]>0)


class Network(nn.Module):
    def __init__(self,features=206,initial_means=None):
        super().__init__()
        self.stem=Stem(features)
        self.enc0=Block(48)
        self.down1=BlurDown(nn.Conv1d(48,72,4,stride=2,padding=1));self.enc1=Block(72)
        self.down2=BlurDown(nn.Conv1d(72,96,4,stride=2,padding=1));self.enc2=Block(96,2)
        self.down3=BlurDown(nn.Conv1d(96,144,4,stride=2,padding=1))
        self.bottleneck=nn.Sequential(Block(144,1),Block(144,2),Block(144,4))
        self.merge2=nn.Conv1d(144+96,96,1);self.dec2=Block(96)
        self.merge1=nn.Conv1d(96+72,72,1);self.dec1=Block(72)
        self.merge0=nn.Conv1d(72+48,48,1);self.dec0=Block(48)
        self.point=EmptyPoint()
        self.combine=nn.Sequential(nn.Conv1d(48,96,1),nn.GELU(),nn.Dropout(0.))
        self.main_head=DensityHead(96,initial_means)
        self.aux_mid=DensityHead(72,initial_means)
        self.aux_coarse=DensityHead(96,initial_means)

    def forward(self,features,relative_depth,valid,auxiliary=False,absolute_depth=None):
        mask=valid[:,None].to(features.dtype);point=features.transpose(1,2)*mask
        assert absolute_depth is not None and absolute_depth.shape==valid.shape
        x=torch.cat([point,relative_depth[:,None]*mask,mask,absolute_depth[:,None]*mask],1)
        e0=self.enc0(self.stem(x));e1=self.enc1(self.down1(e0));e2=self.enc2(self.down2(e1))
        h=self.bottleneck(self.down3(e2))
        h=self.dec2(self.merge2(torch.cat([F.interpolate(h,size=e2.shape[-1],mode='linear',align_corners=False),e2],1)));coarse=h
        h=self.dec1(self.merge1(torch.cat([F.interpolate(h,size=e1.shape[-1],mode='linear',align_corners=False),e1],1)));mid=h
        h=self.dec0(self.merge0(torch.cat([F.interpolate(h,size=e0.shape[-1],mode='linear',align_corners=False),e0],1)))
        main=self.main_head(self.combine(torch.cat([h,self.point(point)],1)))
        return (main,self.aux_mid(mid),self.aux_coarse(coarse)) if auxiliary else main
