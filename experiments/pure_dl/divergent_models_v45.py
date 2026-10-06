"""New private predictors: long-section SW and masked curve-depth axial POR.

All network tensors are randomly initialized. Only the retained fold-local data
normalization, quality weights and original physical decoding grid are reused.
"""
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from independent_sw_scratch_v10 import TemporalBlock
from train_scaling_v21 import Evaluator as OriginalEvaluator
from por_t3_patch_loss import probability
from multiscale_sequence import window_indices
from crosscurve_v32 import PREVIOUS,reference_prediction
from champion_scaling_v21 import load_data as load_original_data
import train_invariant as base

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/divergent_models_v45_20260929'
MODES={'long_section':dict(target='SW',c=2,length=1024),
       'curve_axial':dict(target='POR',c=0,length=512)}
CURVES=('GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT')

def load_data():
    data,old=load_original_data()
    metadata={k:old[k] for k in ('centers','density_edges','selected_feature_names')}
    del old
    return data,metadata

class Attention(nn.Module):
    def __init__(self,width,heads=4):
        super().__init__();self.heads=heads;self.width=width
        self.norm=nn.LayerNorm(width);self.qkv=nn.Linear(width,width*3)
        self.project=nn.Linear(width,width)
        self.ffnorm=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width))

    def forward(self,x,valid):
        b,t,c=x.shape
        q,k,v=self.qkv(self.norm(x)).reshape(b,t,3,self.heads,c//self.heads).permute(2,0,3,1,4).unbind(0)
        # Rows always have at least one valid key (each window has its anchor).
        z=F.scaled_dot_product_attention(q,k,v,attn_mask=valid[:,None,None,:],dropout_p=0.)
        x=(x+self.project(z.transpose(1,2).reshape(b,t,c))).masked_fill(~valid[...,None],0)
        return (x+self.ff(self.ffnorm(x))).masked_fill(~valid[...,None],0)

class LongSection(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Linear(207,96)
        self.local=nn.ModuleList([TemporalBlock(96,d) for d in (1,2,4)])
        self.token_pos=nn.Linear(2,96)
        self.global_blocks=nn.ModuleList([Attention(96) for _ in range(4)])
        self.fuse=nn.Linear(192,128)
        self.detail=nn.ModuleList([TemporalBlock(128,d) for d in (1,2)])

    def forward(self,x,relative,valid,generator=None):
        b,l,_=x.shape
        h=F.gelu(self.stem(torch.cat((x,relative[...,None]),-1))).masked_fill(~valid[...,None],0)
        for block in self.local:h=block(h,valid)
        pv=valid.reshape(b,l//16,16);n=pv.sum(-1).clamp_min(1)
        token=(h.reshape(b,l//16,16,96)*pv[...,None]).sum(2)/n[...,None]
        positions=torch.linspace(-1,1,l,device=x.device).expand(b,-1)
        loc=torch.stack((relative,positions),-1).reshape(b,l//16,16,2)
        loc=(loc*pv[...,None]).sum(2)/n[...,None]
        tv=pv.any(-1);token=(token+self.token_pos(loc)).masked_fill(~tv[...,None],0)
        for block in self.global_blocks:token=block(token,tv)
        global_h=F.interpolate(token.transpose(1,2),size=l,mode='linear',align_corners=False).transpose(1,2)
        h=F.gelu(self.fuse(torch.cat((h,global_h),-1))).masked_fill(~valid[...,None],0)
        for block in self.detail:h=block(h,valid)
        return h,h.new_zeros(()),None

class AxialBlock(nn.Module):
    def __init__(self):
        super().__init__();self.curves=Attention(64);self.depth=Attention(64)

    def forward(self,h,valid):
        b,t,c,d=h.shape
        cv=valid[:,:,None].expand(b,t,c).reshape(b*t,c)
        # Entire padded patches get a harmless valid key; masked after the pass.
        cv=cv.clone();cv[:,0]=True
        h=self.curves(h.reshape(b*t,c,d),cv).reshape(b,t,c,d)
        h=h.masked_fill(~valid[:,:,None,None],0)
        tv=valid[:,None,:].expand(b,c,t).reshape(b*c,t)
        h=self.depth(h.permute(0,2,1,3).reshape(b*c,t,d),tv)
        return h.reshape(b,c,t,d).permute(0,2,1,3).masked_fill(~valid[:,:,None,None],0)

class CurveAxial(nn.Module):
    def __init__(self,indices):
        super().__init__();self.register_buffer('indices',torch.tensor(indices))
        self.patch=nn.Linear(16,64);self.curve_id=nn.Parameter(torch.randn(1,1,9,64)*.02)
        self.position=nn.Linear(2,64)
        self.blocks=nn.ModuleList([AxialBlock() for _ in range(3)])
        self.reconstruct=nn.Linear(64,8)
        self.merge_curves=nn.Linear(9*64,128);self.local=nn.Linear(10,64)
        self.fuse=nn.Linear(192,128)
        self.detail=nn.ModuleList([TemporalBlock(128,d) for d in (1,2)])

    def forward(self,x,relative,valid,generator=None):
        raw=x.index_select(-1,self.indices).float();b,l,_=raw.shape;t=l//8
        target=raw.reshape(b,t,8,9).permute(0,1,3,2)
        pv=valid.reshape(b,t,8);tv=pv.any(-1)
        masked=torch.zeros((b,t,9),device=x.device,dtype=torch.bool)
        if self.training and generator is not None:
            masked=torch.rand((b,t,9),device=x.device,generator=generator)<.25
            masked&=tv[...,None]
        active=masked[...,None]&pv[:,:,None,:]
        corrupt=target.masked_fill(active,0).masked_fill(~pv[:,:,None,:],0)
        flags=active.to(corrupt.dtype)
        h=self.patch(torch.cat((corrupt,flags),-1))+self.curve_id
        pos=torch.linspace(-1,1,l,device=x.device).expand(b,-1)
        loc=torch.stack((relative,pos),-1).reshape(b,t,8,2)
        loc=(loc*pv[...,None]).sum(2)/pv.sum(-1).clamp_min(1)[...,None]
        h=(h+self.position(loc)[:,:,None,:]).masked_fill(~tv[:,:,None,None],0)
        for block in self.blocks:h=block(h,tv)
        auxiliary=h.new_zeros((),dtype=torch.float32)
        if self.training and generator is not None:
            reconstructed=self.reconstruct(h).float()
            auxiliary=F.smooth_l1_loss(reconstructed[active],target[active],beta=1.)
        # Fine detail path sees the same corrupted curves; no hidden bypass.
        clean_local=corrupt.permute(0,1,3,2).reshape(b,l,9)
        local=F.gelu(self.local(torch.cat((clean_local,relative[...,None]),-1)))
        coarse=self.merge_curves(h.reshape(b,t,9*64))
        coarse=F.interpolate(coarse.transpose(1,2),size=l,mode='linear',align_corners=False).transpose(1,2)
        result=F.gelu(self.fuse(torch.cat((coarse,local),-1))).masked_fill(~valid[...,None],0)
        for block in self.detail:result=block(result,valid)
        return result,auxiliary,dict(masked_values=int(active.sum()),valid_values=int(valid.sum())*9)

class Network(nn.Module):
    def __init__(self,mode,metadata):
        super().__init__();self.mode=mode;self.c=MODES[mode]['c'];self.length=MODES[mode]['length']
        self.body=(LongSection() if mode=='long_section' else CurveAxial([metadata['selected_feature_names'].index(n) for n in CURVES]))
        self.norm=nn.LayerNorm(128);self.output=nn.Linear(128,25)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,x,relative,valid,absolute_depth=None,return_aux=False,generator=None):
        h,aux,maskinfo=self.body(x.float(),relative.float(),valid,generator)
        y=self.output(self.norm(h)).float()
        output=(y[...,:1],y[...,1:].reshape(*y.shape[:2],1,3,8))
        return (output,aux,maskinfo) if return_aux else output

    @torch.no_grad()
    def initialize_priors(self,data):
        ids=data['train_raw'];d=data['default'][ids,self.c];q=data['quality'][ids].float()
        rate=(q*d.float()).sum()/q.sum()
        z=data['labels'][ids,self.c][~d].float()
        means=torch.quantile(z,torch.linspace(.06,.94,8,device=z.device))
        self.output.bias[0].copy_(torch.logit(rate.clamp(1e-5,1-1e-5)))
        r=self.output.bias[1:].view(3,8);r[0].zero_();r[1].copy_(means)
        r[2].fill_(math.log(math.expm1(.5-.025)))
        return dict(default_rate=float(rate),means=means.cpu().tolist(),component_sd=.5,
                    training_rows=len(ids),scope='only original64 train labels/quality; random body, zero output weights')

class Evaluator(OriginalEvaluator):
    @torch.no_grad()
    def __call__(self,model):
        model.eval();d=self.data;c=self.c;length=model.length;lo=length//2-64;hi=lo+128
        result=np.full(len(d['validation']),np.nan,np.float64);coverage=np.zeros(len(result),np.int8)
        for begin in range(0,len(self.centers),16):
            centers=torch.tensor(self.centers[begin:begin+16],device='cuda');anchors=torch.tensor(self.anchors[begin:begin+16],device='cuda')
            ix,valid=window_indices(centers,d['lower'],d['upper'],length=length,anchors=anchors)
            relative=torch.tanh((d['depth'][ix]-d['depth'][anchors,None])/10)
            with torch.autocast('cuda',dtype=torch.bfloat16):go,raw=model(d['x'][ix],relative,valid)
            rows=d['raw_to_valid'][ix[:,lo:hi].cpu().numpy()];use=valid[:,lo:hi].cpu().numpy()&(rows>=0)
            dest=self.local[rows[use]]
            assert (dest>=0).all() and (coverage[dest]==0).all() and len(np.unique(dest))==len(dest)
            coverage[dest]+=1;take=torch.as_tensor(use,device='cuda')
            g=go[:,lo:hi][take][:,0].float().sigmoid();r=raw[:,lo:hi][take][:,0].float()
            cp=probability(c,r[:,0].softmax(-1),r[:,1],F.softplus(r[:,2])+.025,self.edges)
            joint=torch.cat((g[:,None],(1-g[:,None])*cp),-1)
            result[dest]=self.actions[(joint@self.utilities).argmax(-1).cpu().numpy()]
        assert (coverage==1).all() and np.isfinite(result).all()
        combined=self.reference.copy();combined[:,c]=result;scores=base.metric(d['truth'][d['validation']],combined)
        return result,float(scores[('POR','PERM','SW')[c]]),scores
