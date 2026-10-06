"""Frozen SW attention with symmetric versus directional physical-depth bias."""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sw_interaction_v56 import ROOT,Network as Parent,load_data
from well_conditioned_v46 import modulate,Evaluator as OriginalEvaluator
from multiscale_sequence import window_indices
from por_t3_patch_loss import probability
import train_invariant as base
STUDY=ROOT/'results/deep_learning/sw_depth_attention_v71_20260929'
PREVIOUS=ROOT/'results/deep_learning/perm_cdfwarp_v70_20260929'
SOURCE=ROOT/'outputs/pure_dl_sw_interaction_v56_SW_additive_fold5/model_best.pt'
PARENT_PRED=SOURCE.with_name('channel_pred.npy')
MODES={m:dict(target='SW',c=2,length=1024) for m in ('symmetric','directional')}
def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')
class DepthAttention(nn.Module):
    def __init__(self,old,mode):
        super().__init__();self.old=old;self.study_mode=mode;self.coefficients=nn.Parameter(torch.zeros(old.heads,8));self.coordinates=None
    def forward(self,x,valid):
        if self.coordinates is None:raise ValueError('Physical token coordinates required')
        b,t,c=x.shape;o=self.old
        q,k,v=o.qkv(o.norm(x)).reshape(b,t,3,o.heads,c//o.heads).permute(2,0,3,1,4).unbind(0)
        original=F.scaled_dot_product_attention(q,k,v,attn_mask=valid[:,None,None,:],dropout_p=0.)
        difference=torch.tanh((self.coordinates[:,:,None]-self.coordinates[:,None,:])/10)
        if self.study_mode=='symmetric':difference=difference.abs();centers=torch.linspace(0,1,8,device=x.device)
        else:centers=torch.linspace(-1,1,8,device=x.device)
        # Equal eight smooth distance bases and parameter count; sign availability differs.
        bandwidth=1/7 if self.study_mode=='symmetric' else 2/7
        basis=(-.5*((difference[...,None]-centers)/bandwidth).square()).softmax(-1)
        bias=torch.einsum('bijr,hr->bhij',basis,.5*torch.tanh(self.coefficients))
        zero=torch.zeros_like(bias).masked_fill(~valid[:,None,None,:],float('-inf'))
        modified=F.scaled_dot_product_attention(q,k,v,attn_mask=zero+bias,dropout_p=0.)
        reference=F.scaled_dot_product_attention(q,k,v,attn_mask=zero,dropout_p=0.)
        z=original+(modified-reference)
        x=(x+o.project(z.transpose(1,2).reshape(b,t,c))).masked_fill(~valid[...,None],0)
        return (x+o.ff(o.ffnorm(x))).masked_fill(~valid[...,None],0)
class Network(Parent):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('additive',metadata)
        saved=torch.load(SOURCE,map_location='cpu',weights_only=False);self.load_state_dict(saved['state_dict'],strict=True)
        self.study_mode=mode;self.requires_grad_(False)
        self.body.global_blocks=nn.ModuleList([DepthAttention(block,mode) for block in self.body.global_blocks])
        self.initialization_info=dict(checkpoint=str(SOURCE.relative_to(ROOT)),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            best_epoch=saved['best_epoch'],target_score=saved['target_score'],optimizer_reset=True,
            initialization='exact parent via original attention plus biased-minus-zero additive-mask attention',new_parameters=128,
            comparison='same128 zero coefficients; absolute or signed physical token depth difference /10 tanh; eight normalized RBF bases')
    def initialize_priors(self,data):return self.initialization_info
    def core_hash(self):
        trainable={n for n,p in self.named_parameters() if p.requires_grad}
        return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for n,v in self.state_dict().items() if n not in trainable)).hexdigest()
    def train(self,mode=True):super().train(False);self.training=mode;return self
    def forward(self,x,relative,valid,absolute_depth=None,return_aux=False,generator=None,well_inputs=None):
        if absolute_depth is None:raise ValueError('Raw physical depth needed only for translation-invariant pair differences')
        b,l,_=x.shape;c=self.body;modulation=self.condition(well_inputs)
        h=F.gelu(c.stem(torch.cat((x,relative[...,None]),-1))).masked_fill(~valid[...,None],0)
        h=modulate(h,modulation[0],valid)
        for block in c.local:h=block(h,valid)
        pv=valid.reshape(b,l//16,16);n=pv.sum(-1).clamp_min(1)
        token=(h.reshape(b,l//16,16,96)*pv[...,None]).sum(2)/n[...,None]
        pos=torch.linspace(-1,1,l,device=x.device).expand(b,-1)
        loc=torch.stack((relative,pos),-1).reshape(b,l//16,16,2);loc=(loc*pv[...,None]).sum(2)/n[...,None]
        tv=pv.any(-1);token=(token+c.token_pos(loc)).masked_fill(~tv[...,None],0)
        # Center before averaging: no absolute-depth feature enters the learned bias.
        centered=absolute_depth.float()-absolute_depth[:,l//2:l//2+1].float()
        coordinate=(centered.reshape(b,l//16,16)*pv).sum(-1)/n
        for j,block in enumerate(c.global_blocks):
            block.coordinates=coordinate
            try:token=modulate(block(token,tv),modulation[j+1],tv)
            finally:block.coordinates=None
        global_h=F.interpolate(token.transpose(1,2),size=l,mode='linear',align_corners=False).transpose(1,2)
        h=F.gelu(c.fuse(torch.cat((h,global_h),-1))).masked_fill(~valid[...,None],0);h=modulate(h,modulation[5],valid)
        for block in c.detail:h=block(h,valid)
        y=self.output(self.norm(h)).float();output=(y[...,:1],y[...,1:].reshape(*y.shape[:2],1,3,8))
        return (output,y.new_zeros(()),None) if return_aux else output
class Evaluator(OriginalEvaluator):
    @torch.no_grad()
    def __call__(self,model):
        model.eval();d=self.data;c=self.c;length=model.length;lo=length//2-64;hi=lo+128
        result=np.full(len(d['validation']),np.nan,np.float64);coverage=np.zeros(len(result),np.int8)
        for begin in range(0,len(self.centers),16):
            centers=torch.tensor(self.centers[begin:begin+16],device='cuda');anchors=torch.tensor(self.anchors[begin:begin+16],device='cuda')
            ix,valid=window_indices(centers,d['lower'],d['upper'],length=length,anchors=anchors)
            rel=torch.tanh((d['depth'][ix]-d['depth'][anchors,None])/10)
            with torch.autocast('cuda',dtype=torch.bfloat16):go,raw=model(d['x'][ix],rel,valid,absolute_depth=d['depth'][ix],well_inputs=d['well_bank'][d['row_well'][anchors]])
            rows=d['raw_to_valid'][ix[:,lo:hi].cpu().numpy()];use=valid[:,lo:hi].cpu().numpy()&(rows>=0);dest=self.local[rows[use]]
            assert (dest>=0).all() and (coverage[dest]==0).all() and len(np.unique(dest))==len(dest)
            coverage[dest]+=1;take=torch.as_tensor(use,device='cuda')
            g=go[:,lo:hi][take][:,0].float().sigmoid();r=raw[:,lo:hi][take][:,0].float()
            cp=probability(c,r[:,0].softmax(-1),r[:,1],F.softplus(r[:,2])+.025,self.edges)
            mass=torch.cat((g[:,None],(1-g[:,None])*cp),-1)
            result[dest]=self.actions[(mass@self.utilities).argmax(-1).cpu().numpy()]
        assert (coverage==1).all() and np.isfinite(result).all()
        combined=self.reference.copy();combined[:,c]=result;s=base.metric(d['truth'][d['validation']],combined)
        return result,float(s['SW']),s
