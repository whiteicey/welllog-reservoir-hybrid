"""Train a whole-well input encoder to condition the new long-section network."""
import hashlib,json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from divergent_models_v45 import ROOT,Network as Original,load_data as original_data
from divergent_models_v45 import Evaluator as OriginalEvaluator
from multiscale_sequence import window_indices
from por_t3_patch_loss import probability
import train_invariant as base

STUDY=ROOT/'results/deep_learning/well_conditioned_v46_20260929'
PREVIOUS=ROOT/'results/deep_learning/divergent_models_v45_20260929'
PARENT=ROOT/'outputs/pure_dl_divergent_v45_SW_long_section_fold5/model_best.pt'
MODES={m:dict(target='SW',c=2,length=1024) for m in ('reference','well')}
WIDTHS=(96,96,96,96,96,128)

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

def load_data():
    data,metadata=original_data();starts=np.asarray(data['starts'])
    sample=np.stack([lo+np.minimum(((np.arange(128)+.5)*(hi-lo)/128).astype(np.int64),hi-lo-1) for lo,hi in zip(starts[:-1],starts[1:])])
    row_well=np.repeat(np.arange(len(starts)-1),np.diff(starts))
    ids=data['train_raw'].cpu().numpy();train_wells=np.unique(row_well[ids])
    val_rows=np.flatnonzero(np.isin(row_well,np.setdiff1d(np.arange(80),train_wells)))
    assert len(train_wells)==64 and not np.isin(ids,val_rows).any()
    bank=data['x'][torch.as_tensor(sample,device='cuda')].detach()
    data['well_bank']=bank;data['row_well']=torch.as_tensor(row_well,device='cuda')
    metadata['reference_inputs']=bank[torch.as_tensor(train_wells,device='cuda')].mean(0)
    metadata['bank_manifest']=dict(points_per_well=128,features=206,train_wells=train_wells.tolist(),
        sample_raw_rows=sample.tolist(),sample_sha256=hashlib.sha256(sample.tobytes()).hexdigest(),
        reference='positionwise equal-well average of64 training input banks only',
        context='deterministic depth-stratified input-only samples; heldout well bank used only by its inference; no label/ID feature',
        bank_shape=list(bank.shape),validation_fit=False,bank_labels_used=False)
    return data,metadata

class WellEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.point=nn.Sequential(nn.Linear(206,64),nn.GELU(),nn.Linear(64,64),nn.GELU())
        self.code=nn.Sequential(nn.Linear(192,96),nn.GELU())
        self.output=nn.Linear(96,2*sum(WIDTHS))
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,inputs):
        z=self.point(inputs)
        stats=torch.cat((z.mean(1),z.var(1,unbiased=False).add(1e-6).sqrt(),z.amax(1)),1)
        raw=.5*torch.tanh(self.output(self.code(stats)).float())
        return tuple(raw.split([2*w for w in WIDTHS],-1))

def modulate(x,values,valid):
    gamma,beta=values.chunk(2,-1)
    return (x.float()*(1+gamma[:,None])+beta[:,None]).to(x.dtype).masked_fill(~valid[...,None],0)

class Network(Original):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('long_section',metadata)
        parent=torch.load(PARENT,map_location='cpu',weights_only=False)
        self.load_state_dict(parent['state_dict'],strict=True)
        self.mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261146);self.condition=WellEncoder()
        self.register_buffer('reference_inputs',metadata['reference_inputs'].detach().clone(),persistent=False)
        self.parent_info=dict(checkpoint=str(PARENT.relative_to(ROOT)),sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),
            best_epoch=parent['best_epoch'],target_score=parent['target_score'],optimizer_reset=True,
            initialization='v45 own-training-fold model plus identical zero-output well encoder; all parameters trainable')

    def initialize_priors(self,data):return self.parent_info

    def forward(self,x,relative,valid,absolute_depth=None,return_aux=False,generator=None,well_inputs=None):
        if well_inputs is None:raise ValueError('Explicit input-only well bank required')
        b,l,_=x.shape;c=self.body
        context=well_inputs if self.mode=='well' else self.reference_inputs[None].expand(b,-1,-1)
        modulation=self.condition(context)
        h=F.gelu(c.stem(torch.cat((x,relative[...,None]),-1))).masked_fill(~valid[...,None],0)
        h=modulate(h,modulation[0],valid)
        for block in c.local:h=block(h,valid)
        pv=valid.reshape(b,l//16,16);n=pv.sum(-1).clamp_min(1)
        token=(h.reshape(b,l//16,16,96)*pv[...,None]).sum(2)/n[...,None]
        pos=torch.linspace(-1,1,l,device=x.device).expand(b,-1)
        loc=torch.stack((relative,pos),-1).reshape(b,l//16,16,2)
        loc=(loc*pv[...,None]).sum(2)/n[...,None]
        tv=pv.any(-1);token=(token+c.token_pos(loc)).masked_fill(~tv[...,None],0)
        for j,block in enumerate(c.global_blocks):token=modulate(block(token,tv),modulation[j+1],tv)
        global_h=F.interpolate(token.transpose(1,2),size=l,mode='linear',align_corners=False).transpose(1,2)
        h=F.gelu(c.fuse(torch.cat((h,global_h),-1))).masked_fill(~valid[...,None],0)
        h=modulate(h,modulation[5],valid)
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
            relative=torch.tanh((d['depth'][ix]-d['depth'][anchors,None])/10)
            context=d['well_bank'][d['row_well'][anchors]]
            with torch.autocast('cuda',dtype=torch.bfloat16):go,raw=model(d['x'][ix],relative,valid,well_inputs=context)
            rows=d['raw_to_valid'][ix[:,lo:hi].cpu().numpy()];use=valid[:,lo:hi].cpu().numpy()&(rows>=0);dest=self.local[rows[use]]
            assert (dest>=0).all() and (coverage[dest]==0).all() and len(np.unique(dest))==len(dest)
            coverage[dest]+=1;take=torch.as_tensor(use,device='cuda')
            g=go[:,lo:hi][take][:,0].float().sigmoid();r=raw[:,lo:hi][take][:,0].float()
            cp=probability(c,r[:,0].softmax(-1),r[:,1],F.softplus(r[:,2])+.025,self.edges)
            joint=torch.cat((g[:,None],(1-g[:,None])*cp),-1)
            result[dest]=self.actions[(joint@self.utilities).argmax(-1).cpu().numpy()]
        assert (coverage==1).all() and np.isfinite(result).all()
        combined=self.reference.copy();combined[:,c]=result;scores=base.metric(d['truth'][d['validation']],combined)
        return result,float(scores['SW']),scores
