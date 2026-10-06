"""Train-only inter-well hidden-moment mixing on the retained private PERM model.

Adapted from the MixStyle idea (Zhou et al., ICLR2021):
https://github.com/KaiyangZhou/mixstyle-release
Here moments are masked over well depth and partners must be different train wells.
"""
import hashlib
import torch
from torch.nn import functional as F
from perm_transfer_v50 import ROOT,Network as Parent,Evaluator,load_data,reference_prediction
from well_conditioned_v46 import modulate

STUDY=ROOT/'results/deep_learning/perm_moment_mix_v51_20260929'
PREVIOUS=ROOT/'results/deep_learning/perm_transfer_v50_20260929'
PARENT=ROOT/'outputs/pure_dl_perm_transfer_v50_PERM_sw_transfer_fold5/model_best.pt'
PARENT_PRED=PARENT.with_name('channel_pred.npy')
MODES={m:dict(target='PERM',c=1,length=1024) for m in ('plain','interwell')}

class Network(Parent):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('sw_transfer',metadata)
        saved=torch.load(PARENT,map_location='cpu',weights_only=False);self.load_state_dict(saved['state_dict'],strict=True)
        self.study_mode=mode;self.register_buffer('mix_counts',torch.zeros(3,dtype=torch.long))
        self.initialization_info=dict(checkpoint=str(PARENT.relative_to(ROOT)),sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),
            best_epoch=saved['best_epoch'],target_score=saved['target_score'],optimizer_reset=True,
            initialization='exact full v50 trained private PERM model; no output reset',
            augmentation='train-only masked hidden mean/std mixing, interwell partners; no label mixing')

    def initialize_priors(self,data):return self.initialization_info

    def moment_mix(self,h,valid,well_ids):
        if not self.training:return h,None
        if well_ids is None:raise ValueError('Training-well grouping required only for augmentation pairing')
        b=h.shape[0];device=h.device
        trigger=bool(torch.rand((),device=device)<.5)
        alpha=torch.tensor(.1,device=device)
        lam=torch.distributions.Beta(alpha,alpha).sample((b,1,1))
        allowed=well_ids[:,None]!=well_ids[None,:]
        eligible=allowed.any(-1)
        choices=torch.rand((b,b),device=device).masked_fill(~allowed,-1)
        partner=choices.argmax(-1)
        self.mix_counts[0]+=1
        apply=self.study_mode=='interwell' and trigger
        if apply:
            v=valid[...,None].float();n=v.sum(1,keepdim=True).clamp_min(1)
            mean=(h.float()*v).sum(1,keepdim=True)/n
            std=(((h.float()-mean).square()*v).sum(1,keepdim=True)/n+1e-6).sqrt()
            mean=mean.detach();std=std.detach()
            mixed_mean=lam*mean+(1-lam)*mean[partner]
            mixed_std=lam*std+(1-lam)*std[partner]
            mixed=((h.float()-mean)/std*mixed_std+mixed_mean).to(h.dtype)
            h=torch.where(eligible[:,None,None],mixed,h).masked_fill(~valid[...,None],0)
            self.mix_counts[1]+=eligible.sum()
            self.mix_counts[2]+=(eligible & (well_ids[partner]!=well_ids)).sum()
        return h,dict(applied=apply,eligible_windows=int(eligible.sum()),
            cumulative_batches=int(self.mix_counts[0]),cumulative_mixed_windows=int(self.mix_counts[1]),
            cumulative_different_well_pairs=int(self.mix_counts[2]),group_ids_not_model_features=True)

    def forward(self,x,relative,valid,absolute_depth=None,return_aux=False,generator=None,well_inputs=None,well_ids=None):
        if well_inputs is None:raise ValueError('Input-only current-well bank required')
        b,l,_=x.shape;c=self.body;modulation=self.condition(well_inputs)
        h=F.gelu(c.stem(torch.cat((x,relative[...,None]),-1))).masked_fill(~valid[...,None],0)
        h=modulate(h,modulation[0],valid);h,mixinfo=self.moment_mix(h,valid,well_ids)
        for block in c.local:h=block(h,valid)
        pv=valid.reshape(b,l//16,16);n=pv.sum(-1).clamp_min(1)
        token=(h.reshape(b,l//16,16,96)*pv[...,None]).sum(2)/n[...,None]
        pos=torch.linspace(-1,1,l,device=x.device).expand(b,-1)
        loc=torch.stack((relative,pos),-1).reshape(b,l//16,16,2);loc=(loc*pv[...,None]).sum(2)/n[...,None]
        tv=pv.any(-1);token=(token+c.token_pos(loc)).masked_fill(~tv[...,None],0)
        for j,block in enumerate(c.global_blocks):token=modulate(block(token,tv),modulation[j+1],tv)
        global_h=F.interpolate(token.transpose(1,2),size=l,mode='linear',align_corners=False).transpose(1,2)
        h=F.gelu(c.fuse(torch.cat((h,global_h),-1))).masked_fill(~valid[...,None],0)
        h=modulate(h,modulation[5],valid)
        for block in c.detail:h=block(h,valid)
        y=self.output(self.norm(h)).float();output=(y[...,:1],y[...,1:].reshape(b,l,1,3,8))
        return (output,y.new_zeros(()),mixinfo) if return_aux else output
