"""Matched head-shared versus head-private DropKey from the retained v77 SW function."""
import hashlib, types
import numpy as np
import torch
from torch.nn import functional as F
from sw_dropout_extend_v77 import ROOT, Network as Parent, Evaluator, load_data

STUDY=ROOT/'results/deep_learning/sw_dropout_heads_v78_20260930'
PREVIOUS=ROOT/'results/deep_learning/sw_dropout_extend_v77_20260930'
SOURCE=ROOT/'outputs/pure_dl_sw_dropout_extend_v77_SW_p40_fold5/model_best.pt'
MODES={m:dict(target='SW',c=2,length=1024,rate=.4) for m in ('shared','per_head')}

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

def attention_forward(self,x,valid):
    b,t,c=x.shape
    q,k,v=self.qkv(self.norm(x)).reshape(b,t,3,self.heads,c//self.heads).permute(2,0,3,1,4).unbind(0)
    keys=valid[:,None,:]
    if self.training:
        # Both arms consume the same BHt draw tensor; windows have a separate RNG.
        random=torch.rand((b,self.heads,t),device=x.device)
        if self.dropout_sharing=='shared':random=random[:,:1,:].expand(-1,self.heads,-1)
        proposed=valid[:,None,:]&(random>=.4)
        keys=torch.where(proposed.any(-1,keepdim=True),proposed,valid[:,None,:])
    z=F.scaled_dot_product_attention(q,k,v,attn_mask=keys[:,:,None,:],dropout_p=0.)
    x=(x+self.project(z.transpose(1,2).reshape(b,t,c))).masked_fill(~valid[...,None],0)
    return (x+self.ff(self.ffnorm(x))).masked_fill(~valid[...,None],0)

class Network(Parent):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('p40',metadata)
        saved=torch.load(SOURCE,map_location='cpu',weights_only=False)
        self.load_state_dict(saved['state_dict'],strict=True);self.study_mode=mode
        for block in self.body.global_blocks:
            block.dropout_sharing=mode;block.forward=types.MethodType(attention_forward,block)
        self.parent_info=dict(checkpoint=str(SOURCE.relative_to(ROOT)),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            best_epoch=saved['best_epoch'],target_score=saved['target_score'],initialization='complete v77 p40 SW function, no prior reset',
            optimizer_reset=True,all_network_trainable=True,added_parameters=0,training_key_drop_probability=.4,
            key_mask_sharing=mode,same_BHt_rng_draws_both_modes=True,all_dropped_fallback='per head original valid keys',
            evaluation_key_drop_probability=0.,new_learning_rate=3e-5)

    def initialize_priors(self,data):
        # Preserve the full retained predictor, including its trained output density.
        return self.parent_info
