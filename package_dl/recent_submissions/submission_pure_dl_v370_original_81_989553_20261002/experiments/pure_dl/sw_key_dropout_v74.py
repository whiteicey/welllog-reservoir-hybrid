"""Matched training-time key dropout; identical complete inference architecture."""
import hashlib,types
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sw_interaction_v56 import ROOT,Network as Parent,Evaluator,load_data
from divergent_models_v45 import Network as PriorNetwork

STUDY=ROOT/'results/deep_learning/sw_key_dropout_v74_20260929'
PREVIOUS=ROOT/'results/deep_learning/sw_patch_moments_v73_20260929'
SOURCE=ROOT/'outputs/pure_dl_sw_interaction_v56_SW_additive_fold5/model_best.pt'
MODES={m:dict(target='SW',c=2,length=1024) for m in ('control','dropkey')}

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

def attention_forward(self,x,valid):
    b,t,c=x.shape
    q,k,v=self.qkv(self.norm(x)).reshape(b,t,3,self.heads,c//self.heads).permute(2,0,3,1,4).unbind(0)
    keys=valid
    if self.training:
        # Draw in both arms, so this mechanism does not change window RNG or RNG draw count.
        proposed=valid&(torch.rand(valid.shape,device=x.device)>=.1)
        proposed=torch.where(proposed.any(-1,keepdim=True),proposed,valid)
        if self.use_key_dropout:keys=proposed
    z=F.scaled_dot_product_attention(q,k,v,attn_mask=keys[:,None,None,:],dropout_p=0.)
    x=(x+self.project(z.transpose(1,2).reshape(b,t,c))).masked_fill(~valid[...,None],0)
    return (x+self.ff(self.ffnorm(x))).masked_fill(~valid[...,None],0)

class Network(Parent):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('additive',metadata)
        saved=torch.load(SOURCE,map_location='cpu',weights_only=False);self.load_state_dict(saved['state_dict'],strict=True)
        self.study_mode=mode
        for block in self.body.global_blocks:
            block.use_key_dropout=mode=='dropkey';block.forward=types.MethodType(attention_forward,block)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.parent_info=dict(checkpoint=str(SOURCE.relative_to(ROOT)),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            best_epoch=saved['best_epoch'],target_score=saved['target_score'],initialization='v56 additive SW full backbone/FiLM; output reset to train64 prior',
            optimizer_reset=True,all_network_trainable=True,added_parameters=0,training_key_drop_probability=.1 if mode=='dropkey' else 0.,
            evaluation_key_drop_probability=0.,key_mask_shared_across_heads_and_queries=True,all_dropped_fallback='original valid keys',same_rng_draws_both_modes=True)

    def initialize_priors(self,data):
        prior=PriorNetwork.initialize_priors(self,data)
        prior['scope']='only original64 training SW labels/quality fit reset output prior; v56 pretrained fold-local backbone retained'
        return prior|self.parent_info
