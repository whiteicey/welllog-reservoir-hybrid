"""Two predefined DropKey rates around the retained historical 10% control."""
import types
import numpy as np
import torch
from torch.nn import functional as F
from sw_key_dropout_v74 import ROOT,Network as Parent,Evaluator,load_data

STUDY=ROOT/'results/deep_learning/sw_dropout_rate_v76_20260930'
PREVIOUS=ROOT/'results/deep_learning/sw_dropout_scope_v75_20260929'
CONTROL=ROOT/'outputs/pure_dl_sw_key_dropout_v74_SW_dropkey_fold5'
ZERO=ROOT/'outputs/pure_dl_sw_key_dropout_v74_SW_control_fold5'
MODES={'p05':dict(target='SW',c=2,length=1024,rate=.05),'p20':dict(target='SW',c=2,length=1024,rate=.2)}

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

def attention_forward(self,x,valid):
    b,t,c=x.shape
    q,k,v=self.qkv(self.norm(x)).reshape(b,t,3,self.heads,c//self.heads).permute(2,0,3,1,4).unbind(0)
    keys=valid
    if self.training:
        proposed=valid&(torch.rand(valid.shape,device=x.device)>=self.drop_probability)
        proposed=torch.where(proposed.any(-1,keepdim=True),proposed,valid)
        if self.use_key_dropout:keys=proposed
    z=F.scaled_dot_product_attention(q,k,v,attn_mask=keys[:,None,None,:],dropout_p=0.)
    x=(x+self.project(z.transpose(1,2).reshape(b,t,c))).masked_fill(~valid[...,None],0)
    return (x+self.ff(self.ffnorm(x))).masked_fill(~valid[...,None],0)

class Network(Parent):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('dropkey',metadata)
        self.study_mode=mode;rate=MODES[mode]['rate']
        for block in self.body.global_blocks:
            block.drop_probability=rate;block.forward=types.MethodType(attention_forward,block)
        self.parent_info.update(training_key_drop_probability=rate,historical_control=str(CONTROL.relative_to(ROOT)),
            historical_control_retrained=False,one_forward_per_batch=True,only_hyperparameter_changed='training key drop probability')
