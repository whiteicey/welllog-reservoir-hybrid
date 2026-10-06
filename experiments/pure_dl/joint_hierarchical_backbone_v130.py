"""New shared multi-target hierarchical convolution/attention backbone, random weights."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from compact_five_stage_v100 import ROOT,SOURCE,load_data as original_data
from por_convgru_backbone_v127 import ConvBlock
from train_stage_fullcv_v97 import Evaluator as TargetEvaluator
from train_scaling_v21 import channel_objective as base_loss
import train_invariant as base
STUDY=ROOT/'results/deep_learning/joint_hierarchical_backbone_v130_20261001'
OUT=ROOT/'outputs/pure_dl_joint_hierarchical_backbone_v130_fold5'
STAGES={'joint':('TOTAL',512,False,0.,6e-4)}
class Network(nn.Module):
    def __init__(self):
        super().__init__();self.stem=nn.Linear(207,96)
        self.local=nn.ModuleList([ConvBlock(d) for d in (1,2,4)])
        self.context=nn.Sequential(nn.Linear(618,192),nn.GELU(),nn.Linear(192,192))
        nn.init.zeros_(self.context[-1].weight);nn.init.zeros_(self.context[-1].bias)
        self.position=nn.Linear(2,96)
        self.global_blocks=nn.ModuleList([nn.TransformerEncoderLayer(96,4,384,dropout=.1,activation='gelu',batch_first=True,norm_first=True) for _ in range(4)])
        self.fuse=nn.Linear(192,96);self.detail=nn.ModuleList([ConvBlock(1),ConvBlock(2)])
        self.norm=nn.LayerNorm(96);self.output=nn.Linear(96,75)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def groups(self):return [dict(params=list(self.parameters()),lr=6e-4,name='new_joint_backbone_three_heads')]
    @torch.no_grad()
    def initialize_priors(self,data):
        ids=data['train_raw'];priors=[]
        for c in range(3):
            d=data['default'][ids,c];y=data['labels'][ids,c][~d].float();rate=d.float().mean().clamp(1e-4,1-1e-4)
            means=torch.quantile(y,torch.linspace(.06,.94,8,device=y.device));offset=3+24*c
            self.output.bias[c]=torch.logit(rate);self.output.bias[offset+8:offset+16].copy_(means)
            self.output.bias[offset+16:offset+24].fill_(math.log(math.expm1(.5-.025)))
            priors.append(dict(target=base.TS[c],default_rate=float(rate),initial_locations=means.cpu().tolist()))
        return dict(random_all_network_weights=True,no_old_network_weights_loaded=True,priors_train_only=priors,
            inputs=206,relative_depth=True,well_context_input_only=True,shared_three_target_training=True,
            POR_family='Student3',PERM_SW_family='Gaussian',conditional_components=8,
            standalone_recipe_phases_including_original_source_Q=2,possible_private_POR_fixedPERM_SW_recipe_phases=5)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        assert x.shape[-1]==206 and well_inputs is not None
        h=F.gelu(self.stem(torch.cat((x,relative[...,None]),-1))).masked_fill(~valid[...,None],0)
        bank=well_inputs.float();stats=torch.cat((bank.mean(1),bank.var(1,unbiased=False).add(1e-6).sqrt(),bank.amax(1)),1)
        gamma,beta=(.5*self.context(stats).float().tanh()).chunk(2,-1)
        h=(h*(1+gamma[:,None])+beta[:,None]).masked_fill(~valid[...,None],0)
        for block in self.local:h=block(h,valid)
        b,l,w=h.shape;pv=valid.reshape(b,l//8,8);counts=pv.sum(-1).clamp_min(1)
        tokens=h.reshape(b,l//8,8,w).sum(2)/counts[...,None];tv=pv.any(-1)
        pos=torch.linspace(-1,1,l,device=x.device)[None].expand(b,-1)
        loc=torch.stack((relative,pos),-1).reshape(b,l//8,8,2)
        loc=(loc*pv[...,None]).sum(2)/counts[...,None]
        tokens=(tokens+self.position(loc)).masked_fill(~tv[...,None],0)
        for block in self.global_blocks:tokens=block(tokens,src_key_padding_mask=~tv).masked_fill(~tv[...,None],0)
        global_h=F.interpolate(tokens.transpose(1,2),size=l,mode='linear',align_corners=False).transpose(1,2)
        h=F.gelu(self.fuse(torch.cat((h,global_h),-1))).masked_fill(~valid[...,None],0)
        for block in self.detail:h=block(h,valid)
        y=self.output(self.norm(h)).float()
        return y[...,:3],y[...,3:].reshape(b,l,3,3,8)
def load_data(stage):return original_data('sw_joint')
def make_model(stage,master,metadata):return Network()
def channel_objective(output,truth,default,mask,quality,target_index,mode):
    go,raw=output;parts=[base_loss((go[...,c:c+1],raw[:,:,c:c+1]),truth,default,mask,quality,c) for c in range(3)]
    return tuple(sum(p[i] for p in parts) for i in range(3))
class View(nn.Module):
    def __init__(self,model,c):super().__init__();self.model=model;self.c=c
    def forward(self,*args,**kwargs):
        gate,raw=self.model(*args,**kwargs);return gate[...,self.c:self.c+1],raw[:,:,self.c:self.c+1]
class Evaluator:
    def __init__(self,data,master,c,length,mode):
        self.data=data;self.evaluators=[TargetEvaluator(data,master,i,length) for i in range(3)]
    @torch.no_grad()
    def __call__(self,model):
        predictions=np.column_stack([e(View(model,i))[0] for i,e in enumerate(self.evaluators)])
        self.last_scores=base.metric(self.data['truth'][self.data['validation']],predictions)
        return predictions,float(self.last_scores['TOTAL'])
