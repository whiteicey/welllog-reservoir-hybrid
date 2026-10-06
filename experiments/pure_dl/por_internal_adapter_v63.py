"""POR internal zero-output adapters: matched pointwise versus depth-local access."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from por_reparam_v31 import Network as Parent,ReparameterizedBranch
from champion_all_heads_v26 import load_data,Evaluator
from blur_ungated_sequence import UngatedBlock

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/por_internal_adapter_v63_20260929'
PREVIOUS=ROOT/'results/deep_learning/sw_density_private_v62_20260929'
MODES=('point','local')

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

class DifferentiableAnchor(ReparameterizedBranch):
    def forward(self,x,valid):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            u=self.output(F.gelu(self.input(x))).squeeze(1).float()
            if self.mode=='anchored':
                # Frozen anchor weights, but feature derivatives must remain live.
                u0=self.anchor_output(F.gelu(self.anchor_input(x))).squeeze(1).float()
                d0=self.bound*torch.tanh(u0/self.bound)
                d=(d0+(u-u0)).clamp(-self.bound,self.bound)
            else:d=self.bound*torch.tanh(u/self.bound)
        return d.masked_fill(~valid,0)

class InternalAdapter(nn.Module):
    def __init__(self,old,mode):
        super().__init__();self.old=old;self.mode=mode;c=old.expand.in_channels
        self.down=nn.Conv1d(c,8,1);self.context=nn.Conv1d(8,8,3,padding=1,groups=8)
        self.up=nn.Conv1d(8,c,1)
        nn.init.zeros_(self.context.weight);nn.init.zeros_(self.context.bias)
        with torch.no_grad():self.context.weight[:,:,1]=1
        nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
    def forward(self,x):
        original=self.old(x)
        h=F.gelu(self.down(self.old.norm(x)))
        if self.mode=='local':z=self.context(h)
        else:z=F.conv1d(h,self.context.weight.sum(-1,keepdim=True),self.context.bias,groups=8)
        correction=.1*torch.tanh(self.up(z).float())
        return original+correction.to(original.dtype)

class Network(nn.Module):
    def __init__(self,target,master,mode):
        super().__init__()
        if target!='POR' or mode not in MODES:raise ValueError((target,mode))
        self.target=target;self.c=0;self.mode=mode;self.parent_mode='anchored'
        recipe=json.loads((PREVIOUS/'CURRENT_RECIPE.json').read_text(encoding='utf-8'))
        self.choice=next(x for x in recipe['choices'] if x['target']=='POR')
        assert self.choice['model_module']=='por_reparam_v31' and self.choice['mode']==self.parent_mode
        self.current=Parent('POR',master,modes=(self.parent_mode,))
        saved=torch.load(ROOT/self.choice['checkpoint'],map_location='cpu',weights_only=False)
        self.current.heads[self.parent_mode].load_state_dict(saved['delta_state'],strict=True)
        self.current.requires_grad_(False);self.current.close();self.current.handles=[]
        bundle=self.current.heads[self.parent_mode]
        for branch in [bundle.route28.unit,*bundle.route29.branches]:branch.__class__=DifferentiableAnchor
        replaced=[]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261163)
            for e,member in enumerate(self.current.source.members):
                for prefix,parent in list(member.named_modules()):
                    for name,child in list(parent.named_children()):
                        if isinstance(child,UngatedBlock):
                            replaced.append(dict(expert=e,path=prefix+'.'+name,width=child.expand.in_channels))
                            setattr(parent,name,InternalAdapter(child,mode))
        assert len(replaced)==18
        self.trainable_names={n for n,p in self.named_parameters() if p.requires_grad}
        assert self.trainable_names and all(any('.'+k+'.' in n for k in ('down','context','up')) for n in self.trainable_names)
        self.manifest=dict(adapters=replaced,rank=8,correction_bound=.1,mode=mode,initialization='up weights/bias zero; depthwise center identity',
            comparison='same tensors: point sums all three kernel taps at center; local reads adjacent depths',
            anchor_gradient='frozen original anchor weights now preserve feature gradients; numeric function unchanged',
            trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad))
        self.captured={};self.handles=[]
        for e,member in enumerate(self.current.source.members):
            def capture(_module,args,_output,e=e):self.captured['h'+str(e)]=args[0]
            self.handles.append(member.main_head.register_forward_hook(capture))
        self.handles.append(self.current.source.router[2].register_forward_hook(lambda _m,_a,v:self.captured.update(logits=v)))
        self.train(False)
    def train(self,mode=True):
        super().train(False);self.training=mode;return self
    def forward(self,x,relative,valid,absolute_depth=None,factors=None):
        if factors is not None and tuple(factors)!=(self.mode,):raise ValueError(factors)
        self.captured.clear()
        with torch.autocast('cuda',dtype=torch.bfloat16):self.current.source(x,relative,valid,absolute_depth=absolute_depth)
        hidden=[self.captured['h'+str(e)] for e in range(2)]
        logits=self.captured['logits'].reshape(*x.shape[:2],3,2)[:,:,:1]
        self.captured.clear()
        return {self.mode:self.current.heads[self.parent_mode](hidden,None,logits,x[...,206:],valid)}
    def trainable_state(self):return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if p.requires_grad}
    def load_trainable_state(self,state):
        assert set(state)==self.trainable_names
        params=dict(self.named_parameters())
        with torch.no_grad():
            for n,v in state.items():params[n].copy_(v.to(params[n].device))
    def frozen_state(self):return {n:v for n,v in self.state_dict().items() if n not in self.trainable_names}
    def close(self):
        for h in self.handles:h.remove()
        self.handles=[];self.captured.clear()
