"""Limited, matched posterior-conditioned corrections on the retained v26 recipe.

Only the already useful factor per target is considered. Self/pair alternatives
have identical tensors and starts; the second posterior descriptor repeats the
current expert or describes the other expert. All parent predictors are frozen.
"""
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from champion_all_heads_v26 import (
    build_source, FactorHead as RetainedFactorHead, Evaluator, load_data, TARGETS,
)
from paired_detached_router_sequence import routed_merge

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/champion_posterior_v27_20260929'
PREVIOUS=ROOT/'results/deep_learning/all_heads_v26_20260928'
MODES=('self','pair')
SELECTED={'POR':'location','PERM':'gate','SW':'weight'}


def reference_prediction():
    # The exact retained prediction path is resolved by setup from v26 evidence.
    import json
    path=json.loads((STUDY/'reference.json').read_text(encoding='utf-8'))['prediction']
    return np.load(ROOT/path)


@torch.no_grad()
def describe(output):
    """B,L,4: default probability, mixture entropy, mean, log1p std.

    Locations/scales are in each target's original standardized transformed
    space. Both the t3 POR and Gaussian others use actual component std here.
    """
    g,r=output
    raw=r[:,:,0].float()
    # Expected raw layout after the channel dimension is removed: B,L,3,8.
    if raw.ndim!=4 or raw.shape[-2:]!=(3,8):
        raise ValueError(f'Unexpected density tensor {tuple(raw.shape)}')
    p=raw[...,0,:].softmax(-1)
    mu=raw[...,1,:]
    std=F.softplus(raw[...,2,:])+.025
    mean=(p*mu).sum(-1)
    variance=(p*(std.square()+(mu-mean[...,None]).square())).sum(-1)
    entropy=-(p*p.clamp_min(1e-12).log()).sum(-1)/math.log(8.)
    return torch.stack((g[...,0].float().sigmoid(),entropy,
        mean.clamp(-8,8),torch.log1p(variance.clamp_min(0).sqrt()).clamp(0,8)),-1).detach()


class ExpertCorrection(nn.Module):
    def __init__(self,outputs):
        super().__init__()
        self.input=nn.Conv1d(199,32,1)
        self.output=nn.Conv1d(32,outputs,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,h,side,own,other,valid):
        context=torch.cat((side,own,other),-1).transpose(1,2).to(h.dtype)
        z=torch.cat((h,context),1).masked_fill(~valid[:,None],0)
        out=self.output(F.gelu(self.input(z))).transpose(1,2).float()
        return out.masked_fill(~valid[...,None],0)


class PosteriorHead(nn.Module):
    def __init__(self,factor,mode):
        super().__init__()
        if mode not in MODES or factor not in SELECTED.values():raise ValueError((factor,mode))
        self.factor=factor;self.mode=mode
        self.experts=nn.ModuleList()
        for e in range(2):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20261103+e)
                self.experts.append(ExpertCorrection(1 if factor=='gate' else 8))

    def corrected_members(self,hidden,outputs,side,valid):
        summaries=[describe(o) for o in outputs]
        result=[]
        for e,(head,h,(g,r)) in enumerate(zip(self.experts,hidden,outputs)):
            other=summaries[1-e] if self.mode=='pair' else summaries[e]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                proposal=head(h,side,summaries[e],other,valid)
            if self.factor=='gate':
                result.append((g.float()+.25*torch.tanh(proposal/.25),r));continue
            field=1 if self.factor=='location' else 0
            bound=.25*(F.softplus(r[:,:,0,2,:].detach().float())+.025) if field==1 else .25
            delta=bound*torch.tanh(proposal/bound)
            slices=[r[...,k,:] for k in range(3)]
            slices[field]=slices[field]+delta.unsqueeze(2)
            result.append((g,torch.stack(slices,dim=-2)))
        return result

    def forward(self,hidden,outputs,logits,side,valid):
        return routed_merge(self.corrected_members(hidden,outputs,side,valid),logits)


class Network(nn.Module):
    def __init__(self,target,master,modes=MODES):
        super().__init__()
        if target not in SELECTED:raise ValueError(target)
        self.target=target;self.c=TARGETS.index(target);self.factor=SELECTED[target]
        self.source,self.source_path=build_source(target,master)
        self.retained=None;self.retained_path=None
        if target!='POR':
            self.retained=RetainedFactorHead(self.factor)
            path=ROOT/'outputs'/f'pure_dl_all_heads_v26_{target}_fold5'/self.factor/'model_best.pt'
            saved=torch.load(path,map_location='cpu',weights_only=False)
            self.retained.load_state_dict(saved['delta_state'],strict=True)
            self.retained.requires_grad_(False);self.retained.eval()
            self.retained_path=str(path.relative_to(ROOT))
        self.source.requires_grad_(False);self.source.eval()
        self.heads=nn.ModuleDict({m:PosteriorHead(self.factor,m) for m in modes})
        self.captured={};self.handles=[]
        for e,member in enumerate(self.source.members):
            def capture(_module,args,output,e=e):
                self.captured['h'+str(e)]=args[0].detach()
                self.captured['o'+str(e)]=tuple(v.detach() for v in output)
            self.handles.append(member.main_head.register_forward_hook(capture))
        self.handles.append(self.source.router[2].register_forward_hook(
            lambda _m,_a,v:self.captured.update(logits=v.detach())))

    def train(self,mode=True):
        super().train(False);self.training=mode;self.heads.train(mode);return self

    @torch.no_grad()
    def extract(self,x,relative,valid,absolute):
        self.captured.clear();self.source.eval()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            self.source(x if self.c==0 else x[...,:206].contiguous(),relative,valid,absolute_depth=absolute)
        h=[self.captured['h'+str(e)] for e in range(2)]
        o=[self.captured['o'+str(e)] for e in range(2)]
        logits=self.captured['logits'].reshape(*x.shape[:2],3,2)[:,:,self.c:self.c+1]
        self.captured.clear()
        if self.retained is not None:
            o=self.retained.corrected_members(h,o,x[...,206:],valid)
        return h,o,logits,None

    def forward(self,x,relative,valid,absolute_depth=None,factors=None):
        chosen=tuple(self.heads) if factors is None else tuple(factors)
        h,o,logits,_=self.extract(x,relative,valid,absolute_depth)
        return {m:self.heads[m](h,o,logits,x[...,206:],valid) for m in chosen}

    def frozen_state(self):
        return {k:v for k,v in self.state_dict().items() if not k.startswith('heads.')}

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[];self.captured.clear()
