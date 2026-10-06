"""v35 reference density with a smooth, initially 1% participation gate.

Same memory, candidate pools, learned attention and density likelihood. Initial
candidate predictions differ from the parent; disabled retrieval alone is exact.
The old hard-clipped study and its trained sources remain immutable.
"""
from pathlib import Path
import math
import torch
from torch import nn
from torch.nn import functional as F
from retrieval_density_v35 import (Network as OldNetwork,RetrievalHead as OldHead,
    Evaluator,load_data,reference_prediction,channel_objective,MODES,TARGETS,ROOT,PREVIOUS)
from multiscale_sequence import window_indices

STUDY=ROOT/'results/deep_learning/retrieval_smooth_gate_v36_20260929'
MEMORY_STUDY=ROOT/'results/deep_learning/retrieval_density_v35_20260929'
LEARNING_RATE=3e-4
INITIAL_ALPHA=.01
MAX_ALPHA=.25

class RetrievalHead(OldHead):
    def __init__(self,mode):
        super().__init__(mode)
        self.force_off=False
        p=INITIAL_ALPHA/MAX_ALPHA
        self.initial_gate_logit=math.log(p/(1-p))

    def forward(self,hidden,parent,context,side,valid):
        ids,dist=context
        if side.shape[-1]!=63:raise ValueError('Expected existing amplitude63')
        x=torch.cat((hidden[0].transpose(1,2).float(),hidden[1].transpose(1,2).float(),side.float()),-1)
        x=x.masked_fill(~valid[...,None],0)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            query=F.gelu(self.query(x));keys=self.key(self.reference_features)
        k=keys[ids].float()
        logits=(query.float()[...,None,:]*k).sum(-1)/math.sqrt(32)-dist.float()
        message=(logits.softmax(-1)[...,None]*k).sum(-2)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            params=self.output(torch.cat((query.float(),message),-1)).float()
        alpha=MAX_ALPHA*torch.sigmoid(params[...,0]+self.initial_gate_logit)
        alpha=alpha.masked_fill(~valid,0)
        if self.force_off:alpha=alpha*0
        sd=.025+F.softplus(self.initial_scale_raw+.5*torch.tanh(params[...,1]/.5))
        return dict(parent=parent,logits=logits,mu=self.reference_mu[ids],sd=sd,alpha=alpha)

class Network(OldNetwork):
    def __init__(self,target,master,modes=MODES):
        super().__init__(target,master,modes=())
        self.heads=nn.ModuleDict({m:RetrievalHead(m) for m in modes})
        self.metadata.update(initialization='alpha=0.01, distinct from parent; explicit disabled-parent verification',
            gate='0.25*sigmoid(u+logit(0.04))',initial_alpha=INITIAL_ALPHA,
            rationale='Avoid inactive negative half-axis of v35 clamp; no straight-through gradient',
            historical_gate_diagnostic=str((MEMORY_STUDY/'gate_diagnostic.json').relative_to(ROOT)))
        self.train(False)

    def trainable_groups(self,mode):
        return [dict(params=list(self.heads[mode].parameters()),lr=LEARNING_RATE,name='reference_density')]

@torch.no_grad()
def factor_scope(model,data,mode,n_anchors=64):
    anchors=data['train_raw'][torch.linspace(0,len(data['train_raw'])-1,n_anchors,device='cuda').long()]
    ix,valid=window_indices(anchors,data['lower'],data['upper'])
    relative=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
    h,parent,ctx,_=model.extract(data['x'][ix],relative,valid,data['absolute'][ix],rows=ix)
    out=model.heads[mode](h,parent,ctx,data['x'][ix,...,206:],valid)
    mask=valid&data['labeled'][ix];mask[:,:64]=False;mask[:,192:]=False
    assert torch.equal(out['parent'][0],parent[0]) and torch.equal(out['parent'][1],parent[1])
    alpha=out['alpha'][mask];sd=out['sd'][mask];weights=out['logits'][mask].softmax(-1)
    entropy=-(weights*weights.clamp_min(1e-30).log()).sum(-1)
    assert bool(torch.isfinite(alpha).all()) and float(alpha.min())>=0 and float(alpha.max())<=MAX_ALPHA+1e-7
    return dict(mode=mode,points=int(mask.sum()),parent_factors_exact=True,alpha_mean=float(alpha.mean()),
        alpha_min=float(alpha.min()),alpha_max=float(alpha.max()),active_fraction=float((alpha>0).float().mean()),
        bandwidth_mean=float(sd.mean()),attention_effective_neighbors_mean=float(entropy.exp().mean()),
        scope='64 fixed training anchors, valid middle128; no validation labels')
