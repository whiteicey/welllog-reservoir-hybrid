"""PERM complete private warm start; matched likelihood and official action utility.

Original packed other-target rows stay buffers. All PERM backbone/readout and
retained input-only corrections learn, with no no_grad/detach capture path.
"""
from pathlib import Path
import hashlib,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from champion_split_route_v29 import Network as Parent,shift
from champion_all_heads_v26 import Evaluator as ParentEvaluator,load_data,TARGETS
from train_scaling_v21 import channel_objective
from train_dual8222_heads80 import SOURCE

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/perm_full_utility_v87_20260930'
PREVIOUS=ROOT/'results/deep_learning/por_joint_utility_v86_20260930'
MODES={'likelihood':0.,'joint':1.}

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

class Network(Parent):
    def __init__(self,mode,master):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('PERM',master,modes=('coupled',))
        path=ROOT/'outputs/pure_dl_split_route_v29_PERM_fold5/coupled/model_best.pt'
        saved=torch.load(path,map_location='cpu',weights_only=False)
        self.heads['coupled'].load_state_dict(saved['delta_state'],strict=True)
        head=self.heads['coupled'];self.heads=nn.ModuleDict({mode:head});self.study_mode=mode
        # Replace the immutable old capture hooks, not their historical source.
        for handle in self.handles:handle.remove()
        self.handles=[];self.captured.clear()
        for e,member in enumerate(self.source.members):
            def capture(_module,args,output,e=e):
                self.captured['h'+str(e)]=args[0]
                self.captured['o'+str(e)]=output
            self.handles.append(member.main_head.register_forward_hook(capture))
        self.handles.append(self.source.router[2].register_forward_hook(
            lambda _m,_a,v:self.captured.update(logits=v)))
        self.requires_grad_(True)
        self.metadata=dict(source_checkpoint=str(path.relative_to(ROOT)),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_epoch=saved['best_epoch'],source_score=saved['target_score'],added_parameters=0,
            backbone_and_own_target_trainable=True,old_capture_detach_removed=True,old_no_grad_removed=True,
            other_target_packed_rows='immutable buffers, not parameters',optimizer_reset=True,
            initialization='complete retained PERM v8+v26gate+v28joint+v29coupled function; no prior reset',
            body_lr=1e-5,head_lr=3e-5,eta_min=1e-5,
            schedule='body starts at eta_min and stays1e-5; target readout/corrections cosine3e-5 to1e-5',
            utility_coefficient=MODES[mode],temperature=.05,utility_chunk=512,gradient_checkpointing=True)
        self.train(False)

    def train(self,mode=True):
        # Bypass Parent's frozen-backbone train override in this NEW subclass.
        nn.Module.train(self,mode);return self

    def extract(self,x,relative,valid,absolute):
        self.captured.clear()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            self.source(x[...,:206].contiguous(),relative,valid,absolute_depth=absolute)
            h=[self.captured['h'+str(e)] for e in range(2)]
            o=[self.captured['o'+str(e)] for e in range(2)]
            logits=self.captured['logits'].reshape(*x.shape[:2],3,2)[:,:,1:2]
            self.captured.clear()
            o=self.retained.corrected_members(h,o,x[...,206:],valid)
            delta=self.retained_route.correction(h,x[...,206:],valid)
            density=shift(logits,delta)
            atom=density if self.parent_mode=='joint' else logits
        return h,o,(atom,density),None

    def forward(self,x,relative,valid,absolute_depth=None,factors=None):
        h,o,context,_=self.extract(x,relative,valid,absolute_depth)
        output=self.heads[self.study_mode](h,o,context,x[...,206:],valid)
        if factors is not None:
            assert tuple(factors)==(self.study_mode,)
            return {self.study_mode:output}
        return output

    def parameter_groups(self):
        body=[];heads=[]
        for n,p in self.named_parameters():
            if not p.requires_grad:continue
            (body if n.startswith('source.') and '.main_head.' not in n else heads).append(p)
        return [dict(params=body,lr=1e-5,name='private_backbones_and_input_router'),
            dict(params=heads,lr=3e-5,name='own_readouts_and_retained_corrections')]

class Evaluator(ParentEvaluator):
    @torch.no_grad()
    def __call__(self,model):
        pred,score,total=self.many(model,(model.study_mode,))[model.study_mode]
        scores=dict(POR=float(self.reference_scores['POR']),PERM=score,SW=float(self.reference_scores['SW']),TOTAL=total)
        return pred,score,scores
    def __init__(self,data,master,c,reference):
        super().__init__(data,master,c,reference)
        import train_invariant as base
        self.reference_scores=base.metric(data['truth'][data['validation']],reference)

def differentiable_gaussian_probability(weight,mean,std,edges):
    safe=torch.where(torch.isfinite(edges),edges,torch.zeros_like(edges))
    z=(safe[None,:,None]-mean[:,None,:])/std[:,None,:]
    cdf=.5*(1+torch.erf(z/math.sqrt(2.)))
    cdf=torch.where(torch.isneginf(edges)[None,:,None],torch.zeros_like(cdf),cdf)
    cdf=torch.where(torch.isposinf(edges)[None,:,None],torch.ones_like(cdf),cdf)
    mass=(cdf.diff(dim=1)*weight[:,None,:]).sum(-1).clamp_min(0)
    return mass/mass.sum(-1,keepdim=True).clamp_min(1e-12)

class DecisionObjective(nn.Module):
    def __init__(self,evaluator,master,mode):
        super().__init__();self.alpha=MODES[mode]
        self.register_buffer('log_actions',torch.as_tensor(evaluator.actions,device='cuda',dtype=torch.float32).clamp_min(1e-5).log10())
        self.register_buffer('utilities',evaluator.utilities.detach().clone())
        self.register_buffer('edges',evaluator.edges.detach().clone())
        self.median=float(master['regression_median'][1]);self.scale=float(master['regression_scale'][1])
    def reward(self,g,r,z,d):
        p=differentiable_gaussian_probability(r[:,0].softmax(-1),r[:,1],F.softplus(r[:,2])+.025,self.edges)
        pg=g.sigmoid();joint=torch.cat((pg[:,None],(1-pg[:,None])*p),-1)
        ap=((joint@self.utilities)/.05).softmax(-1)
        log_y=torch.where(d,torch.full_like(z,-2.),z*self.scale+self.median)
        true_utility=(1-(self.log_actions[None]-log_y[:,None]).abs()).clamp_min(0)
        return (ap*true_utility).sum(-1)
    def forward(self,output,labels,default,mask,quality):
        likelihood,_,_=channel_objective(output,labels,default,mask,quality,1)
        gate,raw=output;g=gate[mask][:,0].float();r=raw[mask][:,0].float()
        z=labels[mask][:,1].float();d=default[mask][:,1];q=quality[mask].detach().float()
        rewards=[]
        for begin in range(0,len(g),512):
            take=slice(begin,begin+512)
            rewards.append(checkpoint(self.reward,g[take],r[take],z[take],d[take],use_reentrant=False,preserve_rng_state=False))
        decision=(q*(1-torch.cat(rewards))).mean()/3
        return likelihood+self.alpha*decision,likelihood,decision
