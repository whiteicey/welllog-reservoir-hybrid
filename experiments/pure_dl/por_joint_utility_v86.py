"""Matched POR route fitting: current anchored function, likelihood versus action utility.

No architecture growth and no expert/backbone updates. The original Student-t3
decoder and action grid remain unchanged. Only training loss changes between arms.
"""
from pathlib import Path
import copy, hashlib, math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from por_reparam_v31 import Network as Parent, Evaluator, load_data, TARGETS
from train_scaling_v21 import channel_objective
from train_dual8222_heads80 import SOURCE

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/por_joint_utility_v86_20260930'
PREVIOUS=ROOT/'results/deep_learning/sw_residual_readout_v85_20260930'
MODES=('likelihood','joint')
LEARNING_RATE=3e-5

def reference_prediction():
    return np.load(PREVIOUS/'assembled_pred.npy')

class Network(Parent):
    def __init__(self,target,master,modes=MODES):
        super().__init__(target,master,modes=('anchored',))
        path=ROOT/'outputs/pure_dl_mechanism_v31_POR_fold5/anchored/model_best.pt'
        saved=torch.load(path,map_location='cpu',weights_only=False)
        original=self.heads['anchored']
        original.load_state_dict(saved['delta_state'],strict=True)
        self.heads=nn.ModuleDict({m:copy.deepcopy(original) for m in modes})
        self.metadata=dict(source_checkpoint=str(path.relative_to(ROOT)),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_epoch=saved['best_epoch'],source_score=saved['target_score'],
            source_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),
            trainable_scope='existing anchored route28/route29 only; frozen expert outputs',
            added_parameters=0,utility_coefficient={'likelihood':0.,'joint':1.},
            temperature=.05,utility_chunk=512,gradient_checkpointing=True,
            target_transform='log1p/inverse expm1',density='Student-t3 std parameterization',
            original_decoder_unchanged=True,optimizer_reset=True)
        self.train(False)

def differentiable_t3_probability(weight,mean,std,edges):
    # Same t3 masses as the inference decoder, with constant infinite tails
    # removed BEFORE arithmetic to keep the backward pass finite.
    safe=torch.where(torch.isfinite(edges),edges,torch.zeros_like(edges))
    phi=torch.atan((safe[None,:,None]-mean[:,None,:])/std[:,None,:])
    cdf=.5+(phi+torch.sin(phi)*torch.cos(phi))/math.pi
    cdf=torch.where(torch.isneginf(edges)[None,:,None],torch.zeros_like(cdf),cdf)
    cdf=torch.where(torch.isposinf(edges)[None,:,None],torch.ones_like(cdf),cdf)
    mass=(cdf.diff(dim=1)*weight[:,None,:]).sum(-1).clamp_min(0)
    return mass/mass.sum(-1,keepdim=True).clamp_min(1e-12)

class DecisionObjective(nn.Module):
    def __init__(self,evaluator,master,mode):
        super().__init__();self.alpha=0. if mode=='likelihood' else 1.
        self.register_buffer('actions',torch.as_tensor(evaluator.actions,device='cuda',dtype=torch.float32))
        self.register_buffer('utilities',evaluator.utilities.detach().clone())
        self.register_buffer('edges',evaluator.edges.detach().clone())
        self.median=float(master['regression_median'][0]);self.scale=float(master['regression_scale'][0])

    def reward(self,g,r,z,d):
        p=differentiable_t3_probability(r[:,0].softmax(-1),r[:,1],F.softplus(r[:,2])+.025,self.edges)
        joint=torch.cat((g.sigmoid()[:,None],(1-g.sigmoid()[:,None])*p),-1)
        action_probability=((joint@self.utilities)/.05).softmax(-1)
        y=torch.expm1(z*self.scale+self.median)
        y=torch.where(d,torch.full_like(y,.1),y)
        true_utility=(1-(self.actions[None]-y[:,None]).abs()/(.08*(y.abs()[:,None]+1e-5))).clamp_min(0)
        return (action_probability*true_utility).sum(-1)

    def forward(self,output,labels,default,mask,quality):
        likelihood,_,_=channel_objective(output,labels,default,mask,quality,0)
        gate,raw=output;g=gate[mask][:,0].float();r=raw[mask][:,0].float()
        z=labels[mask][:,0].float();d=default[mask][:,0];q=quality[mask].detach().float()
        rewards=[]
        for begin in range(0,len(g),512):
            take=slice(begin,begin+512)
            rewards.append(checkpoint(self.reward,g[take],r[take],z[take],d[take],
                use_reentrant=False,preserve_rng_state=False))
        decision=(q*(1-torch.cat(rewards))).mean()/3
        # Both arms traverse the same graph; coefficient alone changes the loss.
        return likelihood+self.alpha*decision,likelihood,decision

@torch.no_grad()
def factor_scope(model,data,mode):
    return dict(trainable_scope=model.metadata['trainable_scope'],added_parameters=0,
        frozen_expert_outputs=True,scope='full frozen-state equality checked by trainer')
