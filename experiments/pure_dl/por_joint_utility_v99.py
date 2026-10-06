"""Matched POR likelihood versus joint-action utility at the retained function.

Only the loss coefficient differs. Original predictive density, actions,
decoder and43635 existing route/location parameters remain unchanged.
"""
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from por_effective_sam_v98 import (Network as Parent, ROOT, PREVIOUS, Evaluator,
                                   load_data, TARGETS, reference_prediction,
                                   LEARNING_RATE, channel_objective)

STUDY = ROOT/'results/deep_learning/por_joint_utility_v99_20260930'
MODES = ('likelihood', 'joint')
UTILITY_WEIGHT = .5
TEMPERATURE = .05


class Network(Parent):
    def __init__(self, target, master, modes=MODES):
        # Parent constructs from the original retained v31 checkpoint, never
        # from any negative v98 output. Copy already-fixed trainable scope.
        super().__init__(target, master, modes=('adamw',))
        template = self.heads['adamw']
        self.heads = nn.ModuleDict({m:copy.deepcopy(template) for m in modes})
        self.study_metadata.update(optimizer_only_ablation=False,
                                   loss_only_ablation=True,
                                   utility_weight=UTILITY_WEIGHT,
                                   temperature=TEMPERATURE,
                                   loaded_v98_trained_weights=False)
        self.train(False)


def factor_scope(model, data, mode):
    return dict(updated='existing routes and existing location correction only',
                optimizer='AdamW', utility_weight=0. if mode=='likelihood' else UTILITY_WEIGHT,
                inference='original Student-t3 grid and hard argmax unchanged',
                backbone_and_other_factor_integrity='whole frozen-state equality after actual training')


class DecisionObjective(nn.Module):
    def __init__(self, evaluator, data):
        super().__init__()
        self.register_buffer('actions', torch.as_tensor(evaluator.actions, device='cuda',dtype=torch.float32))
        self.register_buffer('utilities', evaluator.utilities.detach().clone())
        edges=evaluator.edges
        assert bool(torch.isneginf(edges[0])) and bool(torch.isposinf(edges[-1]))
        assert bool(torch.isfinite(edges[1:-1]).all())
        self.register_buffer('finite_edges', edges[1:-1].detach().clone())
        raw=data['train_raw'].cpu().numpy();rows=data['raw_to_valid'][raw]
        assert len(raw)==578891 and (rows>=0).all() and (data['fold'][rows]!=4).all()
        assert len(np.unique(raw))==len(raw)
        physical=torch.full_like(data['labeled'],float('nan'),dtype=torch.float32)
        physical[data['train_raw']]=torch.as_tensor(data['truth'][rows,0],device='cuda',dtype=torch.float32)
        self.register_buffer('physical_training_only', physical)
        self.isolation=dict(training_rows=len(raw),validation_rows_copied=0,
                            raw_lookup_nontraining='NaN', own_target_only='POR',
                            other_target_predictions_in_loss=False)

    def decision_chunk(self,g,r,y,q):
        sd=F.softplus(r[:,2])+.025
        phi=torch.atan((self.finite_edges[None,:,None]-r[:,1,None,:])/sd[:,None,:])
        interior=.5+(phi+torch.sin(phi)*torch.cos(phi))/math.pi
        cdf=F.pad(interior,(0,0,1,0),value=0.)
        cdf=F.pad(cdf,(0,0,0,1),value=1.)
        # Finite interior-only arithmetic avoids undefined gradients at ±inf.
        p=(cdf.diff(dim=1)*r[:,0].softmax(-1)[:,None,:]).sum(-1).clamp_min(0)
        p=p/p.sum(-1,keepdim=True).clamp_min(1e-12)
        atom=g.sigmoid()[:,None]
        joint=torch.cat((atom,(1-atom)*p),dim=-1)
        predicted_utility=joint@self.utilities
        action_probability=(predicted_utility/TEMPERATURE).softmax(-1)
        # package/src/utils.py original POR tolerance kernel, physical train y.
        true_utility=(1-(self.actions[None,:]-y[:,None]).abs()/
                      (.08*(y.abs()[:,None]+1e-5))).clamp_min(0)
        reward=(action_probability*true_utility).sum(-1)
        return (q*(1-reward)).sum()

    def forward(self,output,labels,default,mask,quality,physical,mode):
        likelihood,bce,nll=channel_objective(output,labels,default,mask,quality,0)
        gate,raw=output;g=gate[mask][:,0].float();r=raw[mask][:,0].float()
        y=physical[mask];q=quality[mask].detach().float()
        if not bool(torch.isfinite(y).all()):
            raise RuntimeError('Nontraining or unlabeled physical target entered POR objective')
        pieces=[]
        for begin in range(0,len(g),1024):
            args=(g[begin:begin+1024],r[begin:begin+1024],y[begin:begin+1024],q[begin:begin+1024])
            if mode=='joint':
                value=checkpoint(self.decision_chunk,*args,use_reentrant=False)
            else:
                with torch.no_grad():value=self.decision_chunk(*args)
            pieces.append(value)
        decision=torch.stack(pieces).sum()/len(g)/3
        alpha=UTILITY_WEIGHT if mode=='joint' else 0.
        return likelihood+alpha*decision,bce,nll
