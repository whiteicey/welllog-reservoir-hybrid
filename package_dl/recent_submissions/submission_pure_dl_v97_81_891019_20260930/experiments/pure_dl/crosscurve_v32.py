"""Frozen current predictor plus matched redundant/cross-curve feature heads.

The final posterior is corrected in one prespecified factor per target.
Only the final small head changes; all current predictor dependencies stay fixed.
"""
from pathlib import Path
import hashlib,importlib,json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from champion_all_heads_v26 import Evaluator,load_data as parent_data,TARGETS
from train_scaling_v21 import channel_objective

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/crosscurve_features_v32_20260929'
PREVIOUS=ROOT/'results/deep_learning/mechanism_checks_v31_20260929'
MODES=('marginal','crosscorr')
LEARNING_RATE=3e-4
FACTORS={'POR':'location','PERM':'gate','SW':'weight'}


def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')


def load_data():
    data,master=parent_data()
    meta=json.loads((STUDY/'feature_manifest.json').read_text(encoding='utf-8'))
    assert not meta['labels_used_to_construct_features'] and not meta['cross_well_windows']
    assert sorted(np.unique(data['wells'][data['fold']!=4]).tolist())==sorted(meta['fit_wells'])
    assert data['x'].shape==(meta['raw_points'],269)
    with np.load(STUDY/'feature_views.npz') as z:
        views=[z[m] for m in MODES]
    assert all(v.shape==(len(data['x']),18) and np.isfinite(v).all() for v in views)
    data['x']=torch.cat((data['x'],*[torch.as_tensor(v,device='cuda',dtype=data['x'].dtype) for v in views]),-1)
    return data,master


class Correction(nn.Module):
    def __init__(self,target,mode):
        super().__init__()
        if mode not in MODES:raise ValueError(mode)
        self.target=target;self.mode=mode;self.factor=FACTORS[target]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261108)
            self.input=nn.Conv1d(337,32,1)
            self.output=nn.Conv1d(32,1 if target=='PERM' else 16,1)
            nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,hidden,parent,_context,side,valid):
        if side.shape[-1]!=99:raise ValueError('Expected 63 amplitude + two 18-dimensional views')
        k=MODES.index(self.mode)
        v=side[...,63+18*k:81+18*k]
        z=torch.cat((hidden[0],hidden[1],side[...,:63].transpose(1,2).to(hidden[0].dtype),v.transpose(1,2).to(hidden[0].dtype)),1)
        z=z.masked_fill(~valid[:,None],0)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            proposal=self.output(F.gelu(self.input(z))).transpose(1,2).float()
        proposal=proposal.masked_fill(~valid[...,None],0)
        gate,raw=parent
        assert raw.shape[-3:]==(1,3,16)
        if self.factor=='gate':return gate.float()+.25*torch.tanh(proposal/.25),raw
        field=1 if self.factor=='location' else 0
        bound=.25*(F.softplus(raw[:,:,0,2,:].detach().float())+.025) if field==1 else .25
        delta=bound*torch.tanh(proposal/bound)
        fields=[raw[...,j,:] for j in range(3)]
        fields[field]=fields[field]+delta.unsqueeze(2)
        return gate,torch.stack(fields,dim=-2)


class Network(nn.Module):
    def __init__(self,target,master,modes=MODES):
        super().__init__()
        if target not in TARGETS:raise ValueError(target)
        self.target=target;self.c=TARGETS.index(target);self.factor=FACTORS[target]
        recipe=json.loads((PREVIOUS/'CURRENT_RECIPE.json').read_text(encoding='utf-8'))
        self.choice=next(c for c in recipe['choices'] if c['target']==target)
        if self.choice['model_module'] not in ('por_reparam_v31','champion_split_route_v29'):raise ValueError('Unexpected parent architecture')
        module=importlib.import_module(self.choice['model_module'])
        self.parent_mode=self.choice['mode']
        self.parent=module.Network(target,master,modes=(self.parent_mode,))
        saved=torch.load(ROOT/self.choice['checkpoint'],map_location='cpu',weights_only=False)
        assert saved['target']==target and saved['mode']==self.parent_mode
        self.parent.heads[self.parent_mode].load_state_dict(saved['delta_state'],strict=True)
        self.parent.requires_grad_(False);self.parent.eval()
        self.heads=nn.ModuleDict({m:Correction(target,m) for m in modes})
        self.metadata=dict(parent=self.choice,feature_manifest=str((STUDY/'feature_manifest.json').relative_to(ROOT)),
            feature_manifest_sha256=hashlib.sha256((STUDY/'feature_manifest.json').read_bytes()).hexdigest(),
            output_factor=self.factor,head_input=337,head_hidden=32,head_output=1 if target=='PERM' else 16,
            correction_bound=.25,location_bound_unit='fixed current component standard deviation',
            parent_input_dimensions=269,new_inputs_only_at_head=True)
        self.train(False)

    def train(self,mode=True):
        super().train(False);self.training=mode;self.heads.train(mode);return self

    @torch.no_grad()
    def extract(self,x,relative,valid,absolute):
        if x.shape[-1]!=305:raise ValueError('Expected original269 plus two18 feature views')
        old=x[...,:269].contiguous()
        h,o,context,_=self.parent.extract(old,relative,valid,absolute)
        output=self.parent.heads[self.parent_mode](h,o,context,old[...,206:],valid)
        return h,output,None,None

    def forward(self,x,relative,valid,absolute_depth=None,factors=None):
        chosen=tuple(self.heads) if factors is None else tuple(factors)
        h,output,context,_=self.extract(x,relative,valid,absolute_depth)
        return {m:self.heads[m](h,output,context,x[...,206:],valid) for m in chosen}

    def frozen_state(self):return {n:v for n,v in self.state_dict().items() if not n.startswith('heads.')}
    def trainable_groups(self,mode):return [dict(params=list(self.heads[mode].parameters()),lr=LEARNING_RATE,name='feature_correction')]
    def close(self):self.parent.close()


@torch.no_grad()
def factor_scope(model,data,mode):
    from multiscale_sequence import window_indices
    anchors=data['train_raw'][torch.tensor([0,len(data['train_raw'])//2],device='cuda')]
    ix,valid=window_indices(anchors,data['lower'],data['upper'])
    rel=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
    h,parent,ctx,_=model.extract(data['x'][ix],rel,valid,data['absolute'][ix])
    gate,raw=model.heads[mode](h,parent,ctx,data['x'][ix,...,206:],valid)
    exact={}
    if model.factor!='gate':assert torch.equal(gate,parent[0]);exact['gate']=True
    for key,j in [('weight',0),('location',1),('scale',2)]:
        if key!=model.factor:assert torch.equal(raw[...,j,:],parent[1][...,j,:]);exact[key]=True
    field={'weight':0,'location':1}.get(model.factor)
    delta=gate-parent[0] if field is None else raw[:,:,0,field,:]-parent[1][:,:,0,field,:]
    bound=.25*(F.softplus(parent[1][:,:,0,2,:])+.025) if model.factor=='location' else .25
    ratio=(delta/bound).abs()[valid]
    assert float(ratio.max())<=1.0001
    return dict(mode=mode,factor=model.factor,unchanged_factors=exact,max_bound_ratio=float(ratio.max()),
        scope='two fixed training windows; parent input path excludes both additional views')


def mechanism_probe(model,data,mode):
    result=factor_scope(model,data,mode)
    assert result['max_bound_ratio']==0.
    result.update(zero_output_initialization=True,parent_original269_only=True,feature_view=mode,
        additional_view_dimensions=18,input_projection_dimensions=337,labels_used_for_features=False)
    return result
