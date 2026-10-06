"""Coordinate retained factor corrections and routes without adding capacity.

The immutable backbone forward is shared only between a target's alternatives.
Each alternative owns copied readouts and existing v28/v29 routing parameters.
No historical checkpoint or trained source is modified.
"""
from pathlib import Path
import copy,json
import numpy as np
import torch
from torch import nn
from champion_posterior_v27 import Network as FrozenV26
from champion_all_heads_v26 import Evaluator,load_data,TARGETS
from champion_route_v28 import RoutingHead
from champion_split_route_v29 import SplitRouteHead,shift

ROOT=Path(__file__).resolve().parents[2]
PREVIOUS=ROOT/'results/deep_learning/champion_split_route_v29_20260929'
STUDY=ROOT/'results/deep_learning/champion_coadapt_v30_20260929'
MODES=('route_only','coadapt')


def reference_prediction():
    return np.load(PREVIOUS/'assembled_pred.npy')


class Bundle(nn.Module):
    def __init__(self,target,source,retained,choice,mode):
        super().__init__()
        if mode not in MODES:raise ValueError(mode)
        self.target=target;self.mode=mode
        self.readouts=nn.ModuleList([copy.deepcopy(member.main_head) for member in source.members])
        # Deep copies must not keep the source forward hooks/closures.
        for readout in self.readouts:
            readout.valid=None
            for module in readout.modules():
                module._forward_hooks.clear();module._forward_pre_hooks.clear()
            if target=='POR':readout.residual.side=None
        self.factor_head=copy.deepcopy(retained)
        old=torch.load(ROOT/choice['retained_v28_checkpoint'],map_location='cpu',weights_only=False)
        self.parent_route_mode=old['mode']
        self.route28=RoutingHead(old['mode'])
        self.route28.load_state_dict(old['delta_state'],strict=True)
        new=torch.load(ROOT/choice['checkpoint'],map_location='cpu',weights_only=False)
        assert new['target']==target and new['mode']==choice['mode']
        self.route29=SplitRouteHead(new['mode'])
        self.route29.load_state_dict(new['delta_state'],strict=True)
        self.requires_grad_(False)
        self.route28.requires_grad_(True);self.route29.requires_grad_(True)
        if mode=='coadapt':
            if target=='POR':
                for readout in self.readouts:
                    readout.residual.delta_in.requires_grad_(True)
                    readout.residual.delta_out.requires_grad_(True)
            else:
                self.factor_head.requires_grad_(True)

    def parameter_groups(self):
        routes=[];factors=[]
        for name,p in self.named_parameters():
            if not p.requires_grad:continue
            (routes if name.startswith(('route28.','route29.')) else factors).append(p)
        groups=[dict(params=routes,lr=3e-5,name='retained_routes')]
        if factors:groups.append(dict(params=factors,lr=3e-5,name='retained_factor'))
        return groups

    def expert_outputs(self,hidden,side,valid):
        outputs=[]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for readout,h in zip(self.readouts,hidden):
                readout.valid=valid
                if self.target=='POR':readout.residual.side=side
                try:outputs.append(readout(h))
                finally:
                    readout.valid=None
                    if self.target=='POR':readout.residual.side=None
        if self.factor_head is not None:
            outputs=self.factor_head.corrected_members(hidden,outputs,side,valid)
        return outputs

    def forward(self,hidden,_frozen_outputs,logits,side,valid):
        outputs=self.expert_outputs(hidden,side,valid)
        d28=self.route28.correction(hidden,side,valid)
        density=shift(logits,d28)
        atom=density if self.parent_route_mode=='joint' else logits
        return self.route29(hidden,outputs,(atom,density),side,valid)


class Network(FrozenV26):
    def __init__(self,target,master,modes=MODES):
        super().__init__(target,master,modes=())
        recipe=json.loads((PREVIOUS/'CURRENT_RECIPE.json').read_text(encoding='utf-8'))
        self.choice=next(x for x in recipe['choices'] if x['target']==target)
        self.factor={'POR':'location','PERM':'gate','SW':'weight'}[target]
        self.route_path=self.choice['checkpoint']
        self.heads=nn.ModuleDict({m:Bundle(target,self.source,self.retained,self.choice,m) for m in modes})
        self.train(False)

    def frozen_state(self):
        trainable={n for n,p in self.named_parameters() if p.requires_grad}
        return {n:v for n,v in self.state_dict().items() if n not in trainable}

    def trainable_groups(self,mode):
        return self.heads[mode].parameter_groups()


@torch.no_grad()
def factor_scope(model,data,mode):
    from multiscale_sequence import window_indices
    from paired_detached_router_sequence import routed_merge
    anchors=data['train_raw'][torch.tensor([0,len(data['train_raw'])//2],device='cuda')]
    ix,valid=window_indices(anchors,data['lower'],data['upper'])
    rel=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
    side=data['x'][ix,...,206:]
    h,old,logits,_=model.extract(data['x'][ix],rel,valid,data['absolute'][ix])
    new=model.heads[mode].expert_outputs(h,side,valid)
    checks={};changed={}
    for e,((g,r),(ng,nr)) in enumerate(zip(old,new)):
        fields={'gate':(g,ng),'weight':(r[...,0,:],nr[...,0,:]),
                'location':(r[...,1,:],nr[...,1,:]),'scale':(r[...,2,:],nr[...,2,:])}
        for key,(a,b) in fields.items():
            if mode=='route_only' or key!=model.factor:
                assert torch.equal(a,b),(model.target,mode,e,key)
                checks[f'expert{e}.{key}']=True
            else:changed[f'expert{e}.{key}']=float((a.float()-b.float()).abs()[valid].max())
    return dict(mode=mode,unchanged_expert_factors=checks,selected_factor_change_max=changed,
                scope='two fixed training windows; compared with retained v26 expert outputs, routes checked in frozen-state assertions')
