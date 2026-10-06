"""Fixed selected stage DAG with explicit, guarded fold-local source bindings.

Historical builders are reused only inside a scoped constructor context. Every
historical fold5 weight path must resolve to this fold's own trained artifact.
Original fold labels and population are never rewritten.
"""
from pathlib import Path
from contextlib import contextmanager
import importlib
import json
import numpy as np
import torch
from torch import nn
from fullcv_context_v97 import ROOT, STUDY, sha

STAGES = [
    ('por_density', 'POR', 'density_only', 3e-5,256,False,0.),
    ('perm_gate', 'PERM', 'gate_only',3e-5,256,False,0.),
    ('por_mean','POR','mean_only',3e-5,256,False,0.),
    ('por_point','POR','point32',3e-5,256,False,0.),
    ('por_amplitude','POR','amplitude',3e-4,256,False,0.),
    ('perm_factor','PERM','gate',3e-4,256,False,0.),
    ('por_route','POR','conditional',3e-4,256,False,0.),
    ('perm_route','PERM','joint',3e-4,256,False,0.),
    ('por_split','POR','split',3e-4,256,False,0.),
    ('perm_split','PERM','coupled',3e-4,256,False,0.),
    ('por_coadapt','POR','coadapt',3e-5,256,False,0.),
    ('por_anchored','POR','anchored',3e-5,256,False,0.),
    ('sw_scratch','SW','long_section',6e-4,1024,False,0.),
    ('sw_well','SW','well',1e-4,1024,False,0.),
    ('sw_additive','SW','additive',3e-4,1024,False,0.),
    ('sw_utility','SW','a100',1e-4,1024,False,1.),
    ('sw_sam','SW','r100',1e-4,1024,True,1.),
    ('sw_density','SW','likelihood',1e-4,1024,True,0.),
]


def fold_dir(fold):return ROOT/'outputs/pure_dl_fullcv_v97'/f'fold_{fold}'
def root_dir(fold):return ROOT/'outputs/pure_dl_fullcv_v97/root' if fold==1 else fold_dir(fold)/'root'
def root_path(fold):return root_dir(fold)/f'fold_{fold}.pt'
def stage_dir(fold,stage):return fold_dir(fold)/stage
def best_path(fold,stage):return stage_dir(fold,stage)/'model_best.pt'


def path_bindings(fold):
    o=ROOT/'outputs'
    paths={
        o/'pure_dl_paired_local_context_router80_sequence/fold_5.pt':root_path(fold),
        o/'pure_dl_dual8222_selected_round2_fold5/selected_heads.pt':fold_dir(fold)/'selected_factors.pt',
        o/'pure_dl_dual8222_v4_por_mean_only80_fold5/selected_heads.pt':best_path(fold,'por_mean'),
        o/'pure_dl_dual8222_v5_location_point3280_fold5/readout_checkpoint.pt':best_path(fold,'por_point'),
        o/'pure_dl_independent_v8_POR_dual80_fold5/model_best.pt':fold_dir(fold)/'private_POR.pt',
        o/'pure_dl_independent_v8_PERM_dual80_fold5/model_best.pt':fold_dir(fold)/'private_PERM.pt',
        o/'pure_dl_champion_shape_v25_amplitude_fold5/model_best.pt':best_path(fold,'por_amplitude'),
        o/'pure_dl_all_heads_v26_PERM_fold5/gate/model_best.pt':best_path(fold,'perm_factor'),
        o/'pure_dl_champion_route_v28_POR_fold5/conditional/model_best.pt':best_path(fold,'por_route'),
        o/'pure_dl_champion_route_v28_PERM_fold5/joint/model_best.pt':best_path(fold,'perm_route'),
        o/'pure_dl_split_route_v29_POR_fold5/split/model_best.pt':best_path(fold,'por_split'),
        o/'pure_dl_split_route_v29_PERM_fold5/coupled/model_best.pt':best_path(fold,'perm_split'),
        o/'pure_dl_coadapt_v30_POR_fold5/coadapt/model_best.pt':best_path(fold,'por_coadapt'),
        o/'pure_dl_mechanism_v31_POR_fold5/anchored/model_best.pt':best_path(fold,'por_anchored'),
        o/'pure_dl_divergent_v45_SW_long_section_fold5/model_best.pt':best_path(fold,'sw_scratch'),
        o/'pure_dl_well_conditioned_v46_SW_well_fold5/model_best.pt':best_path(fold,'sw_well'),
        o/'pure_dl_sw_interaction_v56_SW_additive_fold5/model_best.pt':best_path(fold,'sw_additive'),
        o/'pure_dl_sw_utility_strength_v82_SW_a100_fold5/model_best.pt':best_path(fold,'sw_utility'),
        o/'pure_dl_sw_sam_radius_v94_SW_r100_fold5/model_best.pt':best_path(fold,'sw_sam'),
        o/'pure_dl_sw_factor_sam_v96_SW_likelihood_fold5/model_best.pt':best_path(fold,'sw_density'),
    }
    recipes={
        ROOT/'results/deep_learning/champion_route_v28_20260929/CURRENT_RECIPE.json':
            {'choices':[dict(target=t,mode=m,checkpoint=str(best_path(fold,s).relative_to(ROOT)))
                for t,m,s in [('POR','conditional','por_route'),('PERM','joint','perm_route')]]},
        ROOT/'results/deep_learning/champion_split_route_v29_20260929/CURRENT_RECIPE.json':
            {'choices':[dict(target=t,mode=m,checkpoint=str(best_path(fold,s).relative_to(ROOT)),
                retained_v28_checkpoint=str(best_path(fold,p).relative_to(ROOT)))
                for t,m,s,p in [('POR','split','por_split','por_route'),('PERM','coupled','perm_split','perm_route')]]},
        ROOT/'results/deep_learning/champion_coadapt_v30_20260929/CURRENT_RECIPE.json':
            {'choices':[dict(target='POR',mode='coadapt',checkpoint=str(best_path(fold,'por_coadapt').relative_to(ROOT)))]},
    }
    return paths,recipes


@contextmanager
def constructor_bindings(fold, trace):
    paths,recipes=path_bindings(fold)
    def norm(p):return Path(p).resolve()
    paths={norm(k):norm(v) for k,v in paths.items()};recipes={norm(k):v for k,v in recipes.items()}
    original_load=torch.load;original_bytes=Path.read_bytes;original_text=Path.read_text
    def resolve(path):
        p=norm(path)
        if p in paths:return paths[p]
        lower=p.as_posix().lower()
        if '/outputs/' in lower and p.suffix=='.pt' and ('fold5' in lower or 'fold_5' in lower):
            raise ValueError(f'Unmapped fifth trained source forbidden: {p}')
        return p
    def load(path,*a,**kw):
        p=resolve(path)
        if p.is_relative_to(ROOT/'outputs'):
            if not p.is_relative_to(fold_dir(fold)) and p!=root_path(fold):raise ValueError(f'Cross-fold source forbidden: {p}')
            trace.append(dict(requested=str(path),actual=str(p.relative_to(ROOT)),sha256=sha_unpatched(p),fold=fold))
        return original_load(p,*a,**kw)
    def sha_unpatched(p):
        import hashlib
        return hashlib.sha256(original_bytes(p)).hexdigest()
    def read_bytes(path):return original_bytes(resolve(path))
    def read_text(path,*a,**kw):
        p=norm(path)
        if p in recipes:
            trace.append(dict(requested=str(p.relative_to(ROOT)),fold=fold,recipe=recipes[p]))
            return json.dumps(recipes[p])
        return original_text(path,*a,**kw)
    torch.load=load;Path.read_bytes=read_bytes;Path.read_text=read_text
    try:yield
    finally:torch.load=original_load;Path.read_bytes=original_bytes;Path.read_text=original_text


class Chosen(nn.Module):
    def __init__(self,model,c,mode=None,legacy=False,bank=False):
        super().__init__();self.model=model;self.c=c;self.mode=mode;self.legacy=legacy;self.bank=bank
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        kwargs=dict(absolute_depth=absolute_depth)
        if self.bank:kwargs['well_inputs']=well_inputs
        out=self.model(x,relative,valid,**kwargs)
        if self.mode is not None:out=out[self.mode]
        if self.legacy:out=(out[0][:,:,self.c:self.c+1],out[1][:,:,self.c:self.c+1])
        return out


def make_model(fold,stage,master,metadata):
    spec=next(s for s in STAGES if s[0]==stage);_,target,mode,*_=spec;c=('POR','PERM','SW').index(target)
    trace=[]
    with constructor_bindings(fold,trace):
        if stage in ('por_density','perm_gate','por_mean','por_point'):
            from paired_local_context_router_sequence import Network as Original,retained_indices,calibration_indices
            numeric,conditioning=calibration_indices(master['selected_feature_names'])
            original=Original(retain_indices=retained_indices(master['selected_feature_names']),numeric_indices=numeric,conditioning_indices=conditioning)
            original.load_state_dict(master['state_dict'],strict=True)
            original.requires_grad_(False).eval()
            if stage in ('por_density','perm_gate'):
                from dual8222_factorized_readout import Network
                model=Network(original,mode);model.set_trainable_targets([c==j for j in range(3)])
            elif stage=='por_mean':
                from dual8222_por_subspace import Network
                selected=torch.load(fold_dir(fold)/'selected_factors.pt',map_location='cpu',weights_only=False)
                model=Network(original,selected,mode)
            else:
                from dual8222_location_readout_v5 import Network
                selected=torch.load(best_path(fold,'por_mean'),map_location='cpu',weights_only=False)
                model=Network(original,selected,mode)
            wrapped=Chosen(model,c,legacy=True)
        else:
            modules={'por_amplitude':'champion_shape_v25','perm_factor':'champion_all_heads_v26',
                'por_route':'champion_route_v28','perm_route':'champion_route_v28',
                'por_split':'champion_split_route_v29','perm_split':'champion_split_route_v29',
                'por_coadapt':'champion_coadapt_v30','por_anchored':'por_reparam_v31',
                'sw_scratch':'divergent_models_v45','sw_well':'well_conditioned_v46',
                'sw_additive':'sw_interaction_v56','sw_utility':'sw_utility_strength_v82',
                'sw_sam':'sw_sam_radius_v94','sw_density':'sw_factor_sam_v96'}
            module=importlib.import_module(modules[stage])
            if stage=='por_amplitude':model=module.make_model(mode,master);wrapped=Chosen(model,c)
            elif target=='SW':model=module.Network(mode,metadata);wrapped=Chosen(model,c,bank=stage!='sw_scratch')
            else:
                key='factors' if stage=='perm_factor' else 'modes'
                model=module.Network(target,master,**{key:(mode,)})
                wrapped=Chosen(model,c,mode=mode)
    return wrapped,spec,trace


def export_selected(model,fold,stage,config,best_epoch,best_score):
    raw=model.model;target=config['target'];mode=config['mode']
    result=dict(config=config,target=target,mode=mode,best_epoch=best_epoch,target_score=best_score,
                original_fold=fold,source_checkpoint=str(root_path(fold).relative_to(ROOT)),source_sha256=sha(root_path(fold)))
    def state(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    if stage in ('por_density','perm_gate'):result['head_state_dict']=state(raw.heads[config['channel']])
    elif stage=='por_mean':
        result.update(head_state_dicts=raw.export_legacy_heads(),source_kinds=['target_group']*3)
    elif stage=='por_point':
        result.update(readout=raw.export_readout(),initialization_manifest=torch.load(best_path(fold,'por_mean'),map_location='cpu',weights_only=False))
    elif stage.startswith(('por_route','perm_route','por_split','perm_split','por_coadapt','por_anchored','perm_factor')):
        result['delta_state']=state(raw.heads[mode])
    else:result['state_dict']=state(raw)
    return result
