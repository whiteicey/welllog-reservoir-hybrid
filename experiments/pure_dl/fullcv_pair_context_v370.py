"""Authorized two-recipe grouped CV, fold-local shared input cache, own Q."""
from pathlib import Path
import hashlib,json,os
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/two_seed_fullcv_v370_20261002'
OUTPUT=ROOT/'outputs/pure_dl_two_seed_fullcv_v370'
STAGES=('SOURCE','POR113','POR151','PERM154','SW104','POR341','PERM336','SW351')
BASE_SEEDS={'original':20261006,'seed_20261013':20261009}

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def own_dir(identity,fold):
    assert identity in BASE_SEEDS and fold in range(1,5)
    return OUTPUT/f'fold_{fold}'/identity
def phase_dir(identity,fold,stage):return own_dir(identity,fold)/stage

def require_authorization():
    plan=read(STUDY/'PLAN.json')
    assert plan['remaining_folds_authorized'] and plan['per_fold_per_head_scan_authorized']
    assert plan['models']==list(BASE_SEEDS) and plan['folds_to_train']==[1,2,3,4]
    batch=read(ROOT/'results/deep_learning/complete_seed_compare_v362_20261002/BATCH_STATE.json')
    if batch['status']!='complete' or batch['completed_models']!=5:
        raise RuntimeError('Finish the current five-seed batch before fullCV GPU work')
    import psutil
    for p in psutil.process_iter(['pid','cmdline']):
        try:
            if 'run_complete_seeds_v362r1.py' in ' '.join(p.info['cmdline'] or []):
                raise RuntimeError('The previous controller has not exited')
        except psutil.Error:pass
    return plan

def save_fold_cache(source_data,master,out,fold):
    """Same-fold label/input fitting is seed invariant; source conflict-Q is not."""
    from fullcv_resume_v97 import atomic_save
    from fullcv_context_v97 import original_population,validate_metadata_source
    pop=original_population();validate_metadata_source(master,pop,fold-1)
    cache=OUTPUT/f'fold_{fold}'/'shared_input_data.pt'
    metadata_keys=('feature_median','feature_scale','regression_median','regression_scale',
                   'absolute_depth_median','absolute_depth_scale','selected_feature_names')
    signature={k:master[k] for k in metadata_keys}
    data={k:v.cpu() if torch.is_tensor(v) else v for k,v in source_data.items()
          if k not in ('quality','qinfo','quality_ready')}
    if cache.exists():
        saved=torch.load(cache,map_location='cpu',weights_only=False)
        assert saved['normalization']==signature
        # Verify exactly the arrays reused; no source/model weights in this file.
        for k,v in data.items():
            if torch.is_tensor(v):assert torch.equal(v,saved['data'][k]),k
            elif isinstance(v,np.ndarray):assert np.array_equal(v,saved['data'][k]),k
        del saved
    else:atomic_save(dict(data=data,normalization=signature,fold=fold,learned_weights=False),cache)
    side_path=cache.parent/'amplitude.npz'
    if not side_path.exists():
        # Same 63 amplitude features as the fifth recipe, fit on this fold only.
        import pandas as pd
        from fullcv_resume_v97 import atomic_npz
        curves=('GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT')
        train=source_data['population_rows']['train_wells'];raw=source_data['population_rows']['training_raw']
        med={n:float(np.nanmedian(np.concatenate([pop['records'][w][1][n].to_numpy(np.float64) for w in train]))) for n in curves}
        views=[]
        for _,frame in pop['records']:
            cols=[]
            for name in curves:
                s=frame[name].replace([np.inf,-np.inf],np.nan).interpolate(method='linear',limit_direction='both').fillna(med[name])
                values=s.to_numpy(np.float64)
                if name in ('GR','RXO','RT'):values=np.log1p(np.maximum(values,0))
                s=pd.Series(values);cols.append(values)
                cols.extend(s.rolling(w,center=True,min_periods=1).mean().to_numpy() for w in (9,51,201))
                cols.extend(s.rolling(w,center=True,min_periods=1).std(ddof=0).fillna(0).to_numpy() for w in (9,51,201))
            views.append(np.stack(cols,-1).astype(np.float32))
        x=np.concatenate(views);q=np.quantile(x[raw],[.25,.5,.75],axis=0)
        center=q[1].astype(np.float32);scale=np.maximum(q[2]-q[0],1e-3).astype(np.float32)
        amplitude=np.clip((x-center)/scale,-8,8).astype(np.float16)
        assert amplitude.shape==(len(data['x']),63) and np.isfinite(amplitude).all()
        atomic_npz(side_path,amplitude=amplitude)
        write(cache.parent/'amplitude_manifest.json',dict(fold=fold,fit_wells=train,labels_used=False,
              imputation_medians=med,center=center.tolist(),scale=scale.tolist(),sha256=sha(side_path)))
    atomic_save(dict(shared_input_path=str(cache),shared_input_sha256=sha(cache),fold=fold,
                     quality=source_data['quality'].cpu(),quality_ready=True,qinfo=source_data['qinfo'],
                     source_sha256=sha(Path(out)/f'fold_{fold}.pt')),Path(out)/'prepared_data.pt')

def load_fold_cache(own,source,loader=torch.load):
    descriptor=loader(own/'SOURCE/prepared_data.pt',map_location='cpu',weights_only=False)
    cache=Path(descriptor['shared_input_path'])
    assert cache==own.parent/'shared_input_data.pt' and sha(cache)==descriptor['shared_input_sha256']
    assert sha(source)==descriptor['source_sha256']
    shared=loader(cache,map_location='cpu',weights_only=False)
    data=shared['data'];master=loader(source,map_location='cpu',weights_only=False)
    assert shared['fold']==descriptor['fold'] and not shared['learned_weights']
    for k,v in shared['normalization'].items():assert master[k]==v,k
    data.update({k:descriptor[k] for k in ('quality','quality_ready','qinfo')})
    for k,v in list(data.items()):
        if torch.is_tensor(v):data[k]=v.cuda()
    from fullcv_context_v97 import require_training_context
    require_training_context(data)
    return data,master

def compact_fold_data(own,source,stage,loader=torch.load):
    from fullcv_context_v97 import add_well_bank
    data,master=load_fold_cache(own,source,loader)
    metadata={k:master[k] for k in ('centers','density_edges','selected_feature_names')}
    metadata.update(add_well_bank(data))
    metadata.update(sw_target_median=float(master['regression_median'][2]),sw_target_scale=float(master['regression_scale'][2]))
    if stage in ('por','perm'):
        with np.load(own.parent/'amplitude.npz') as z:side=torch.as_tensor(z['amplitude'],device='cuda')
        assert side.shape==(len(data['x']),63)
        data['x']=torch.cat((data['x'],side),-1)
    return data,master,metadata
