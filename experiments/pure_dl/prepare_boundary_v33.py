"""Input-only boundary diagnostic and matched fixed/segmented statistics.

Single preregistered rule: RMS left/right 5-point mean contrast, robust-scaled
by training-input IQR; train-input 95th percentile, peak separation 11 points.
No target values enter boundary detection, segmentation, or feature scaling.
"""
from pathlib import Path
from datetime import datetime
import json,time
import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks
import train_invariant as base
from train import metric,dmask
from train_dual8222_heads80 import atomic_json
from prepare_feature_structure_v20 import CURVES,transform

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/boundary_context_v33_20260929'
PARENT=ROOT/'results/deep_learning/mechanism_checks_v31_20260929'
WIDTH=5
RADIUS=25

def moments(x,left,right):
    cs=np.vstack((np.zeros((1,x.shape[1])),np.cumsum(x,axis=0,dtype=np.float64)))
    ss=np.vstack((np.zeros((1,x.shape[1])),np.cumsum(x*x,axis=0,dtype=np.float64)))
    count=(right-left)[:,None]
    mean=(cs[right]-cs[left])/count
    std=np.sqrt(np.maximum((ss[right]-ss[left])/count-mean*mean,0.))
    return np.concatenate((mean,std),axis=1)

def kernels(y,p):
    return np.column_stack((np.maximum(0,1-np.abs(p[:,0]-y[:,0])/(.08*(np.abs(y[:,0])+1e-5))),
        np.maximum(0,1-np.abs(np.log10(np.maximum(p[:,1]/np.maximum(y[:,1],1e-5),1e-5)))),
        np.maximum(0,1-np.abs(p[:,2]-y[:,2])/(.05*(np.abs(y[:,2])+1e-5)))))

def main():
    started=time.perf_counter();torch.set_num_threads(2)
    if (STUDY/'diagnostic.json').exists():raise RuntimeError('Completed diagnostic already exists; do not refit or overwrite')
    STUDY.mkdir(parents=True,exist_ok=True)
    cache=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
    amp=json.loads((ROOT/'results/deep_learning/feature_structure_v20_20260928/feature_manifest.json').read_text(encoding='utf-8'))
    starts=np.asarray(cache['starts']);fit=cache['train_raw'].numpy()
    valid_to_raw=np.flatnonzero(cache['raw_to_valid']>=0)
    val=valid_to_raw[np.asarray(cache['validation'])]
    assert len(fit)==578891 and len(val)==144665
    records=base.load_split(ROOT.parent/'data','train')
    assert len(records)==80 and np.array_equal(starts,np.r_[0,np.cumsum([len(f) for _,f in records])])
    fit_wells=amp['fit_wells'];monitor_wells=amp['monitor_wells']
    assert set(fit_wells).isdisjoint(monitor_wells)
    iqr=np.array([p['training_IQR'] for p in amp['profile']])
    raw=[];contrasts=[]
    for _,frame in records:
        x=np.column_stack([transform(frame[c].replace([np.inf,-np.inf],np.nan).interpolate(limit_direction='both').fillna(amp['imputation_medians'][c]).to_numpy(np.float64),c) for c in CURVES])
        assert np.isfinite(x).all()
        n=len(x);idx=np.arange(WIDTH,n-WIDTH+1)
        cs=np.vstack((np.zeros((1,9)),np.cumsum(x,axis=0,dtype=np.float64)))
        diff=((cs[idx+WIDTH]-cs[idx])-(cs[idx]-cs[idx-WIDTH]))/WIDTH
        score=np.zeros(n,dtype=np.float64)
        score[idx]=np.sqrt(np.mean((diff/iqr)**2,axis=1))
        raw.append(x);contrasts.append(score)
    fullscore=np.concatenate(contrasts)
    threshold=float(np.quantile(fullscore[fit],.95))
    fixed=[];segmented=[];distances=[];widths=[];peaks_by_well=[]
    for wi,(x,score) in enumerate(zip(raw,contrasts)):
        n=len(x);i=np.arange(n)
        peak,_=find_peaks(score,height=threshold,distance=11)
        peak=peak[(peak>=WIDTH)&(peak<=n-WIDTH)]
        edges=np.r_[0,peak,n]
        segment=np.searchsorted(edges,i,side='right')-1
        left=np.maximum(i-RADIUS,0);right=np.minimum(i+RADIUS+1,n)
        cutleft=np.maximum(left,edges[segment]);cutright=np.minimum(right,edges[segment+1])
        assert (cutright>cutleft).all()
        fixed.append(moments(x,left,right).astype(np.float32))
        segmented.append(moments(x,cutleft,cutright).astype(np.float32))
        if len(peak):
            p=np.r_[-n,peak,2*n];pos=np.searchsorted(p,i)
            dist=np.minimum(i-p[pos-1],p[pos]-i)
        else:dist=np.full(n,n)
        distances.append(dist.astype(np.int32));widths.append((cutright-cutleft).astype(np.int32))
        peaks_by_well.append(dict(well_index=wi,name=records[wi][0],n_points=n,peaks=len(peak),peak_raw_rows=(peak+starts[wi]).tolist()))
    views={'fixed':np.concatenate(fixed),'segmented':np.concatenate(segmented)}
    quant=np.quantile(views['fixed'][fit],[.25,.5,.75],axis=0)
    center=quant[1].astype(np.float32);scale=np.maximum(quant[2]-quant[0],1e-3).astype(np.float32)
    scaled={m:np.clip((v-center)/scale,-8,8).astype(np.float16) for m,v in views.items()}
    assert all(v.shape==(730268,18) and np.isfinite(v).all() for v in scaled.values())
    distance=np.concatenate(distances);count=np.concatenate(widths)
    np.savez(STUDY/'feature_views.npz',**scaled)
    np.savez(STUDY/'boundary_arrays.npz',input_contrast=fullscore.astype(np.float32),distance=distance,local_count=count)
    meta=dict(created_at=datetime.now().astimezone().isoformat(),raw_points=730268,fit_points=len(fit),monitor_points=len(val),
        fit_wells=fit_wells,monitor_wells=monitor_wells,curves=list(CURVES),side_features=18,modes=['fixed','segmented'],
        boundary_rule='RMS(left5-right5)/training_curve_IQR; q95 on original fit points; scipy peak separation11',
        quantile=.95,threshold=threshold,min_peak_distance=11,window=51,labels_used_to_construct_features=False,
        cross_well_windows=False,scaler_fit='fixed51 view, same constants for both modes, original fit points only',
        median=center.tolist(),scale=scale.tolist(),imputation_source='feature_structure_v20_20260928/feature_manifest.json',
        peak_records=peaks_by_well,feature_names=[c+'_'+s for s in ['mean51','std51'] for c in CURVES],
        feature_change_fraction=float((np.abs(views['fixed']-views['segmented'])>1e-5).any(axis=1).mean()),
        fit_near_fraction=float((distance[fit]<=5).mean()),monitor_near_fraction=float((distance[val]<=5).mean()))
    atomic_json(STUDY/'feature_manifest.json',meta)
    truth=cache['truth'][cache['fold']==4];wells=cache['wells'][cache['fold']==4]
    pred=np.load(PARENT/'assembled_pred.npy');ref=json.loads((PARENT/'CURRENT_RECIPE.json').read_text(encoding='utf-8'))['scores']
    scores=metric(truth,pred)
    assert np.array_equal(cache['raw_to_valid'][val],np.flatnonzero(cache['fold']==4))
    assert max(abs(float(scores[k])-ref[k]) for k in ref)<1e-9
    u=kernels(truth,pred);dm=dmask(truth);loss=100*(1-u);weights=np.array([.3,.35,.35])
    groups={'near_0_5':distance[val]<=5,'middle_6_25':(distance[val]>5)&(distance[val]<=25),'far_gt25':distance[val]>25}
    rows=[];wellrows=[]
    for gi,g in groups.items():
        for c,t in enumerate(('POR','PERM','SW')):
            for population,pm in [('all',np.ones(len(val),bool)),('default',dm[:,c]),('numeric',~dm[:,c])]:
                sel=g&pm;n=int(sel.sum())
                rows.append(dict(group=gi,target=t,population=population,n=n,
                    score=float(100*u[sel,c].mean()) if n else None,
                    mean_loss=float(loss[sel,c].mean()) if n else None,
                    zero_fraction=float((u[sel,c]==0).mean()) if n else None,
                    TOTAL_loss_contribution=float(weights[c]*loss[sel,c].sum()/len(val))))
    for w in np.unique(wells):
        for c,t in enumerate(('POR','PERM','SW')):
            near=(wells==w)&groups['near_0_5']&(~dm[:,c]);far=(wells==w)&groups['far_gt25']&(~dm[:,c])
            if near.sum()>=20 and far.sum()>=20:
                wellrows.append(dict(well_index=int(w),target=t,near_n=int(near.sum()),far_n=int(far.sum()),numeric_near_loss_minus_far=float(loss[near,c].mean()-loss[far,c].mean())))
    summary={}
    for t in ('POR','PERM','SW'):
        pair={z['group']:z for z in rows if z['target']==t and z['population']=='numeric'}
        wr=[z for z in wellrows if z['target']==t]
        summary[t]=dict(numeric_near_score=pair['near_0_5']['score'],numeric_far_score=pair['far_gt25']['score'],
            wells_with_higher_numeric_loss_near=sum(z['numeric_near_loss_minus_far']>0 for z in wr),eligible_wells=len(wr))
    diag=dict(observed_at=datetime.now().astimezone().isoformat(),scores=scores,input_boundary_threshold=threshold,
        peak_count=sum(x['peaks'] for x in peaks_by_well),training_input_only_threshold=True,
        near_fraction=float(groups['near_0_5'].mean()),groups=rows,per_well=wellrows,summary=summary,
        caveat='Exploratory association on repeatedly used original fifth; no causal or independent-test claim. No targets used in features.',seconds=time.perf_counter()-started)
    atomic_json(STUDY/'diagnostic.json',diag)
    pd.DataFrame(rows).to_csv(STUDY/'boundary_error_groups.csv',index=False)
    pd.DataFrame(wellrows).to_csv(STUDY/'boundary_error_by_well.csv',index=False)
    print('BOUNDARY_DIAGNOSTIC',json.dumps({k:v for k,v in diag.items() if k not in ('groups','per_well')},ensure_ascii=False),flush=True)
    print('FEATURES',json.dumps({k:meta[k] for k in ['threshold','feature_change_fraction','fit_near_fraction','monitor_near_fraction']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
