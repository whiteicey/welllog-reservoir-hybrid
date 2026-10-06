"""Prepare two input-only side views. All global constants use fit wells only."""
from pathlib import Path
from datetime import datetime
import json,sys,time
import numpy as np
import pandas as pd
import train_invariant as base
from train_dual8222_heads80 import atomic_json

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/feature_structure_v20_20260928'
CACHE=ROOT/'results/deep_learning/latent_fold5_v16_20260928'
CURVES=('GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT')
WINDOWS=(9,51,201)

def transform(values,name):
    return np.log1p(np.maximum(values,0)) if name in ('GR','RXO','RT') else values

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if (STUDY/'side_views.npz').exists():raise RuntimeError('Existing cache immutable')
    started=time.perf_counter()
    manifest=json.loads((CACHE/'data_outer_manifest.json').read_text(encoding='utf-8'))
    assert not manifest['outer_fifth_used_for_fitting']
    with np.load(CACHE/'data_outer.npz') as z:
        starts=z['starts'];fit=z['fit_raw'];monitor=z['monitor_raw'];fitmask=z['fitmask'];wells=z['wells']
    records=base.load_split(ROOT.parent/'data','train');fit_wells=manifest['fit_wells']
    assert len(records)==80 and np.array_equal(starts,np.r_[0,np.cumsum([len(f) for _,f in records])])
    medians={}
    for name in CURVES:
        values=np.concatenate([records[w][1][name].to_numpy(np.float64) for w in fit_wells])
        medians[name]=float(np.nanmedian(values))
        if not np.isfinite(medians[name]):raise ValueError('No finite training input: '+name)
    raw=[];missing=[]
    for _,frame in records:
        channels=[]
        for name in CURVES:
            s=frame[name].replace([np.inf,-np.inf],np.nan)
            s=s.interpolate(method='linear',limit_direction='both').fillna(medians[name])
            channels.append(transform(s.to_numpy(np.float64),name))
        raw.append(np.stack(channels,-1))
        missing.append(frame[list(CURVES)].isna().to_numpy())
    full=np.concatenate(raw);missing=np.concatenate(missing)
    q=np.quantile(full[fit],[.25,.5,.75],axis=0)
    iqr=np.maximum(q[2]-q[0],1e-6);floor=.05*iqr
    amplitude=[];shape=[];floor_hits=np.zeros((9,3),np.int64)
    for wi,values in enumerate(raw):
        amp_cols=[];shape_cols=[]
        fit_local=fitmask[starts[wi]:starts[wi+1]]
        for ci,name in enumerate(CURVES):
            s=pd.Series(values[:,ci])
            means={w:s.rolling(w,center=True,min_periods=1).mean().to_numpy() for w in WINDOWS}
            stds={w:s.rolling(w,center=True,min_periods=1).std(ddof=0).fillna(0).to_numpy() for w in WINDOWS}
            denom={w:np.sqrt(stds[w]**2+floor[ci]**2) for w in WINDOWS}
            # Seven amplitude descriptors; same number as seven shape descriptors.
            amp_cols.extend([s.to_numpy(),means[9],means[51],means[201],stds[9],stds[51],stds[201]])
            change5=(s-s.shift(5)).fillna(0).to_numpy()
            change20=(s-s.shift(20)).fillna(0).to_numpy()
            shape_cols.extend([(values[:,ci]-means[w])/denom[w] for w in WINDOWS])
            shape_cols.extend([(means[9]-means[51])/denom[51],(means[51]-means[201])/denom[201],change5/denom[51],change20/denom[201]])
            for k,w in enumerate(WINDOWS):floor_hits[ci,k]+=int((stds[w][fit_local]<floor[ci]).sum())
        amplitude.append(np.stack(amp_cols,-1).astype(np.float32))
        shape.append(np.stack(shape_cols,-1).astype(np.float32))
    arrays={'amplitude':np.concatenate(amplitude),'shape':np.concatenate(shape)}
    output={};scales={};profile=[]
    for key,x in arrays.items():
        assert x.shape==(len(full),63) and np.isfinite(x).all()
        qs=np.quantile(x[fit],[.25,.5,.75],axis=0)
        center=qs[1].astype(np.float32);scale=np.maximum(qs[2]-qs[0],1e-3).astype(np.float32)
        z=(x-center)/scale
        output[key]=np.clip(z,-8,8).astype(np.float16)
        scales[key]=dict(median=center.tolist(),scale=scale.tolist(),training_clipped_fraction=float((np.abs(z[fit])>8).mean()))
    for ci,name in enumerate(CURVES):
        well_medians=np.array([np.median(raw[w][:,ci]) for w in fit_wells])
        spread=float(np.quantile(well_medians,.75)-np.quantile(well_medians,.25))
        profile.append(dict(curve=name,transform='log1p(max(x,0))' if name in ('GR','RXO','RT') else 'identity',
            training_IQR=float(iqr[ci]),training_std_floor=float(floor[ci]),
            train_well_median_IQR_over_global_IQR=spread/float(iqr[ci]),training_missing_fraction=float(missing[fit,ci].mean()),
            local_std_below_floor_fraction=(floor_hits[ci]/len(fit)).tolist()))
    names_amp=[c+'_'+s for c in CURVES for s in ('value','mean9','mean51','mean201','std9','std51','std201')]
    names_shape=[c+'_'+s for c in CURVES for s in ('contrast9','contrast51','contrast201','band9_51','band51_201','change5_std51','change20_std201')]
    np.savez(STUDY/'side_views.npz',**output)
    meta=dict(created_at=datetime.now().astimezone().isoformat(),curves=list(CURVES),raw_points=len(full),side_features=63,
        fit_wells=fit_wells,monitor_wells=manifest['monitor_wells'],fit_points=len(fit),monitor_points=len(monitor),
        original206_unchanged=True,global_constants_fitted_only_on_original64=True,labels_used_to_construct_features=False,
        imputation_medians=medians,local_windows=list(WINDOWS),local_window_units='sample points; centered within one well',
        std_floor_fraction_of_training_IQR=.05,profile=profile,scalers=scales,amplitude_names=names_amp,shape_names=names_shape,
        cache='side_views.npz',seconds=time.perf_counter()-started)
    atomic_json(STUDY/'feature_manifest.json',meta)
    print('FEATURES_READY',json.dumps({k:v for k,v in meta.items() if k not in ('scalers','amplitude_names','shape_names','fit_wells','monitor_wells')},ensure_ascii=False))

if __name__=='__main__':main()
