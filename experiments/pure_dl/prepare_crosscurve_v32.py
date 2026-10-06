"""Input-only cross-curve correlations; train-only normalization and exact row alignment."""
from pathlib import Path
from datetime import datetime
import hashlib,json,time
import numpy as np
import pandas as pd
import train_invariant as base
from train_dual8222_heads80 import atomic_json

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/crosscurve_features_v32_20260929'
OLD=ROOT/'results/deep_learning/feature_structure_v20_20260928'
CACHE=ROOT/'results/deep_learning/latent_fold5_v16_20260928'
PAIRS=(('AC','DEN'),('CNL','DEN'),('GR','RT'),('RT','RXO'),('PE','DEN'),('SP','GR'))
WINDOWS=(9,51,201)


def curve_values(frame,meta):
    result={}
    for name in sorted({c for pair in PAIRS for c in pair}):
        s=frame[name].replace([np.inf,-np.inf],np.nan)
        s=s.interpolate(method='linear',limit_direction='both').fillna(meta['imputation_medians'][name])
        values=s.to_numpy(np.float64)
        result[name]=np.log1p(np.maximum(values,0)) if name in ('GR','RT','RXO') else values
    return result


def cross_features(frame,meta):
    values=curve_values(frame,meta)
    floors={p['curve']:p['training_std_floor'] for p in meta['profile']}
    stats={}
    for name,v in values.items():
        s=pd.Series(v)
        for w in WINDOWS:
            rolling=s.rolling(w,center=True,min_periods=1)
            stats[name,w]=(rolling.mean().to_numpy(),rolling.var(ddof=0).fillna(0).clip(lower=0).to_numpy())
    result=[]
    for a,b in PAIRS:
        product=pd.Series(values[a]*values[b])
        for w in WINDOWS:
            ma,va=stats[a,w];mb,vb=stats[b,w]
            cov=product.rolling(w,center=True,min_periods=1).mean().to_numpy()-ma*mb
            rho=cov/np.sqrt((va+floors[a]**2)*(vb+floors[b]**2))
            result.append(np.clip(rho,-1,1))
    return np.stack(result,-1).astype(np.float32)


def main():
    if (STUDY/'feature_views.npz').exists():raise RuntimeError('Feature cache already exists; do not overwrite')
    tick=time.perf_counter()
    meta=json.loads((OLD/'feature_manifest.json').read_text(encoding='utf-8'))
    with np.load(CACHE/'data_outer.npz') as z:
        starts=z['starts'];fit=z['fit_raw'];monitor=z['monitor_raw']
    records=base.load_split(ROOT.parent/'data','train')
    assert len(records)==80 and np.array_equal(starts,np.r_[0,np.cumsum([len(f) for _,f in records])])
    assert len(fit)==578891 and len(monitor)==144665 and not np.intersect1d(fit,monitor).size
    cross=np.concatenate([cross_features(frame,meta) for _,frame in records])
    with np.load(OLD/'side_views.npz') as z:amplitude=z['amplitude'].astype(np.float32)
    cols=meta['amplitude_names'];marginal=[]
    for a,b in PAIRS:
        for w in WINDOWS:
            marginal.append(.5*(amplitude[:,cols.index(f'{a}_std{w}')]+amplitude[:,cols.index(f'{b}_std{w}')]))
    marginal=np.stack(marginal,-1)
    names=[f'{a}_{b}_corr{w}' for a,b in PAIRS for w in WINDOWS]
    scalers={};output={}
    for key,values in [('marginal',marginal),('crosscorr',cross)]:
        assert values.shape==(len(amplitude),18) and np.isfinite(values).all()
        qs=np.quantile(values[fit],[.25,.5,.75],axis=0)
        center=qs[1].astype(np.float32);scale=np.maximum(qs[2]-qs[0],1e-3).astype(np.float32)
        standardized=(values-center)/scale
        output[key]=np.clip(standardized,-8,8).astype(np.float16)
        scalers[key]=dict(median=center.tolist(),scale=scale.tolist(),clipped_training_fraction=float((np.abs(standardized[fit])>8).mean()))
    # Check the population-covariance implementation against centered products
    # at predetermined training positions, with no target labels or model scans.
    first=meta['fit_wells'][0];frame=records[first][1];v=curve_values(frame,meta)
    floors={p['curve']:p['training_std_floor'] for p in meta['profile']};errors=[]
    for j,(a,b) in enumerate(PAIRS):
        for k,w in enumerate(WINDOWS):
            for i in (0,len(frame)//2,len(frame)-1):
                lo=max(0,i-w//2);hi=min(len(frame),i+w//2+1)
                aa=v[a][lo:hi];bb=v[b][lo:hi]
                cov=((aa-aa.mean())*(bb-bb.mean())).mean()
                ref=cov/np.sqrt((aa.var()+floors[a]**2)*(bb.var()+floors[b]**2))
                errors.append(abs(float(cross[starts[first]+i,j*3+k])-np.clip(ref,-1,1)))
    assert max(errors)<1e-4,max(errors)
    np.savez(STUDY/'feature_views.npz',**output)
    report=dict(created_at=datetime.now().astimezone().isoformat(),pairs=PAIRS,windows=WINDOWS,features_per_view=18,
        raw_points=len(amplitude),training_points=len(fit),validation_points=len(monitor),fit_wells=meta['fit_wells'],monitor_wells=meta['monitor_wells'],
        original_269_inputs_unchanged=True,labels_used_to_construct_features=False,cross_well_windows=False,
        marginal='0.5*(existing normalized std_a + existing normalized std_b), followed by train-only affine scaling; redundant with original63',
        crosscorr='population covariance / sqrt((var_a+floor_a^2)*(var_b+floor_b^2)); floor=0.05*training_IQR',
        correlation_names=names,scalers=scalers,correlation_training_mean=cross[fit].mean(0).tolist(),
        correlation_training_std=cross[fit].std(0).tolist(),centered_product_check_max_abs_error=float(max(errors)),
        source_metadata=str((OLD/'feature_manifest.json').relative_to(ROOT)),source_metadata_sha256=hashlib.sha256((OLD/'feature_manifest.json').read_bytes()).hexdigest(),
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cache='feature_views.npz',seconds=time.perf_counter()-tick)
    atomic_json(STUDY/'feature_manifest.json',report)
    with (STUDY/'PROJECTS.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## 特征准备完成\n18维局部相关与18维冗余幅值对照均就绪，原269维未改；只用训练输入拟合。实际训练尚未执行。\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('scalers','fit_wells','monitor_wells','correlation_names','correlation_training_mean','correlation_training_std')},ensure_ascii=False))


if __name__=='__main__':main()
