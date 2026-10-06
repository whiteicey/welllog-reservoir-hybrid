from pathlib import Path
import sys,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'package'))
from src.data import read_curve
from directional_depth_features import directional_features,FEATURE_NAMES,CURVES

def main():
    out=ROOT/'outputs/pure_dl_feature_expansion_study';out.mkdir(exist_ok=True)
    cfg=json.loads((ROOT/'outputs/pure_dl_paired_adaptive_dilation80_sequence/config.json').read_text())
    excluded=set(cfg['outer_wells']['4'])
    paths=[p for p in sorted((ROOT.parent/'data/train').glob('*.txt')) if p.stem not in excluded]
    assert len(paths)==64
    started=time.perf_counter(); rows=0; gaps=[];miss=np.zeros(9,np.int64);samples=[];ranges=[]
    for path in paths:
        # Labels are never read into the feature computation or analysis arrays.
        frame=read_curve(path)[['DEPTH',*CURVES]]
        side=directional_features(frame)
        rows+=len(frame);gaps.append(np.diff(frame['DEPTH'].to_numpy(np.float64)))
        miss+=frame[list(CURVES)].isna().sum().to_numpy()
        take=np.linspace(0,len(frame)-1,min(512,len(frame)),dtype=int)
        samples.append(side[take]);ranges.append(float(frame.DEPTH.iloc[-1]-frame.DEPTH.iloc[0]))
    side=np.concatenate(samples);gap=np.concatenate(gaps)
    quantiles=[0,.01,.25,.5,.75,.99,1]
    stats={name:{'quantiles':np.quantile(side[:,j],quantiles).tolist(),'std':float(side[:,j].std()),'zero_fraction':float((side[:,j]==0).mean())} for j,name in enumerate(FEATURE_NAMES)}
    schema=json.loads((ROOT/'outputs/pure_dl_paired_adaptive_dilation80_sequence/feature_contract.json').read_text())['whitelist']
    categories={}
    for name in schema:
        if name in CURVES:cat='raw_curves'
        elif name.endswith('_MISSING'):cat='missing_flags'
        elif name=='DEPTH_REL':cat='relative_depth'
        elif '_WELL_' in name:cat='well_relative'
        elif any(t in name for t in ('_ROLL_','_RESID_','_SLOPE_','_DIFF','_TREND')) and any(name.startswith(c+'_') for c in CURVES) and name not in ('SP_GR_DIFF','DEN_AC_DIFF','CNL_DENSITY_POROSITY_DIFF'):cat='local_statistics'
        else:cat='transforms_interactions_proxies'
        categories.setdefault(cat,[]).append(name)
    result={'scope':'64 original fold5 training wells, inputs only; no validation feature tuning or target association','training_wells':64,'raw_rows':rows,'sampled_rows_for_feature_statistics':len(side),'quantile_levels':quantiles,'depth_step_quantiles':np.quantile(gap,quantiles).tolist(),'nonpositive_depth_steps':int((gap<=0).sum()),'interval_span_quantiles':np.quantile(ranges,quantiles).tolist(),'missing_fraction_by_curve':dict(zip(CURVES,(miss/rows).tolist())),'original_feature_groups':{k:len(v) for k,v in categories.items()},'original_group_names':categories,'candidate_count':len(FEATURE_NAMES),'candidate_statistics':stats,'feature_dtype':'float32 CPU; float16 if appended for training','added_gpu_input_mib_at730268rows':730268*len(FEATURE_NAMES)*2/2**20,'seconds':time.perf_counter()-started,'training_gain':'UNTESTED; distribution audit is not accuracy evidence'}
    (out/'input_feature_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('candidate_statistics','original_group_names')},indent=2),flush=True)
if __name__=='__main__':main()
