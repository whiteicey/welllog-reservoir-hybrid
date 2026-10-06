"""Record the bounded compact experiment and build only its input-side cache."""
from pathlib import Path
from datetime import datetime
import hashlib,json,time
import numpy as np
import pandas as pd
import torch
import train_invariant as base
from compact_five_stage_v100 import STUDY,ROOT,SOURCE
from fullcv_models_v97 import STAGES,root_dir,stage_dir

def main():
    STUDY.mkdir(parents=True,exist_ok=True)
    if (STUDY/'PLAN.json').exists():raise RuntimeError('Preparation already exists')
    start=time.perf_counter()
    am=json.loads((ROOT/'results/deep_learning/feature_structure_v20_20260928/feature_manifest.json').read_text())
    master=torch.load(SOURCE,map_location='cpu',weights_only=False)
    assert len(am['fit_wells'])==64
    records=base.load_split(ROOT.parent/'data','train');curves=('GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT');arrays=[]
    for _,frame in records:
        cols=[]
        for name in curves:
            s=frame[name].replace([np.inf,-np.inf],np.nan).interpolate(method='linear',limit_direction='both').fillna(am['imputation_medians'][name])
            values=s.to_numpy(np.float64)
            if name in ('GR','RXO','RT'):values=np.log1p(np.maximum(values,0))
            s=pd.Series(values);cols.append(values)
            cols.extend(s.rolling(w,center=True,min_periods=1).mean().to_numpy() for w in (9,51,201))
            cols.extend(s.rolling(w,center=True,min_periods=1).std(ddof=0).fillna(0).to_numpy() for w in (9,51,201))
        arrays.append(np.stack(cols,-1).astype(np.float32))
    x=np.concatenate(arrays);scaler=am['scalers']['amplitude']
    side=np.clip((x-np.asarray(scaler['median'],np.float32))/np.asarray(scaler['scale'],np.float32),-8,8).astype(np.float16)
    np.savez_compressed(STUDY/'amplitude.npz',amplitude=side)
    rows=[]
    for f in range(1,5):
        scores=json.loads((root_dir(f)/f'fold_{f}_report.json').read_text())['scores'].copy()
        for stage,target,*_ in STAGES:
            r=json.loads((stage_dir(f,stage)/'report.json').read_text())
            rows.append(dict(fold=f,stage=stage,target=target,previous_selected_score=scores[target],selected_score=r['target_score'],
                             actual_chain_delta=r['target_score']-scores[target],best_epoch=r['best_epoch'],seconds=r['seconds']))
            scores[target]=r['target_score']
    grouped=[]
    for stage,*_ in STAGES:
        subset=[r for r in rows if r['stage']==stage]
        grouped.append(dict(stage=stage,zero_selected_epochs=sum(r['best_epoch']==0 for r in subset),
                            mean_channel_delta=float(np.mean([r['actual_chain_delta'] for r in subset])),
                            total_seconds=sum(r['seconds'] for r in subset)))
    (STUDY/'stage_contributions.json').write_text(json.dumps(dict(scope='diagnostic across all four completed folds; one new common recipe, no per-fold selection',
        important='actual chain delta uses previous selected predictor, not reset-prior INITIAL scores',grouped=grouped,rows=rows),indent=2),encoding='utf-8')
    plan=dict(created_at=datetime.now().astimezone().isoformat(),total_training_stages=5,stage_order=['original_source_pretrain','combined_por','combined_perm','sw_joint_from_random','sw_density'],
        source_checkpoint=str(SOURCE.relative_to(ROOT)),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),source_best_epoch=int(master['best_epoch']),
        source_training_reused=True,source_stage_counted=True,downstream_trained_weight_reuse=False,
        original_fold=5,seed=20261010,epochs=80,patience=18,minimum_epochs=10,batch_size=64,steps=64,screening_threshold=80.8,
        reference_fold5=80.50419742447689,reference_full_oof=81.89101929722543,amplitude_manifest_sha256=hashlib.sha256((ROOT/'results/deep_learning/feature_structure_v20_20260928/feature_manifest.json').read_bytes()).hexdigest(),
        remaining_folds_started=False,preparation_seconds=time.perf_counter()-start,quality='unchanged original fold5 training-only Q50',
        hypothesis='jointly train useful readout, side-feature and route corrections; train complete final SW architecture directly without repeated reset/warmup chains',
        timing='record source pretraining cost separately; historical source reuse is not a free training stage or an end-to-end speedup')
    (STUDY/'PLAN.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    print('COMPACT_PREPARED',json.dumps(plan),flush=True)

if __name__=='__main__':main()
