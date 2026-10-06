"""CPU terminal three-head assembly; never train, restart, or launch folds."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import psutil
import torch
from train import metric,dmask
from train_dual8222_heads80 import atomic_json,atomic_array

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/perm_depth_residual_v369_20261002'
OUT=ROOT/'outputs/pure_dl_perm_depth_residual_v369_fold5'
TARGETS=('POR','PERM','SW')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def assemble(stage):
    tick=time.perf_counter();plan=read(STUDY/'PLAN.json');phase=OUT/stage
    state_file=phase/'runtime_status.json'
    if not state_file.exists():return dict(stage=stage,status='not_started',unscored=True)
    state=read(state_file)
    try:
        cmd=psutil.Process(state['pid']).cmdline()
        if any(Path(arg).name=='train_perm_depth_residual_v369.py' for arg in cmd):
            return dict(stage=stage,status=state['status'],live=True,unscored=True)
    except (psutil.Error,KeyError):pass
    if state['status'] not in ('complete','complete_early_terminated'):
        return dict(stage=stage,status=state['status'],unscored=True)
    report=read(phase/'report.json');cfg=read(phase/'config.json')
    assert report['target']=='PERM' and report['predictions_reproduced_exactly'] and report['frozen_parameters_exact']
    assert report['validation_rows']==144665 and cfg['trainable_parameters']==plan['trainable_parameters']
    result_file=STUDY/(stage+'_assembled.json')
    checkpoint_hash=sha(phase/'model_best.pt');prediction_hash=sha(phase/'channel_pred.npy')
    if result_file.exists():
        cached=read(result_file)
        assert cached['checkpoint_sha256']==checkpoint_hash and cached['PERM_prediction_sha256']==prediction_hash
        return cached
    expected={'PERM':report['target_score']};columns={};fixed_hashes={}
    for target in ('POR','SW'):
        item=plan['fixed_other_heads'][target];checkpoint=Path(item['checkpoint'])
        assert sha(checkpoint)==item['sha256']
        saved=checkpoint.parent/'channel_pred.npy';columns[target]=np.load(saved);fixed_hashes[target]=sha(saved)
        r=read(checkpoint.parent/'report.json');assert r['predictions_reproduced_exactly']
        expected[target]=r['target_score']
    columns['PERM']=np.load(phase/'channel_pred.npy')
    data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
    truth=np.asarray(data['truth'][data['validation']]);del data
    pred=np.column_stack([columns[t] for t in TARGETS]);assert pred.shape==truth.shape==(144665,3) and np.isfinite(pred).all()
    scores=metric(truth,pred);assert all(abs(scores[t]-expected[t])<1e-9 for t in TARGETS)
    mask=dmask(truth);strata={t:{kind:dict(rows=int(m.sum()),score=metric(truth[m],pred[m])[t]) for kind,m in [('DEFAULT',mask[:,c]),('CONT',~mask[:,c])]} for c,t in enumerate(TARGETS)}
    normalized=json.loads(json.dumps(cfg));normalized.pop('stage');normalized['initialization'].pop('temporal_mode')
    result=dict(stage=stage,scores=scores,strata=strata,report=report,matched_configuration=normalized,
                total_training_stages=8,added_head_stages=dict(POR=1,PERM=1,SW=1),fixed_other_heads=plan['fixed_other_heads'],
                checkpoint_sha256=checkpoint_hash,PERM_prediction_sha256=prediction_hash,fixed_other_prediction_sha256=fixed_hashes,
                gain_vs_original_PERM=scores['PERM']-plan['SOTA_target_reference'],gain_vs_original_TOTAL=scores['TOTAL']-plan['SOTA_TOTAL_reference'],
                strict_808_passed=scores['TOTAL']>80.8,validation_rows=144665,screening_fold=5,
                complete_five_folds=False,remaining_folds_started=False,independent_model_disk_reload=False,
                CPU_saved_prediction_readback=True,CPU_seconds=time.perf_counter()-tick,
                full_recipe_wall_seconds=None,timing_scope='Report seconds cover the one replacement phase, not complete8stage work',
                control_limit='Same parameter count/init but point kernel functionally redundant; not matched effective capacity')
    atomic_array(STUDY/(stage+'_assembled_pred.npy'),pred)
    assert np.array_equal(np.load(STUDY/(stage+'_assembled_pred.npy')),pred)
    atomic_json(result_file,result)
    note=f'\nV369 {stage} terminal PERM{scores["PERM"]:.12f}/actual TOTAL{scores["TOTAL"]:.12f}, deltaoriginal{result["gain_vs_original_TOTAL"]:+.12f}, strict808={result["strict_808_passed"]}, 8stages/one PERM addition, all144665 saved-pred assembly/disk readback; no independent model reload/otherfolds/full-recipe speed claim. Matched connectivity gain awaits both candidates.\n'
    for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
        with p.open('a',encoding='utf-8') as h:h.write(note)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('point','local'),required=True);a=ap.parse_args()
    result=assemble(a.stage)
    paths=[STUDY/(x+'_assembled.json') for x in ('point','local')]
    if all(p.exists() for p in paths) and not (STUDY/'MATCHED_COMPARISON.json').exists():
        point,local=[read(p) for p in paths]
        assert point['matched_configuration']==local['matched_configuration']
        paired=dict(point_scores=point['scores'],local_scores=local['scores'],
                    deltas={t:local['scores'][t]-point['scores'][t] for t in TARGETS+('TOTAL',)},
                    point_strata=point['strata'],local_strata=local['strata'],
                    comparison='Same parameter count/architecture/init/budget, connectivity only; point effective kernel redundant',
                    complete_five_folds=False,remaining_folds_started=False,automatic_adoption=False)
        atomic_json(STUDY/'MATCHED_COMPARISON.json',paired)
        note='\nV369 fixed connectivity ablation completed: '+json.dumps(paired)+'. No automatic adoption/fullOOF or fold launch.\n'
        for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
            with p.open('a',encoding='utf-8') as h:h.write(note)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
