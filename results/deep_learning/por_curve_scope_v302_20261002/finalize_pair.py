import sys,json,hashlib,copy
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/pure_dl'))
import train_invariant as base
STUDY=Path(__file__).parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
write=lambda p,x:p.write_text(json.dumps(x,indent=2),encoding='utf-8')
data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
truth=np.asarray(data['truth'][data['validation']]);dm=base.dmask(truth)[:,0]
reports={a:read(ROOT/f'outputs/pure_dl_por_curve_scope_v302_fold5/{a}/report.json') for a in ('grouped','full')}
strata={}
for a in reports:
    p=truth.copy();p[:,0]=np.load(ROOT/f'outputs/pure_dl_por_curve_scope_v302_fold5/{a}/channel_pred.npy')
    strata[a]={k:base.metric(truth[m],p[m])['POR'] for k,m in [('default',dm),('continuous',~dm)]}
paired=dict(full_minus_grouped=reports['full']['target_score']-reports['grouped']['target_score'],gain_vs_290={a:r['target_score']-78.0235265552551 for a,r in reports.items()},strata=strata,reports=reports,screening_fold=5,complete_five_folds=False,single_seed_only=True,attention_backward_warn_only_nondeterminism=True)
write(STUDY/'PAIRED_FINAL.json',paired)
recipe=copy.deepcopy(read(ROOT/'results/deep_learning/sw_raw_well_condition_v297_20261002/CURRENT_EIGHT_STAGE_RECIPE.json'))
pred=np.column_stack((np.load(ROOT/'outputs/pure_dl_por_curve_scope_v302_fold5/full/channel_pred.npy'),np.load(ROOT/'outputs/pure_dl_perm_sparse_reacquisition_v291_fold5/point/channel_pred.npy'),np.load(ROOT/'outputs/pure_dl_sw_raw_well_condition_v297_fold5/well/channel_pred.npy')))
assert pred.shape==(144665,3) and np.isfinite(pred).all()
scores=base.metric(truth,pred)
assert abs(scores['POR']-reports['full']['target_score'])<1e-9
np.save(STUDY/'optional_eight_stage_pred.npy',pred)
cp=ROOT/'outputs/pure_dl_por_curve_scope_v302_fold5/full/model_best.pt'
recipe.update(scores=scores,stages=['SOURCE','POR113','POR151','PERM154','SW104','POR302full','PERM291point','SW297well'],strict_808_qualified=scores['TOTAL']>80.8,gain_vs_previous_optional8=scores['TOTAL']-80.69351272227345,gain_vs_default_five=scores['TOTAL']-80.61247541799393,pred_sha256=hashlib.sha256((STUDY/'optional_eight_stage_pred.npy').read_bytes()).hexdigest(),verification='Actual144665 composition; each trained own-best RAM replay/frozen exact. Not independent diskreload.',single_seed_only=True)
recipe['heads']['POR']=dict(checkpoint=str(cp),sha256=hashlib.sha256(cp.read_bytes()).hexdigest())
for k in ('matched_well_minus_reference','matched_continuous_delta','paired_reference_final_pending'):recipe.pop(k,None)
write(STUDY/'CURRENT_EIGHT_STAGE_RECIPE.json',recipe)
timing=read(STUDY/'BATCH_TIMING.json')
ends={a:(ROOT/f'outputs/pure_dl_por_curve_scope_v302_fold5/{a}/report.json').stat().st_mtime for a in reports}
timing.update(terminal=True,end_epochs=ends,individual_seconds={a:r['seconds'] for a,r in reports.items()},paired_same_work_wall_seconds=max(ends.values())-min(x['start_epoch'] for x in timing['starts']),interpretation='Concurrent two-arm wall includes load/initial evaluation/train/own-best restore and report, excludes CPU record/code prep; no serial speed ranking.')
write(STUDY/'BATCH_TIMING.json',timing)
note=f"\nFinal v302: grouped POR{reports['grouped']['target_score']:.12f}, full POR{scores['POR']:.12f}; matched +{paired['full_minus_grouped']:.12f}, full versus290 +{paired['gain_vs_290']['full']:.12f}. Actual eight-stage TOTAL{scores['TOTAL']:.12f}, versus297optional8 +{recipe['gain_vs_previous_optional8']:.12f}; one added stage per head, no chained290weights. Single seed/warn-only attention nondeterminism: small gain, not robust or fullOOF proof. Retain full/grouped positive controls; do not scan connectivity/width on this small result. Strict808 false, no remaining folds. Pair wall{timing['paired_same_work_wall_seconds']:.3f}s; individual grouped{reports['grouped']['seconds']:.3f}s/full{reports['full']['seconds']:.3f}s.\n"
for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
    with p.open('a',encoding='utf-8') as f:f.write(note)
print(json.dumps(dict(scores=scores,paired=paired['full_minus_grouped'],timing=timing['paired_same_work_wall_seconds'])))
