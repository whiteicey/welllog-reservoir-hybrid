"""CPU terminal evidence and optional actual eight-stage composition."""
import sys,json,copy,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).parent
sys.path.insert(0,str(ROOT/'experiments/pure_dl'))
import train_invariant as base
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
write=lambda p,v:p.write_text(json.dumps(v,indent=2),encoding='utf-8')
out=ROOT/'outputs/pure_dl_perm_default_anchor_v309_fold5/anchor'
report=read(out/'report.json');assert report['predictions_reproduced_exactly'] and report['frozen_parameters_exact']
assert (STUDY/'anchor_assembled.json').exists()
data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
truth=np.asarray(data['truth'][data['validation']]);dm=base.dmask(truth)[:,1]
strata={}
for name,path in [('retained291',ROOT/'outputs/pure_dl_perm_sparse_reacquisition_v291_fold5/point/channel_pred.npy'),('control307full',ROOT/'outputs/pure_dl_perm_curve_scope_v307_fold5/full/channel_pred.npy'),('candidate309',out/'channel_pred.npy')]:
    p=truth.copy();p[:,1]=np.load(path)
    strata[name]=dict(PERM=float(base.metric(truth,p)['PERM']),default_PERM=float(base.metric(truth[dm],p[dm])['PERM']),continuous_PERM=float(base.metric(truth[~dm],p[~dm])['PERM']))
delta={k:strata['candidate309'][k]-strata['control307full'][k] for k in strata['candidate309']}
delta_retained={k:strata['candidate309'][k]-strata['retained291'][k] for k in strata['candidate309']}
adopted=delta_retained['PERM']>0
timing=read(STUDY/'BATCH_TIMING.json');timing.update(terminal=True,individual_seconds=report['seconds'],launch_to_report_seconds=(out/'report.json').stat().st_mtime-timing['starts'][0]['start_epoch'],peak_MiB=report['peak_MiB'],interpretation='Individual time includes data/initial/train/best restoration. Actual overlap from launch records; different historical contention, no speedranking.')
final=dict(report=report,strata=strata,gain_vs_control307=delta,gain_vs_retained291=delta_retained,adopted=bool(adopted),same_capacity55490=True,default_auxiliary_CE_coefficient=.05,reference_historical=True,conditional_density_semantic_regularizer=True,complete_five_folds=False,remaining_folds_started=False,timing=timing)
write(STUDY/'FINAL.json',final);write(STUDY/'BATCH_TIMING.json',timing)
if adopted:
    control=ROOT/'results/deep_learning/por_curve_cross_v308_20261002/CURRENT_EIGHT_STAGE_RECIPE.json'
    if not control.exists():control=ROOT/'results/deep_learning/por_curve_scope_v302_20261002/CURRENT_EIGHT_STAGE_RECIPE.json'
    recipe=copy.deepcopy(read(control));previous_total=recipe['scores']['TOTAL']
    pred=np.column_stack((np.load(Path(recipe['heads']['POR']['checkpoint']).parent/'channel_pred.npy'),np.load(out/'channel_pred.npy'),np.load(Path(recipe['heads']['SW']['checkpoint']).parent/'channel_pred.npy')))
    assert pred.shape==(144665,3) and np.isfinite(pred).all()
    scores=base.metric(truth,pred);assert abs(scores['PERM']-report['target_score'])<1e-9
    np.save(STUDY/'optional_eight_stage_pred.npy',pred);recipe['stages'][6]='PERM309anchor'
    recipe.update(scores=scores,strict_808_qualified=bool(scores['TOTAL']>80.8),gain_vs_previous_optional8=float(scores['TOTAL']-previous_total),gain_vs_default_five=float(scores['TOTAL']-80.61247541799393),pred_sha256=hashlib.sha256((STUDY/'optional_eight_stage_pred.npy').read_bytes()).hexdigest())
    recipe['heads']['PERM']=dict(checkpoint=str(out/'model_best.pt'),sha256=hashlib.sha256((out/'model_best.pt').read_bytes()).hexdigest())
    write(STUDY/'CURRENT_EIGHT_STAGE_RECIPE.json',recipe)
note=f"\nFinal309 PERM{report['target_score']:.12f}; versus307full{delta}, versusretained291{delta_retained}; adopted={bool(adopted)}. WholePERM-best/stop{report['best_epoch']}/{report['stopped_epoch']},144665RAM/frozenexact, no independentdiskreload or fullOOF. DEFAULT-only true-label regularizer notproperconditionalNLL correction; checkCONTtradeoff, no coeff/edge/decodergrid. Individual{report['seconds']:.3f}s/peak{report['peak_MiB']:.3f}MiB; historicalcontrol notconcurrent/timefairnessnotclaimed. OnePERMaddedphase, currentdelivery intact, nootherfolds.\n"
for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
    with p.open('a',encoding='utf-8') as f:f.write(note)
print(json.dumps(final))
