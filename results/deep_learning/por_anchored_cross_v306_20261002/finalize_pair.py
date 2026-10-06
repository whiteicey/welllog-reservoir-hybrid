"""Terminal CPU pair and actual composition; no validation-gradient selection."""
import json,sys,hashlib,copy
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).parent
sys.path.insert(0,str(ROOT/'experiments/pure_dl'))
import train_invariant as base
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
write=lambda p,v:p.write_text(json.dumps(v,indent=2),encoding='utf-8')
data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
truth=np.asarray(data['truth'][data['validation']]);dm=base.dmask(truth)[:,0]
reports={};strata={};learned={};ends={}
for arm in ('cross','additive'):
    out=ROOT/f'outputs/pure_dl_por_anchored_cross_v306_fold5/{arm}'
    report=read(out/'report.json');assert report['predictions_reproduced_exactly'] and report['frozen_parameters_exact']
    assert (STUDY/f'{arm}_assembled.json').exists()
    reports[arm]=report;ends[arm]=(out/'report.json').stat().st_mtime
    p=truth.copy();p[:,0]=np.load(out/'channel_pred.npy')
    strata[arm]={k:float(base.metric(truth[m],p[m])['POR']) for k,m in [('default',dm),('continuous',~dm)]}
    best=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False)['state_dict']
    learned[arm]={n:float(t.float().norm()) for n,t in best.items() if n.startswith('cross_blocks.') and ('.up.weight' in n or n.endswith('.bias'))}
delta=float(reports['cross']['target_score']-reports['additive']['target_score'])
selected=max(reports,key=lambda a:reports[a]['target_score']);adopted=reports[selected]['target_score']>78.03055279414464
timing=read(STUDY/'BATCH_TIMING.json');timing.update(terminal=True,end_epochs=ends,individual_seconds={a:r['seconds'] for a,r in reports.items()},same_work_pair_wall_seconds=max(ends.values())-min(x['start_epoch'] for x in timing['starts']),interpretation='Two arms concurrent with different starts; includes data/initial/train/best restore. Codeprep and CPUrecord excluded; no serialspeedranking.')
final=dict(reports=reports,strata=strata,cross_minus_additive=delta,whole_best_module_norms=learned,selected_arm=selected,adopted_absolute=bool(adopted),capacity38272=True,only_pair_variable='input-anchor product',same_initialization=True,not_same_effective_function_family=True,screening_fold=5,complete_five_folds=False,remaining_folds_started=False,timing=timing)
write(STUDY/'PAIRED_FINAL.json',final);write(STUDY/'BATCH_TIMING.json',timing)
if adopted:
    recipe=copy.deepcopy(read(ROOT/'results/deep_learning/por_curve_scope_v302_20261002/CURRENT_EIGHT_STAGE_RECIPE.json'))
    out=ROOT/f'outputs/pure_dl_por_anchored_cross_v306_fold5/{selected}'
    pred=np.column_stack((np.load(out/'channel_pred.npy'),np.load(ROOT/'outputs/pure_dl_perm_sparse_reacquisition_v291_fold5/point/channel_pred.npy'),np.load(ROOT/'outputs/pure_dl_sw_raw_well_condition_v297_fold5/well/channel_pred.npy')))
    assert pred.shape==(144665,3) and np.isfinite(pred).all()
    scores=base.metric(truth,pred);assert abs(scores['POR']-reports[selected]['target_score'])<1e-9
    np.save(STUDY/'optional_eight_stage_pred.npy',pred)
    recipe.update(scores=scores,stages=['SOURCE','POR113','POR151','PERM154','SW104',f'POR306{selected}','PERM291point','SW297well'],strict_808_qualified=bool(scores['TOTAL']>80.8),gain_vs_previous_optional8=float(scores['TOTAL']-80.69562059394032),gain_vs_default_five=float(scores['TOTAL']-80.61247541799393),pred_sha256=hashlib.sha256((STUDY/'optional_eight_stage_pred.npy').read_bytes()).hexdigest())
    recipe.pop('attention_backward_warn_only_nondeterminism',None)
    recipe['heads']['POR']=dict(checkpoint=str(out/'model_best.pt'),sha256=hashlib.sha256((out/'model_best.pt').read_bytes()).hexdigest())
    write(STUDY/'CURRENT_EIGHT_STAGE_RECIPE.json',recipe)
note=f"\nFinal306 cross{reports['cross']['target_score']:.12f}/additive{reports['additive']['target_score']:.12f}; paired{delta:+.12f}; absoluteadoption={bool(adopted)}, selected{selected}. Same38272params/initialfunction, effective interaction family differs. No oldrepairweights, onePORaddedphase, no independentdiskreload/fullOOF/otherfolds. Pairwall{timing['same_work_pair_wall_seconds']:.3f}s, individuals{timing['individual_seconds']}. If cross<=additive close rank/layer/scalegrid; localpositivebutabsolutebelowchampion not adopted. Bestcheckpoint blocknorms recorded to distinguish zero-init function from trained interaction.\n"
for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
    with p.open('a',encoding='utf-8') as f:f.write(note)
print(json.dumps(final))
