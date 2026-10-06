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
truth=np.asarray(data['truth'][data['validation']]);dm=base.dmask(truth)[:,1]
reports={};strata={};learned={};ends={}
for arm in ('full','grouped'):
    out=ROOT/f'outputs/pure_dl_perm_curve_scope_v307_fold5/{arm}'
    report=read(out/'report.json');assert report['predictions_reproduced_exactly'] and report['frozen_parameters_exact']
    assert (STUDY/f'{arm}_assembled.json').exists()
    reports[arm]=report;ends[arm]=(out/'report.json').stat().st_mtime
    p=truth.copy();p[:,1]=np.load(out/'channel_pred.npy')
    strata[arm]={k:float(base.metric(truth[m],p[m])['PERM']) for k,m in [('default',dm),('continuous',~dm)]}
    best=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False)['state_dict']
    learned[arm]={n:float(t.float().norm()) for n,t in best.items() if n.startswith('curve_interaction.bridge.')}
delta=float(reports['full']['target_score']-reports['grouped']['target_score'])
selected=max(reports,key=lambda a:reports[a]['target_score']);adopted=reports[selected]['target_score']>89.5155547932572
timing=read(STUDY/'BATCH_TIMING.json');timing.update(terminal=True,end_epochs=ends,individual_seconds={a:r['seconds'] for a,r in reports.items()},same_work_pair_wall_seconds=max(ends.values())-min(x['start_epoch'] for x in timing['starts']),interpretation='Two arms concurrent with different starts; includes data/initial/train/best restore. Codeprep and CPUrecord excluded; no serialspeedranking.')
final=dict(reports=reports,strata=strata,full_minus_grouped=delta,whole_best_module_norms=learned,selected_arm=selected,adopted_absolute=bool(adopted),capacity55490=True,only_pair_variable='grouped versusfull tokenconnectivity',same_initialization=True,not_same_effective_function_family=True,screening_fold=5,complete_five_folds=False,remaining_folds_started=False,timing=timing)
write(STUDY/'PAIRED_FINAL.json',final);write(STUDY/'BATCH_TIMING.json',timing)
if adopted:
    recipe=copy.deepcopy(read(ROOT/'results/deep_learning/por_curve_scope_v302_20261002/CURRENT_EIGHT_STAGE_RECIPE.json'))
    out=ROOT/f'outputs/pure_dl_perm_curve_scope_v307_fold5/{selected}'
    pred=np.column_stack((np.load(ROOT/'outputs/pure_dl_por_curve_scope_v302_fold5/full/channel_pred.npy'),np.load(out/'channel_pred.npy'),np.load(ROOT/'outputs/pure_dl_sw_raw_well_condition_v297_fold5/well/channel_pred.npy')))
    assert pred.shape==(144665,3) and np.isfinite(pred).all()
    scores=base.metric(truth,pred);assert abs(scores['PERM']-reports[selected]['target_score'])<1e-9
    np.save(STUDY/'optional_eight_stage_pred.npy',pred)
    recipe.update(scores=scores,stages=['SOURCE','POR113','POR151','PERM154','SW104','POR302full',f'PERM307{selected}','SW297well'],strict_808_qualified=bool(scores['TOTAL']>80.8),gain_vs_previous_optional8=float(scores['TOTAL']-80.69562059394032),gain_vs_default_five=float(scores['TOTAL']-80.61247541799393),pred_sha256=hashlib.sha256((STUDY/'optional_eight_stage_pred.npy').read_bytes()).hexdigest())
    recipe.pop('attention_backward_warn_only_nondeterminism',None)
    recipe['heads']['PERM']=dict(checkpoint=str(out/'model_best.pt'),sha256=hashlib.sha256((out/'model_best.pt').read_bytes()).hexdigest())
    write(STUDY/'CURRENT_EIGHT_STAGE_RECIPE.json',recipe)
note=f"\nFinal307 full{reports['full']['target_score']:.12f}/grouped{reports['grouped']['target_score']:.12f}; paired{delta:+.12f}; absoluteadoption={bool(adopted)}, selected{selected}. Same55490params/init/initialfunction, only tokenconnectivity differs; selectors point in both. No oldrepairweights, onePERMaddedphase, no independentdiskreload/fullOOF/otherfolds. Pairwall{timing['same_work_pair_wall_seconds']:.3f}s, individuals{timing['individual_seconds']}. If absolute belowretained291 close thisplacement/head/widthgrid; localgain recorded separately. Bestcheckpoint bridgenorms recorded to distinguish zero-init function from trained interaction.\n"
for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
    with p.open('a',encoding='utf-8') as f:f.write(note)
print(json.dumps(final))
