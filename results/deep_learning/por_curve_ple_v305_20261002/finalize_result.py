"""Terminal CPU evidence only; no early stratum selection or GPU replay."""
import sys,json,copy,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).parent
sys.path.insert(0,str(ROOT/'experiments/pure_dl'))
import train_invariant as base
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
write=lambda p,v:p.write_text(json.dumps(v,indent=2),encoding='utf-8')
out=ROOT/'outputs/pure_dl_por_curve_ple_v305_fold5/ple'
report=read(out/'report.json')
assert report['predictions_reproduced_exactly'] and report['frozen_parameters_exact']
assert (STUDY/'ple_assembled.json').exists()
data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
truth=np.asarray(data['truth'][data['validation']]);default=base.dmask(truth)[:,0]
strata={}
for name,path in [('POR290',ROOT/'outputs/pure_dl_por_direct_share_v290_fold5/raw/channel_pred.npy'),('POR302',ROOT/'outputs/pure_dl_por_curve_scope_v302_fold5/full/channel_pred.npy'),('POR305',out/'channel_pred.npy')]:
    p=truth.copy();p[:,0]=np.load(path)
    strata[name]=dict(POR=base.metric(truth,p)['POR'],default_POR=base.metric(truth[default],p[default])['POR'],continuous_POR=base.metric(truth[~default],p[~default])['POR'])
gain=float(strata['POR305']['POR']-strata['POR302']['POR']);adopted=bool(gain>0)
timing=read(STUDY/'BATCH_TIMING.json');timing.update(terminal=True,individual_seconds=report['seconds'],observed_launch_to_report_seconds=(out/'report.json').stat().st_mtime-timing['starts'][0]['start_epoch'],peak_MiB=report['peak_MiB'],concurrency='Brief initial overlap with304 finalepoch, then singleGPU. Not historical302 speedranking.')
final=dict(report=report,strata=strata,gain_vs302=gain,gain_vs290=strata['POR305']['POR']-strata['POR290']['POR'],adopted=adopted,capacity_difference224=True,single_seed=True,complete_five_folds=False,remaining_folds_started=False,timing=timing)
write(STUDY/'BATCH_TIMING.json',timing);write(STUDY/'FINAL.json',final)
if adopted:
    recipe=copy.deepcopy(read(ROOT/'results/deep_learning/por_curve_scope_v302_20261002/CURRENT_EIGHT_STAGE_RECIPE.json'))
    pred=np.column_stack((np.load(out/'channel_pred.npy'),np.load(ROOT/'outputs/pure_dl_perm_sparse_reacquisition_v291_fold5/point/channel_pred.npy'),np.load(ROOT/'outputs/pure_dl_sw_raw_well_condition_v297_fold5/well/channel_pred.npy')))
    assert pred.shape==(144665,3) and np.isfinite(pred).all()
    scores=base.metric(truth,pred);assert abs(scores['POR']-report['target_score'])<1e-9
    np.save(STUDY/'optional_eight_stage_pred.npy',pred)
    recipe.update(scores=scores,stages=['SOURCE','POR113','POR151','PERM154','SW104','POR305ple','PERM291point','SW297well'],gain_vs_previous_optional8=scores['TOTAL']-80.69562059394032,strict_808_qualified=scores['TOTAL']>80.8,pred_sha256=hashlib.sha256((STUDY/'optional_eight_stage_pred.npy').read_bytes()).hexdigest(),gain_vs_default_five=scores['TOTAL']-80.61247541799393)
    recipe['heads']['POR']=dict(checkpoint=str(out/'model_best.pt'),sha256=hashlib.sha256((out/'model_best.pt').read_bytes()).hexdigest())
    write(STUDY/'CURRENT_EIGHT_STAGE_RECIPE.json',recipe)
note=f"\nFinal305 POR{report['target_score']:.12f}, versus302{gain:+.12f}, versus290{final['gain_vs290']:+.12f}; best/stop{report['best_epoch']}/{report['stopped_epoch']}. Adopted={adopted}; +224capacity/nonconcurrent control/single-seed nondeterministicattention means no pure mechanism/robustness claim. Individual{report['seconds']:.3f}s/peak{report['peak_MiB']:.3f}MiB; trainquantilefit included. Whole-best144665RAM/frozenexact, not independentdiskreload, no fullOOF/remainingfolds. If negative close knot/endpoint/LR grid.\n"
for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
    with p.open('a',encoding='utf-8') as f:f.write(note)
print(json.dumps(final))
