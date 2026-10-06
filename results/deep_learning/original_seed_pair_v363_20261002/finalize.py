"""Score the fixed probability mixture after independent disk-reload equality."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pure_dl'))
from train import metric,dmask
OWN=ROOT/'outputs/pure_dl_complete_seeds_v362_fold5/seed_20261010'
ref=json.loads((ROOT/'results/deep_learning/sw_dynamic_cross_spectrum_v351_20261002/CURRENT_EIGHT_STAGE_RECIPE.json').read_text())
new=json.loads((OWN/'COMPLETE_MODEL.json').read_text())
reports={t:json.loads((STUDY/(t+'_RESULT.json')).read_text()) for t in ('POR','PERM','SW')}
assert all(v['old_disk_replay_exact'] and v['new_disk_replay_exact'] and v['coverage_exactly_once'] for v in reports.values())
pred=np.column_stack([np.load(STUDY/(t+'_prediction.npy')) for t in ('POR','PERM','SW')])
data=torch.load(OWN/'SOURCE/prepared_data.pt',map_location='cpu',weights_only=False)
y=np.asarray(data['truth'][data['validation']]);del data
assert pred.shape==y.shape==(144665,3) and np.isfinite(pred).all()
scores=metric(y,pred)
assert all(abs(scores[t]-reports[t]['score'])<1e-9 for t in reports)
mask=dmask(y)
strata={t:{label:dict(rows=int(m.sum()),score=metric(y[m],pred[m])[t]) for label,m in [('DEFAULT',mask[:,c]),('CONT',~mask[:,c])]} for c,t in enumerate(('POR','PERM','SW'))}
out=STUDY/'equal_probability_mixture_pred.npy';np.save(out,pred)
point_path=ROOT/'results/deep_learning/complete_seed_compare_v362_20261002/ORIGINAL_SEED20261010_EQUAL_PREDICTION_BLEND.json'
point=json.loads(point_path.read_text())
result=dict(method='Equal1:1 native predictive distribution mixture with original fixed official utility decoder',
            original_scores=ref['scores'],new_seed=20261010,new_seed_scores=new['scores'],scores=scores,
            delta_vs_original=scores['TOTAL']-ref['scores']['TOTAL'],
            delta_vs_new_seed=scores['TOTAL']-new['scores']['TOTAL'],strata=strata,
            arithmetic_prediction_average_scores=point['scores'],weights=[.5,.5],weight_search=False,
            validation_rows=144665,screening_fold=5,independent_disk_reload=True,
            singleton_predictions_exactly_reproduced=True,extra_training_stages=0,
            independently_trained_models=2,stages_per_model=8,total_preexisting_training_stages=16,
            sum_channel_evaluation_seconds=sum(v['seconds'] for v in reports.values()),
            max_evaluation_peak_MiB=max(v['peak_MiB'] for v in reports.values()),
            study_wall_seconds=time.time()-(STUDY/'AGENTS.md').stat().st_ctime,
            timing_scope='Channel sequential, concurrent with one existing SOURCE training job. Per-channel includes loading/preparation/warmup/both model inference; study wall also includes implementation/failure correction/gaps/final CPU assembly. Not a solo inference benchmark.',
            strict_808_qualified=scores['TOTAL']>80.8,complete_five_folds=False,remaining_folds_started=False,
            adopted=False,prediction_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),reports=reports,
            failure_record='First evaluator failed before inference because a NumPy mapping used torch.equal. Original failed source retained; r1 accepts NumPy or Tensor according to actual type.')
(STUDY/'FINAL.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
text=(f'\nTerminal fixed1:1 distribution mixture: {json.dumps(scores)}; '
      f'vs original {result["delta_vs_original"]:+.12f}, vsnew {result["delta_vs_new_seed"]:+.12f}. '
      f'Both models independently reloaded/native predictions exact on all144665 rows. CONT={json.dumps({t:v["CONT"]["score"] for t,v in strata.items()})}. '
      f'Channel evaluations sum{result["sum_channel_evaluation_seconds"]:.3f}s/peak{result["max_evaluation_peak_MiB"]:.3f}MiB; '
      'sequential channels concurrently with existing SOURCE32596, no training interruption/max2. '
      'No weight grid or followup launch; original retained/notadopted. Two models8stages each/16historical phases/noaddedtraining. '
      'Five-seed study continues; no otherfolds/fullOOF claim. Final artifact FINAL.json.\n')
for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
    with p.open('a',encoding='utf-8') as h:h.write(text)
with (ROOT/'EXPLORATIONS.md').open('a',encoding='utf-8') as h:
    h.write('\n- [V363 user-requested original plus new complete seed equal prediction/distribution fusion](results/deep_learning/original_seed_pair_v363_20261002/README.md): fixed fifth, no training, terminal result in FINAL.json.\n')
print(json.dumps({k:v for k,v in result.items() if k!='reports'},indent=2))
