"""User-requested per-head fixed weights, actual cached-native assembly, CPU only."""
from pathlib import Path
import csv,hashlib,json,sys,time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).resolve().parent
GRID=ROOT/'results/deep_learning/original_seed_weight_grid_v364_20261002'
sys.path.insert(0,str(ROOT/'experiments/pure_dl'))
from train import metric,dmask
started=time.perf_counter();targets=('POR','PERM','SW');chosen={};columns=[]
for target in targets:
    report=json.loads((GRID/(target+'_RESULT.json')).read_text())
    assert report['endpoints_and_half_exact'] and report['old_disk_replay_exact'] and report['new_disk_replay_exact']
    values=report['scores_by_weight']
    i=max(range(len(values)),key=lambda k:(values[k]['score'],values[k]['original_weight']))
    chosen[target]=dict(values[i],selected_grid_index=i)
    with np.load(GRID/(target+'_grid_actions.npz')) as z:
        assert z['indices'].shape==(144665,101)
        assert abs(z['weights'][i]-values[i]['original_weight'])<1e-12
        columns.append(z['actions'][z['indices'][:,i]])
pred=np.column_stack(columns)
data=torch.load(ROOT/'outputs/pure_dl_complete_seeds_v362_fold5/seed_20261010/SOURCE/prepared_data.pt',map_location='cpu',weights_only=False)
y=np.asarray(data['truth'][data['validation']]);del data
assert pred.shape==y.shape==(144665,3) and np.isfinite(pred).all()
scores=metric(y,pred)
assert all(abs(scores[t]-chosen[t]['score'])<1e-9 for t in targets)
mask=dmask(y)
strata={t:{label:dict(rows=int(m.sum()),score=metric(y[m],pred[m])[t]) for label,m in [('DEFAULT',mask[:,c]),('CONT',~mask[:,c])]} for c,t in enumerate(targets)}
path=STUDY/'per_head_weight_prediction.npy';np.save(path,pred)
common=json.loads((GRID/'FINAL.json').read_text())
ref=json.loads((ROOT/'results/deep_learning/sw_dynamic_cross_spectrum_v351_20261002/CURRENT_EIGHT_STAGE_RECIPE.json').read_text())
result=dict(selected_weights=chosen,scores=scores,strata=strata,
            delta_vs_original=scores['TOTAL']-ref['scores']['TOTAL'],
            delta_vs_common_81_19=scores['TOTAL']-common['scores']['TOTAL'],
            original_scores=ref['scores'],common_weight_scores=common['scores'],
            continuous_delta_vs_original={t:strata[t]['CONT']['score']-ref[t+'_continuous'] for t in targets},
            grid='0:0.01:1',selection='Per-target official whole score, one fixed ratio for every row; exact ties prefer greater original weight',
            diagnostic_limits='Each fixed head ratio selected on same fifth; maximum within tested grid, not independent generalization/fullOOF. No CONT-specific checkpoint or per-row choice.',
            validation_rows=144665,screening_fold=5,cached_native_grid_reused=True,
            independent_disk_reload_already_verified=True,extra_training_stages=0,extra_GPU_evaluations=0,
            models=2,stages_per_model=8,total_preexisting_training_stages=16,
            seconds_cpu=time.perf_counter()-started,strict_808_qualified=scores['TOTAL']>80.8,
            complete_five_folds=False,remaining_folds_started=False,adopted_as_single_eight_stage_model=False,
            prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(STUDY/'FINAL.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
with (STUDY/'SELECTED_WEIGHTS.csv').open('w',encoding='utf-8',newline='') as h:
    writer=csv.DictWriter(h,fieldnames=('target','original_weight','new_seed_weight','score'));writer.writeheader()
    writer.writerows(dict(target=t,**{k:v for k,v in chosen[t].items() if k!='selected_grid_index'}) for t in targets)
note=(f'\nTerminal userauthorized per-head fixed ratios {json.dumps(chosen)}. '
      f'ActualTOTAL{scores["TOTAL"]:.12f}/originaldelta{result["delta_vs_original"]:+.12f}/common81:19delta{result["delta_vs_common_81_19"]:+.12f}. '
      f'CONTsame selectedratios {json.dumps({t:v["CONT"]["score"] for t,v in strata.items()})}; '
      f'CONTdeltaoriginal {json.dumps(result["continuous_delta_vs_original"])}. '
      f'CPU{result["seconds_cpu"]:.3f}s/nativeverifiedcachedgrids/noextraGPU/training. '
      'Same-fifth tuned maximum0.01grid/fixedperchannel ratios, notroworacle/CONToracle/generalization/OOF. '
      'Two8stage models/16prior phases/noadoption as one8stage model/nootherfolds. Existingfive-seedtraining continues.\n')
for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
    with p.open('a',encoding='utf-8') as h:h.write(note)
with (ROOT/'EXPLORATIONS.md').open('a',encoding='utf-8') as h:
    h.write('\n- [V365 user-requested fixed per-head seed distribution weights](results/deep_learning/per_head_seed_weights_v365_20261002/README.md): cached0.01grid, actual three-head prediction, CPU only, FINAL.json.\n')
print(json.dumps(result,indent=2))
