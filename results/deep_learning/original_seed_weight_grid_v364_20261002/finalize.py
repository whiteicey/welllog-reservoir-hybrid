"""Assemble the common-weight ranking, same-checkpoint strata and export curve."""
from pathlib import Path
import csv,hashlib,json,sys,time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pure_dl'))
from train import metric,dmask
targets=('POR','PERM','SW')
reports={t:json.loads((STUDY/(t+'_RESULT.json')).read_text()) for t in targets}
weights=np.array(reports['POR']['weight_grid']);assert np.array_equal(weights,np.arange(101)/100)
assert all(v['weight_grid']==weights.tolist() and v['endpoints_and_half_exact'] for v in reports.values())
rows=[]
for i,w in enumerate(weights):
    scores={t:reports[t]['scores_by_weight'][i]['score'] for t in targets}
    scores['TOTAL']=sum(scores[t]*a for t,a in zip(targets,(.3,.35,.35)))
    rows.append(dict(original_weight=float(w),new_seed_weight=float(1-w),**scores,delta_vs_original=scores['TOTAL']-80.70612972323806))
best_index=max(range(len(rows)),key=lambda i:rows[i]['TOTAL']);best=rows[best_index]
coarse=max(rows[::5],key=lambda x:x['TOTAL'])
own=ROOT/'outputs/pure_dl_complete_seeds_v362_fold5/seed_20261010'
data=torch.load(own/'SOURCE/prepared_data.pt',map_location='cpu',weights_only=False)
y=np.asarray(data['truth'][data['validation']]);del data
columns=[]
for t in targets:
    with np.load(STUDY/(t+'_grid_actions.npz')) as z:
        assert np.array_equal(z['weights'],weights)
        assert z['indices'].shape==(144665,101)
        columns.append(z['actions'][z['indices'][:,best_index]])
pred=np.column_stack(columns);scores=metric(y,pred)
assert pred.shape==y.shape==(144665,3) and np.isfinite(pred).all()
assert all(abs(scores[t]-best[t])<1e-9 for t in (*targets,'TOTAL'))
mask=dmask(y)
strata={t:{label:dict(rows=int(m.sum()),score=metric(y[m],pred[m])[t]) for label,m in [('DEFAULT',mask[:,c]),('CONT',~mask[:,c])]} for c,t in enumerate(targets)}
path=STUDY/'best_common_weight_pred.npy';np.save(path,pred)
with (STUDY/'WEIGHT_SCORES.csv').open('w',encoding='utf-8',newline='') as h:
    writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
fig,ax=plt.subplots(figsize=(8,4.4),layout='constrained')
ax.plot(weights,[r['TOTAL'] for r in rows],color='#2463a9',lw=1.8)
ax.axhline(80.70612972323806,color='#687385',ls='--',lw=1,label='Original alone: 80.70613')
ax.scatter([best['original_weight']],[best['TOTAL']],color='#c83737',s=36,zorder=4)
ax.annotate(f"Best w={best['original_weight']:.2f}: {best['TOTAL']:.5f}",xy=(best['original_weight'],best['TOTAL']),xytext=(-125,-26),textcoords='offset points',fontsize=9,arrowprops={'arrowstyle':'-','color':'#c83737'})
ax.set(xlabel='Weight of original model (new seed weight = 1 - w)',ylabel='Fold5 official TOTAL',title='Original + seed 20261010: predictive distribution mixture',xlim=(0,1))
ax.grid(alpha=.2);ax.legend(loc='lower right',fontsize=9)
fig.savefig(STUDY/'weight_curve.png',dpi=180);plt.close(fig)
result=dict(best_common_weight=best,best_coarse_005=coarse,
            top_five=sorted(rows,key=lambda x:x['TOTAL'],reverse=True)[:5],
            original_alone=rows[-1],new_seed_alone=rows[0],equal_half=rows[50],
            scores=scores,strata=strata,weight_grid='0:0.01:1',tested_common_weights=101,
            common_weight_all_heads=True,per_target_weight_selection=False,per_row_choice=False,
            comparison='Maximum only within tested grid; same fifth used for weight selection, not independent generalization/fullOOF.',
            screening_fold=5,validation_rows=144665,independent_disk_reload=True,endpoints_half_replayed_exactly=True,
            extra_training_stages=0,models=2,stages_per_model=8,total_preexisting_training_stages=16,
            sum_channel_evaluation_seconds=sum(v['seconds'] for v in reports.values()),
            maximum_evaluation_peak_MiB=max(v['peak_MiB'] for v in reports.values()),
            total_study_wall_seconds=time.time()-(STUDY/'AGENTS.md').stat().st_ctime,
            timing_scope='Serial target evaluations concurrent with one existing SOURCE job, load/preparation/warmup/inference/grid included. Study wall includes implementation/gaps/CPU assembly/plot, not a solo inference benchmark.',
            strict_808_qualified=scores['TOTAL']>80.8,complete_five_folds=False,remaining_folds_started=False,
            adopted_as_single_eight_stage_model=False,
            prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reports=reports)
(STUDY/'FINAL.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
note=(f'\nTerminal101 common weights: best original{best["original_weight"]:.2f}/new{best["new_seed_weight"]:.2f}, '
      f'TOTAL{best["TOTAL"]:.12f}/originaldelta{best["delta_vs_original"]:+.12f}. '
      f'Coarse0.05 best{coarse["original_weight"]:.2f}/{coarse["TOTAL"]:.12f}; native0/1/0.5 andsame144665coverage exact. '
      f'CONT samegridselectedprediction {json.dumps({t:v["CONT"]["score"] for t,v in strata.items()})}. '
      f'Channel evaluations sum{result["sum_channel_evaluation_seconds"]:.3f}s/peak{result["maximum_evaluation_peak_MiB"]:.3f}MiB, '
      f'studywall{result["total_study_wall_seconds"]:.3f}s inclprep/warmup/grid/CPUassembly/implementation/gaps/plot; '
      'serial targets concurrent with existing SOURCE32596/max2, notsolo benchmark. '
      'Same-fifth validation-tuned maximum within0.01grid only; no finergrid/otherfolds/newtraining/adoption as single8stage model. '
      'Two models8each/16 prior phases. Full table WEIGHT_SCORES.csv, curve weight_curve.png, FINAL.json. Five-seed training continues.\n')
for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
    with p.open('a',encoding='utf-8') as h:h.write(note)
with (ROOT/'EXPLORATIONS.md').open('a',encoding='utf-8') as h:
    h.write('\n- [V364 user-requested original/new seed probability weight grid](results/deep_learning/original_seed_weight_grid_v364_20261002/README.md): common0.01grid, no training, same-fifth tuned diagnostic in FINAL.json.\n')
print(json.dumps({k:v for k,v in result.items() if k!='reports'},indent=2))
