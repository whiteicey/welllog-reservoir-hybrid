"""Read-only observed stage gains and exact leaf-deletion recipes."""
from pathlib import Path
import hashlib,itertools,json,sys,time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pure_dl'))
from train import metric,dmask
started=time.perf_counter();targets=('POR','PERM','SW');weights=(.3,.35,.35)
stages=('SOURCE','POR113','POR151','PERM154','SW104','POR341','PERM336','SW351')
old_paths={
    'SOURCE':'pure_dl_paired_local_context_router80_sequence',
    'POR113':'pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy',
    'POR151':'pure_dl_por_ranked_probability_v151_fold5/joint_rps',
    'PERM154':'pure_dl_perm_ranked_probability_v154_fold5/perm_conditional',
    'SW104':'pure_dl_compact_frequency_v104_fold5/local',
    'POR341':'pure_dl_por_coupling_conditional_rps_v341_fold5/conditional',
    'PERM336':'pure_dl_perm_low_rank_cross_v336_fold5/cross',
    'SW351':'pure_dl_sw_dynamic_cross_spectrum_v351_fold5/aligned'}
new_root=ROOT/'outputs/pure_dl_complete_seeds_v362_fold5/seed_20261010'
data=torch.load(new_root/'SOURCE/prepared_data.pt',map_location='cpu',weights_only=False)
truth=np.asarray(data['truth'][data['validation']]);del data
default=dmask(truth)
all_results={}
for label in ('original','new_seed20261010'):
    paths={s:(ROOT/'outputs'/old_paths[s] if label=='original' else new_root/('SW351_fixed' if s=='SW351' else s)) for s in stages}
    reports={s:json.loads((paths[s]/('fold_5_report.json' if s=='SOURCE' else 'report.json')).read_text(encoding='utf-8-sig')) for s in stages}
    source=np.load(paths['SOURCE']/'fold_5_pred.npy')
    assert source.shape==truth.shape==(144665,3)
    assert all(abs(metric(truth,source)[t]-reports['SOURCE']['scores'][t])<1e-9 for t in (*targets,'TOTAL'))
    predictions={s:np.load(paths[s]/'channel_pred.npy') for s in stages[1:]}
    for s,p in predictions.items():
        c=targets.index(reports[s]['target']);test=truth.copy();test[:,c]=p
        assert p.shape==(144665,) and np.isfinite(p).all()
        assert abs(metric(truth,test)[targets[c]]-reports[s]['target_score'])<1e-9
    previous={'POR113':'SOURCE','POR151':'POR113','PERM154':'SOURCE','SW104':'SOURCE','POR341':'POR151','PERM336':'PERM154','SW351':'SW104'}
    table=[]
    for s in stages[1:]:
        r=reports[s];target=r['target'];c=targets.index(target);parent=previous[s]
        before=source[:,c] if parent=='SOURCE' else predictions[parent]
        after=predictions[s];q=truth.copy();q[:,c]=before;a=metric(truth,q)[target]
        q[:,c]=after;b=metric(truth,q)[target]
        strata={}
        for kind,m in [('DEFAULT',default[:,c]),('CONT',~default[:,c])]:
            x=truth[m].copy();x[:,c]=before[m];sa=metric(truth[m],x)[target]
            x[:,c]=after[m];sb=metric(truth[m],x)[target]
            strata[kind]=dict(before=sa,after=sb,delta=sb-sa)
        table.append(dict(stage=s,target=target,predecessor=parent,
                          before=a,after=b,actual_target_gain=b-a,actual_TOTAL_contribution=(b-a)*weights[c],
                          own_initialized_score=r['initial_score'],optimizer_gain=r['against_initial'],
                          best_epoch=r['best_epoch'],stop_epoch=r['stopped_epoch'],
                          seconds_reported=r['seconds'],predictions_identical_to_predecessor=bool(np.array_equal(before,after)),
                          changed_prediction_rows=int(np.count_nonzero(before!=after)),strata=strata))
    terminal=('POR341','PERM336','SW351');prefix=('POR151','PERM154','SW104')
    full=np.column_stack([predictions[s] for s in terminal]);fullscore=metric(truth,full)
    deletions=[]
    for remove_count in range(4):
        for dropped in itertools.combinations(terminal,remove_count):
            selected=[prefix[c] if s in dropped else s for c,s in enumerate(terminal)]
            p=np.column_stack([predictions[s] for s in selected]);score=metric(truth,p)
            deletions.append(dict(dropped=list(dropped),selected_heads=dict(zip(targets,selected)),
                                  training_stages=8-len(dropped),scores=score,
                                  delta_vs_full=score['TOTAL']-fullscore['TOTAL'],
                                  interpretation='Valid leaf deletion: use actual predecessor checkpoint for deleted final head; no descendant of the dropped phase is retained.'))
    all_results[label]=dict(source_scores=reports['SOURCE']['scores'],full_scores=fullscore,stages=table,
                           leaf_deletions=deletions,
                           table_note='Actual gain relative to preceding same-target prediction; optimizer_gain compares own stage initializer. SW104 is cold architecture replacement, not a fine-tune of SOURCE SW.')
old_ablation=ROOT/'results/deep_learning/por_stage_ablation_v201_20261001'
prefix=json.loads((old_ablation/'prefix_ablation.json').read_text())
skip=json.loads((old_ablation/'skip_middle_assembled.json').read_text())
new2=ROOT/'outputs/pure_dl_complete_seeds_v362_fold5/seed_20261011/POR151'
new2_report=new2/'report.json';new2_state=new2/'runtime_status.json'
partial=json.loads((new2_report if new2_report.exists() else new2_state).read_text())
recommendations=[
    'Original recipe: no final phase can currently be removed with zero score loss. Least damaging proven leaf deletion is SW351, yielding seven stages; true score delta is in leaf_deletions.',
    'New first seed: SW351 best0/optimizergain0, leaf deletion is effectively score-neutral (tiny initializer numerical change), so seven-stage recipe is directly supported by saved predictions.',
    'POR151/PERM154 new first seed best0: candidate redundant optimization phases in THIS seed. Do not generalize; original151 and154 positive, and current second-seed151 has positive gain. For intermediate removal, reconstruct an untrained adapter/initializer and verify selected state equivalence, or retrain downstream POR341/PERM336 from the retained earlier source. Existing trained descendants do not establish deletion.',
    'Prior direct skip113 experiment V201 has final POR77.415345964852 vs retained15177.926380760724 (-0.511034795872); this evidence does not support deleting113. Historical patience18 vs skip14 disclosed.',
    'If targeting six stages next, merge PERM154 and336 into one fresh joint encoder/readout phase from SOURCE first (not from trained154); this is untested and requires new matched fifth validation. Alternatively merge SW104/351 jointly from random SW body, also untested. Preserve existing original8 until gains are verified.',
    'Do not remove SOURCE while still loading its trained weights; that retains SOURCE training cost. Removing it requires a separately cold-trained architecture/input-quality recipe.',
    'Historical original stages use18/14 patience, newseeds10->14 plus their ownSOURCE/Q/parent checkpoints. Gain differences cannot be attributed to seed or patience alone; all current five-seed runs remain unchanged.']
result=dict(models=all_results,original_POR_stage_ablation=dict(prefix=prefix,skip_middle=skip),
            second_seed_POR151_snapshot=partial,recommendations=recommendations,
            validation_rows=144665,screening_fold=5,CPU_only=True,extra_training_stages=0,
            analysis_seconds=time.perf_counter()-started,
            timing_limit='Reported historic per-stage seconds reflect each experiment work/concurrency and are not a matched serial wall-time comparison.',
            complete_five_folds=False,remaining_folds_started=False)
(STUDY/'FINAL.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
lines=['# Stage gains and deletion analysis','','## Focus','User asks original versus new complete seed stage gains and valid deletion options. Same original fifth, unchanged actual artifact predictions, CPU only. Separate training gain from predecessor gain, and leaf versus intermediate deletion. Existing five-seed training continues.','','## Explored and verification','AllSOURCE and7head saved predictions scored on actual144665original heldout rows, checked against their original reports. All8 combinations of dropping final341/336/351 use actual predecessor columns; no descendant trained on a deleted leaf is retained. Original V201 prefix/skip113 evidence reused.','','## Gain','Observed tables below; no new training gain claimed. Historical budget/Q/parents differ from newseeds, not seed-only causal attribution.']
lines+=['','| Stage | Original target gain | New first-seed target gain | Original TOTAL contribution | New TOTAL contribution |','|---|---:|---:|---:|---:|']
for a,b in zip(all_results['original']['stages'],all_results['new_seed20261010']['stages']):
    lines.append(f'| {a["stage"]} | {a["actual_target_gain"]:+.9f} | {b["actual_target_gain"]:+.9f} | {a["actual_TOTAL_contribution"]:+.9f} | {b["actual_TOTAL_contribution"]:+.9f} |')
lines+=['','SW104 gain in this table compares replacement to SOURCE SW; its own random-start training gain is2.924637746686 original/2.839013676297 new. SOURCE is the prerequisite, not a claimed incremental stage gain.','','## Not worth continuing / deletion advice']+['- '+v for v in recommendations]
lines+=['','## Timing, limitations and pending',f'CPU analysis {result["analysis_seconds"]:.3f}s includes source preparation read/scoring/assembly; no GPU or training added. Full original OOF>82 and fifth>80.8 still unmet. No deletion of weights or code performed. Any proposed intermediate merge/skip is untrained; finish5seeds then decide a scoped comparison. FINAL.json preserves all per-channel DEFAULT/CONT, leaf scores and exact artifact comparisons.']
(STUDY/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
note='\nV366 original/new firstseed stage gain and leaf-deletion analysis complete, CPU only. OriginalSW351 least damaging tail removal; firstnewSW351 score-neutral within1.4e-7 target/lastbest0. Original151/154 positive vs newfirstbest0; second151alreadypositive, no universal pruning. Original V201 directskip113 lower0.511035POR. Intermediate151/154 removal needs untrained-state equivalence or fresh downstream training, cannot retain learned descendants and eraseancestor cost. Recommended optional next mergePERM154+336 fromSOURCE, untested; original8/newfullseed batch unchanged. Evidence results/deep_learning/stage_gain_analysis_v366_20261002/FINAL.json.\n'
for p in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
    with p.open('a',encoding='utf-8') as h:h.write(note)
with (ROOT/'EXPLORATIONS.md').open('a',encoding='utf-8') as h:
    h.write('\n- [V366 original/new complete seed stage gain and deletion analysis](results/deep_learning/stage_gain_analysis_v366_20261002/README.md): CPU native artifact analysis, leaf deletions verified, intermediate proposals explicitly untrained.\n')
print(json.dumps(dict(stage_gains={k:v['stages'] for k,v in all_results.items()},leaf_deletions={k:v['leaf_deletions'] for k,v in all_results.items()},second_POR151=partial),indent=2))
