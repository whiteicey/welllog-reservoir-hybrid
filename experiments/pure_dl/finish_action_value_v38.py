"""Restore improved channel heads, score fixed assembly, preserve compact evidence."""
from datetime import datetime
from types import SimpleNamespace
import gc,hashlib,json,time
import numpy as np
import torch
from action_value_v38 import ROOT,STUDY,PREVIOUS,MODES,TARGETS,Network,Evaluator,load_data,reference_prediction
from run_retrieval_smooth_v36 import jobs,append
from finish_all_heads_v26 import diagnostics,native
from train_dual8222_heads80 import atomic_json,atomic_array
from train import metric

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def main():
    if jobs():raise RuntimeError('Training still active')
    start=time.perf_counter();label_check=read(ROOT/'outputs/pure_dl_action_value_v38_POR_fixed_fold5/utility_label_verification.json')
    assert label_check['max_absolute_error']<1e-4 and not label_check['validation_labels_used']
    # Final targeted verification of the other inverse after the POR defect.
    from train_dual8222_heads80 import SOURCE
    from action_value_v38 import Objective
    from train_invariant import kernel_matrix,BOUNDS,DEFAULTS
    cached=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
    master=torch.load(SOURCE,map_location='cpu',weights_only=False)
    ct=np.asarray(master['centers'][2]);actions=np.unique(np.r_[DEFAULTS['SW'],np.clip(np.linspace(ct[0],ct[-1],800),*BOUNDS['SW'])])
    stub=SimpleNamespace(c=2,target='SW',scale=float(master['regression_scale'][2]),median=float(master['regression_median'][2]),actions=torch.tensor(actions,dtype=torch.float32))
    raw=cached['train_raw'][torch.linspace(0,len(cached['train_raw'])-1,2048).long()]
    rows=cached['raw_to_valid'][raw.numpy()]
    actual=Objective(stub).labels(cached['labels'][raw],cached['default'][raw],torch.ones(len(raw),dtype=torch.bool)).numpy()
    sw_error=float(np.abs(actual-kernel_matrix(actions,cached['truth'][rows,2],'SW').T).max())
    assert sw_error<1e-4
    label_checks=dict(POR=label_check,SW=dict(training_points=len(raw),max_absolute_error=sw_error,inverse='exp',validation_labels_used=False))
    del cached,master;gc.collect()
    old=read(PREVIOUS/'CURRENT_RECIPE.json');reference=reference_prediction()
    with np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/oof.npz') as z:
        truth=z['truth'][z['fold']==4].copy();wells=z['well'][z['fold']==4].copy()
    assert truth.shape==reference.shape==(144665,3)
    refs=metric(truth,reference);assert all(abs(refs[k]-v)<1e-9 for k,v in old['scores'].items())
    combined=reference.copy();records=[];chosen=[];contrasts=[];timings=[];choices=[dict(x) for x in old['choices']]
    for t in ('POR','SW'):
        c=TARGETS.index(t);out=ROOT/('outputs/pure_dl_action_value_v38_POR_fixed_fold5' if t=='POR' else 'outputs/pure_dl_action_value_v38_SW_fold5')
        cfg=read(out/'config.json');r=read(out/'fold_5_report.json');own=[]
        for name,digest in cfg['source_hashes'].items():assert hashlib.sha256((ROOT/'experiments/pure_dl'/name).read_bytes()).hexdigest()==digest,name
        for m in MODES:
            arm=out/m;rep=read(arm/'fold_5_report.json');p=np.load(arm/'channel_pred.npy')
            diag=diagnostics(truth,reference,p,wells,t)
            assert abs(diag['scores'][t]-rep['target_score'])<1e-9
            assert rep['all144665_selected_predictions_restored_exactly'] and rep['all_frozen_state_unchanged']
            if rep['best_epoch']==0:assert np.array_equal(p,reference[:,c])
            row=dict(target=t,mode=m,report=rep,diagnostics=diag);records.append(row);own.append((rep['target_score'],m,p,rep))
        best=max(own,key=lambda x:x[0]);adopt=best[0]>refs[t]+1e-9
        ctrl=reference.copy();ctrl[:,c]=own[0][2]
        contrasts.append(dict(target=t,local_minus_point=diagnostics(truth,ctrl,own[1][2],wells,t),
                              updated_only_delta=own[1][3]['best_after_update']['target_score']-own[0][3]['best_after_update']['target_score']))
        if adopt:
            combined[:,c]=best[2];chosen.append(dict(target=t,mode=best[1],checkpoint=best[3]['checkpoint'],best_epoch=best[3]['best_epoch'],target_score=best[0]))
        launch=read(out/'launch.json')
        timings.append(dict(target=t,start=launch['started_at'],end=r['finished_at'],seconds=r['seconds'],peak_allocated_MiB=r['peak_allocated_MiB'],
                            median_epoch_seconds=float(np.median([x['seconds'] for x in r['epoch_timing']]))))
    replay=[]
    if chosen:
        torch.set_num_threads(3);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
        data,master=load_data()
        for s in chosen:
            t=s['target'];m=s['mode'];c=TARGETS.index(t)
            model=Network(t,master,modes=(m,)).cuda();cp=torch.load(ROOT/s['checkpoint'],map_location='cpu',weights_only=False)
            model.heads[m].load_state_dict(cp['delta_state'],strict=True)
            p,score,_=Evaluator(data,master,c,reference).many(model,(m,))[m]
            assert np.array_equal(p,combined[:,c]) and abs(score-s['target_score'])<1e-9
            for choice in choices:
                if choice['target']==t:
                    choice.update(s,key='v38_'+t+'_'+m,architecture='frozen_v31_direct_action_utility_readout',model_module='action_value_v38_por_fixed' if t=='POR' else 'action_value_v38',
                                  parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),
                                  builder='load_data(); Network(target,master,modes=(mode,)); heads[mode].load_state_dict(delta_state)',additional_parameters=sum(p.numel() for p in model.heads[m].parameters()))
            replay.append(t);model.close();del model,cp;gc.collect();torch.cuda.empty_cache()
        del data,master
    scores=metric(truth,combined)
    result=native(dict(reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],
                       candidates=records,selections=chosen,contrasts=contrasts,full_disk_replays=replay,utility_label_verification=label_checks,invalid_attempt_excluded='pure_dl_action_value_v38_POR_fold5',
                       actual_fixed_three_column_scoring=True,rows=144665,screening_fold=5,strict805_qualified=scores['TOTAL']>80.5,
                       complete_five_folds=False,remaining_folds_started=False,old_deliveries_unchanged=True,finished_at=datetime.now().astimezone().isoformat()))
    atomic_array(STUDY/'assembled_pred.npy',combined);atomic_json(STUDY/'assembled_report.json',result)
    recipe=dict(old);recipe.update(scores=native(scores),choices=choices,reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
    atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    invalid_out=ROOT/'outputs/pure_dl_action_value_v38_POR_fold5'
    invalid_start=read(invalid_out/'launch.json')['started_at'];invalid_end=read(invalid_out/'failure.json')['observed_at']
    timing=dict(targets=timings,invalid_interval_note='Upper bound ending at invalidation record; not pure GPU compute time',invalid_attempt_launch_to_invalidation_record_seconds=(datetime.fromisoformat(invalid_end)-datetime.fromisoformat(invalid_start)).total_seconds(),first_launch_to_last_completion_seconds=(max(datetime.fromisoformat(x['end']) for x in timings)-min(datetime.fromisoformat(x['start']) for x in timings)).total_seconds(),
                finalization_seconds=time.perf_counter()-start,scope='Original POR overlapped SW, then corrected POR started after SW completed. Max2 processes. Each process has two independent heads sharing frozen forward. Per-target times include loading/warmup/evaluation. Overall interval also includes correction/staging gaps; initial code authoring excluded.')
    atomic_json(STUDY/'timing.json',timing)
    lines=['# v38 Direct official action-value readouts','',f"Fixed original fifth TOTAL {refs['TOTAL']:.12f} -> {scores['TOTAL']:.12f}, delta {scores['TOTAL']-refs['TOTAL']:+.12f}.",'',
           '|Target|Context|Best after update|Selected with parent fallback|Best/stop epoch|','|---|---|---:|---:|---|']
    for x in records:
        r=x['report'];lines.append(f"|{x['target']}|{x['mode']}|{r['best_after_update']['target_score']:.9f}|{r['target_score']:.9f}|{r['best_epoch']}/{r['stopped_epoch']}|")
    lines+=['','## Scope, execution correction and verification',
            'Same initialized parameters, original input/hidden features/actions/score; identity vs five-point masked average at the hidden readout. All parent predictors frozen. New heads regress official action utility with uniform squared error, not a likelihood or soft-action objective.',
            'The first POR run incorrectly inverted log1p with exp. It was stopped after epoch4; its scores are excluded and no weights were reused. Corrected POR restarted from the original parent, uses expm1, and checked 2048 training labels against the official kernel (maximum difference below1e-4). SW log/exp was correct and finished unchanged.',
            'Selected-head full disk reconstruction: '+str(replay)+'. Fixed three-column predictions scored on144665 original fifth rows.',
            '', '## Mechanism contrasts and scope of conclusions']
    for x in contrasts:
        d=x['local_minus_point'];lines.append(f"{x['target']} local-point selected channel difference {d['channel_delta']:+.12f}; post-update-only difference {x['updated_only_delta']:+.12f}.")
    lines+=['Adopted: '+json.dumps(chosen,ensure_ascii=False),
            'Epoch0 fallback is not training improvement. Positive comparisons are development-fold evidence, not independent validation. Both valid targets failed to improve the parent; close this direct-value readout setting without an additional width/context/loss sweep.',
            '', '## Time and outstanding objective',
            'Valid training interval wall time '+str(timing['first_launch_to_last_completion_seconds'])+' seconds; per-target times, peak memory and invalid-attempt overhead are in timing.json.',
            'Full original5foldOOF>82 is unachieved. No other folds, old tasks or delivery replacements were started.']
    txt='\n'.join(lines)+'\n';(STUDY/'FINAL.md').write_text(txt,encoding='utf-8')
    append(STUDY/'README.md','## 最终报告','## 最终报告\n见[FINAL.md](FINAL.md)，'+f"TOTAL={scores['TOTAL']:.12f}。")
    marker='## v38直接动作奖励对照完成';msg=marker+f"\n四项完成，TOTAL={scores['TOTAL']:.12f}，相对80.336513增加{scores['TOTAL']-refs['TOTAL']:+.12f}；采用{[(x['target'],x['mode']) for x in chosen]}。完整记录action_value_v38_20260929/FINAL.md；错误POR初次运行被剔除，无其他折或交付改动。"
    for p in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md'):append(p,marker,msg)
    print(json.dumps(dict(scores=scores,chosen=chosen,contrasts=[dict(target=x['target'],delta=x['local_minus_point']['channel_delta'],updated_delta=x['updated_only_delta']) for x in contrasts],timing=timing)),flush=True)

if __name__=='__main__':main()
