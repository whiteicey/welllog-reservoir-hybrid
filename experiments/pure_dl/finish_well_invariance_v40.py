"""Finalize blocked versus reversed training-well supervision on SW."""
from datetime import datetime
import gc,hashlib,json,time
import numpy as np
import torch
from well_invariance_v40 import ROOT,STUDY,PREVIOUS,MODES,Network,Evaluator,load_data,reference_prediction
from run_retrieval_smooth_v36 import jobs,append
from finish_all_heads_v26 import diagnostics,native
from train_dual8222_heads80 import atomic_json,atomic_array
from train import metric

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def main():
    if jobs():raise RuntimeError('Training still active')
    start=time.perf_counter();previous=read(PREVIOUS/'CURRENT_RECIPE.json');reference=reference_prediction()
    with np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/oof.npz') as z:
        truth=z['truth'][z['fold']==4].copy();wells=z['well'][z['fold']==4].copy()
    assert truth.shape==reference.shape==(144665,3)
    refs=metric(truth,reference);assert all(abs(refs[k]-v)<1e-9 for k,v in previous['scores'].items())
    records=[];predictions={};updates={};costs=[]
    for mode in MODES:
        out=ROOT/f'outputs/pure_dl_well_invariance_v40_SW_{mode}_fold5';cfg=read(out/'config.json');rep=read(out/'fold_5_report.json')
        for n,d in cfg['source_hashes'].items():assert hashlib.sha256((ROOT/'experiments/pure_dl'/n).read_bytes()).hexdigest()==d,n
        pred=np.load(out/'channel_pred.npy');updated=np.load(out/'best_updated_pred.npy')
        diag=diagnostics(truth,reference,pred,wells,'SW');up=diagnostics(truth,reference,updated,wells,'SW')
        assert abs(diag['scores']['SW']-rep['target_score'])<1e-9 and abs(up['scores']['SW']-rep['best_after_update']['target_score'])<1e-9
        assert rep['all144665_selected_predictions_restored_exactly'] and rep['all_frozen_state_unchanged']
        if rep['best_epoch']==0:assert np.array_equal(pred,reference[:,2])
        records.append(dict(mode=mode,report=rep,selected=diag,post_update=up));predictions[mode]=pred;updates[mode]=updated
        launch=read(out/'launch.json');costs.append(dict(mode=mode,start=launch['started_at'],end=rep['finished_at'],seconds=rep['seconds'],peak_allocated_MiB=rep['peak_allocated_MiB'],median_epoch_seconds=float(np.median([x['seconds'] for x in rep['epoch_timing']]))))
    warmup={}
    histories={m:read(ROOT/f'outputs/pure_dl_well_invariance_v40_SW_{m}_fold5/validation_history.json') for m in MODES}
    for i in range(3):
        warmup[str(i+1)]=histories['reversed'][i]['target_score']-histories['blocked'][i]['target_score']
    best=max(records,key=lambda x:x['report']['target_score']);mode=best['mode'];adopted=best['report']['target_score']>refs['SW']+1e-9
    combined=reference.copy();choices=[dict(c) for c in previous['choices']];replay=False
    if adopted:
        torch.set_num_threads(3);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
        data,master=load_data();model=Network('SW',master,mode).cuda()
        cp=torch.load(ROOT/best['report']['checkpoint'],map_location='cpu',weights_only=False)
        model.load_trainable_state(cp['trainable_state'])
        pred,score,_=Evaluator(data,master,2,reference).many(model,(mode,))[mode]
        assert np.array_equal(pred,predictions[mode]) and abs(score-best['report']['target_score'])<1e-9
        combined[:,2]=pred;replay=True
        assert len(model.domain_features)==0, 'Domain branch must not run at inference'
        new=dict(target='SW',key='v40_SW_'+mode,mode=mode,architecture='private_SW_norm_with_training_only_well_adversary',
                 model_module='well_invariance_v40',checkpoint=best['report']['checkpoint'],best_epoch=best['report']['best_epoch'],target_score=score,
                 parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),added_inference_parameters=0,training_only_discriminator_parameters=20608,
                 builder='load_data(); Network(SW,master,mode); load_trainable_state(checkpoint[trainable_state]); output dict[mode]')
        choices=[new if c['target']=='SW' else c for c in choices]
        model.close();del model,cp,data,master;gc.collect();torch.cuda.empty_cache()
    scores=metric(truth,combined);ctrl=reference.copy();ctrl[:,2]=predictions['blocked']
    contrast=diagnostics(truth,ctrl,predictions['reversed'],wells,'SW')
    upctrl=reference.copy();upctrl[:,2]=updates['blocked'];upcontrast=diagnostics(truth,upctrl,updates['reversed'],wells,'SW')
    diag=diagnostics(truth,reference,combined[:,2],wells,'SW')
    result=native(dict(reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],candidates=records,
        adopted=adopted,selected_mode=mode if adopted else None,new_target_full_disk_replay=replay,
        reversed_minus_blocked_selected=contrast,reversed_minus_blocked_post_update=upcontrast,combined_diagnostics=diag,
        parameter_matched=True,gradient_reversal_comparison=True,all_two_complete=True,actual_fixed_three_column_scoring=True,warmup_reversed_minus_blocked_SW=warmup,
        rows=144665,screening_fold=5,strict805_qualified=scores['TOTAL']>80.5,complete_five_folds=False,remaining_folds_started=False,
        old_deliveries_unchanged=True,finished_at=datetime.now().astimezone().isoformat()))
    atomic_array(STUDY/'assembled_pred.npy',combined);atomic_json(STUDY/'assembled_report.json',result)
    recipe=dict(previous);recipe.update(scores=native(scores),choices=choices,reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
    atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    events=[]
    for c in costs:events.extend([(datetime.fromisoformat(c['start']),1),(datetime.fromisoformat(c['end']),-1)])
    live=peak=0
    for _,d in sorted(events):live+=d;peak=max(peak,live)
    timing=dict(candidates=costs,first_launch_to_last_completion_seconds=(max(t for t,_ in events)-min(t for t,_ in events)).total_seconds(),peak_process_concurrency=peak,
                finalization_seconds=time.perf_counter()-start,scope='per-process load/warmup/train/eval included; total interval includes explicit staging; coding and finalization separate')
    atomic_json(STUDY/'timing.json',timing)
    lines=['# v40 Training-well gradient reversal: final results','',f"Original fifth TOTAL {refs['TOTAL']:.12f} -> {scores['TOTAL']:.12f}, delta {scores['TOTAL']-refs['TOTAL']:+.12f}.",'',
        '|Gradient|Trainable parameters|Best SW after update|SW with parent fallback|Best/stop epoch|','|---|---:|---:|---:|---|']
    for r in records:
        p=r['report'];lines.append(f"|{r['mode']}|{p['trainable_parameters']}|{p['best_after_update']['target_score']:.9f}|{p['target_score']:.9f}|{p['best_epoch']}/{p['stopped_epoch']}|")
    lines+=['','## Mechanism and verification',
        'Matched5120 existing SW normalization affine parameters and20608 auxiliary classifier parameters. Both classifiers minimize train-well CE; only reversed sends the negative gradient to the backbone, after the prespecified warmup/ramp. Original heads/convolutions/routes, data/Q50 and decoder stay fixed.',
        'Historical well_adversarial_sequence was a multi-target full-network single-classifier study. This is a current-private-SW warm-start comparison, not its restart or a first discovery of gradient reversal.',
        'Well IDs are auxiliary labels for64 training wells only. Held-out wells map to-1 and never enter auxiliary training. IDs and classifier outputs are absent from prediction inputs. First actual batch verified trainable gradients; final frozen-state and144665 selected prediction restoration checks passed.',
        'Selected SW full disk reconstruction: '+str(replay)+'. Fixed three-column scoring completed.',
        '', '## Classifier diagnostics and comparison',
        'Identical warmup SW score differences: '+json.dumps(warmup)]
    for r in records:
        p=r['report'];dm=p['domain_metadata'];d=p['last_domain_diagnostic']
        lines.append(r['mode']+' final training classifier diagnostics: '+json.dumps(d)+f"; majority-class prior={dm['majority_prior_accuracy']:.6f}; normalization change L2={p['last_normalization_change_L2']:.6f}.")
    lines += [f"Reversed-blocked selected SW difference {contrast['channel_delta']:+.12f}, post-update-only difference {upcontrast['channel_delta']:+.12f}. Adopted={mode if adopted else None}.",
        'A learnable training-well classifier is not causal proof of harmful domain dependence. Positive differences remain selected development-fold evidence; they are not independent generalization proof.',
        'Close this coefficient/curriculum setting if neither arm improves the parent. Do not conclude that all adversarial learning is ineffective. Epoch0 fallback is not training gain.',
        '', '## Timing and outstanding objective',
        f"Same-work wall time {timing['first_launch_to_last_completion_seconds']:.2f}s, measured peak process overlap={peak}; individual times and peak memory are in timing.json.",
        'CUDA linear interpolation backward uses warn_only and is not claimed bitwise repeatable. Original5foldOOF>82 remains unachieved. No other folds, old jobs or delivery replacements were started.']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v40骨干归一化对照完成';msg=marker+f"\n两项完成，TOTAL={scores['TOTAL']:.12f}，增量{scores['TOTAL']-refs['TOTAL']:+.12f}，采用={mode if adopted else None}。无其他折或交付修改；见well_invariance_v40_20260929/FINAL.md。"
    for p in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md',STUDY/'README.md'):append(p,marker,msg)
    print(json.dumps(native(dict(scores=scores,adopted=adopted,mode=mode if adopted else None,post_update_contrast=upcontrast['channel_delta'],timing=timing))),flush=True)

if __name__=='__main__':main()
