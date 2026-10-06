"""Finalize matched SW final-block original-anchor comparison."""
from datetime import datetime
import gc,hashlib,json,time
import numpy as np
import torch
from anchored_backbone_v42 import ROOT,STUDY,PREVIOUS,MODES,Network,Evaluator,load_data,reference_prediction
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
        out=ROOT/f'outputs/pure_dl_anchored_backbone_v42_SW_{mode}_fold5';cfg=read(out/'config.json');rep=read(out/'fold_5_report.json')
        for n,d in cfg['source_hashes'].items():assert hashlib.sha256((ROOT/'experiments/pure_dl'/n).read_bytes()).hexdigest()==d,n
        pred=np.load(out/'channel_pred.npy');updated=np.load(out/'best_updated_pred.npy')
        diag=diagnostics(truth,reference,pred,wells,'SW');up=diagnostics(truth,reference,updated,wells,'SW')
        assert abs(diag['scores']['SW']-rep['target_score'])<1e-9 and abs(up['scores']['SW']-rep['best_after_update']['target_score'])<1e-9
        assert rep['all144665_selected_predictions_restored_exactly'] and rep['all_frozen_state_unchanged']
        if rep['best_epoch']==0:assert np.array_equal(pred,reference[:,2])
        records.append(dict(mode=mode,report=rep,selected=diag,post_update=up));predictions[mode]=pred;updates[mode]=updated
        launch=read(out/'launch.json');costs.append(dict(mode=mode,start=launch['started_at'],end=rep['finished_at'],seconds=rep['seconds'],peak_allocated_MiB=rep['peak_allocated_MiB'],median_epoch_seconds=float(np.median([x['seconds'] for x in rep['epoch_timing']]))))
    histories={m:read(ROOT/f'outputs/pure_dl_anchored_backbone_v42_SW_{m}_fold5/validation_history.json') for m in MODES}
    free_by_epoch={r['epoch']:r for r in histories['free']}
    matched=[dict(epoch=r['epoch'],anchored_minus_free_SW=r['target_score']-free_by_epoch[r['epoch']]['target_score'],
                  free_anchor=free_by_epoch[r['epoch']]['anchor'],anchored_anchor=r['anchor'])
             for r in histories['anchored'] if r['epoch'] in free_by_epoch]
    trajectory=dict(matched_epochs=matched,positive_epochs=sum(r['anchored_minus_free_SW']>0 for r in matched),
                    last_delta=matched[-1]['anchored_minus_free_SW'],median_delta=float(np.median([r['anchored_minus_free_SW'] for r in matched])),
                    interpretation='Matched-seed/window descriptive trajectory; epochs are dependent, not independent statistical replicates.')
    atomic_json(STUDY/'matched_trajectory.json',trajectory)
    best=max(records,key=lambda x:x['report']['target_score']);mode=best['mode'];adopted=best['report']['target_score']>refs['SW']+1e-9
    combined=reference.copy();choices=[dict(c) for c in previous['choices']];replay=False
    if adopted:
        torch.set_num_threads(3);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
        data,master=load_data();model=Network('SW',master,mode).cuda()
        cp=torch.load(ROOT/best['report']['checkpoint'],map_location='cpu',weights_only=False)
        model.load_trainable_state(cp['backbone_state'])
        pred,score,_=Evaluator(data,master,2,reference).many(model,(mode,))[mode]
        assert np.array_equal(pred,predictions[mode]) and abs(score-best['report']['target_score'])<1e-9
        combined[:,2]=pred;replay=True
        new=dict(target='SW',key='v42_SW_'+mode,mode=mode,architecture='retained_private_SW_dec0_adaptation',
                 model_module='anchored_backbone_v42',checkpoint=best['report']['checkpoint'],best_epoch=best['report']['best_epoch'],target_score=score,
                 parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),added_parameters=0,
                 builder='load_data(); Network(SW,master,mode); load_trainable_state(checkpoint[backbone_state]); output dict[mode]')
        choices=[new if c['target']=='SW' else c for c in choices]
        model.close();del model,cp,data,master;gc.collect();torch.cuda.empty_cache()
    scores=metric(truth,combined);ctrl=reference.copy();ctrl[:,2]=predictions['free']
    contrast=diagnostics(truth,ctrl,predictions['anchored'],wells,'SW')
    upctrl=reference.copy();upctrl[:,2]=updates['free'];upcontrast=diagnostics(truth,upctrl,updates['anchored'],wells,'SW')
    diag=diagnostics(truth,reference,combined[:,2],wells,'SW')
    result=native(dict(reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],candidates=records,
        adopted=adopted,selected_mode=mode if adopted else None,new_target_full_disk_replay=replay,
        anchored_minus_free_selected=contrast,anchored_minus_free_post_update=upcontrast,combined_diagnostics=diag,
        parameter_matched=True,original_anchor_comparison=True,matched_trajectory=trajectory,all_two_complete=True,actual_fixed_three_column_scoring=True,
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
    lines=['# v42 SW末端残差块：原权重约束对照','',f"第五折TOTAL {refs['TOTAL']:.12f} → {scores['TOTAL']:.12f}，差值 {scores['TOTAL']-refs['TOTAL']:+.12f}。",'',
        '|候选|参数|训练后最高SW|含原模型回退SW|最佳/停止轮|最后原点惩罚|','|---|---:|---:|---:|---|---:|']
    for r in records:
        p=r['report'];lines.append(f"|{r['mode']}|{p['trainable_parameters']}|{p['best_after_update']['target_score']:.9f}|{p['target_score']:.9f}|{p['best_epoch']}/{p['stopped_epoch']}|{p['last_normalized_anchor']:.9f}|")
    lines+=['','## 探索重点及已完成验证','两个私有SW专家已有dec0残差块全部参数可训练，其他原参数冻结。free/anchored同初始化、同容量、同训练窗口和优化条件，只改原权重惩罚系数0/1；公式与固定RMS下限见README和config。无伪标签、扩容或冷启动。',
        '原输入、数据分折、Q50、SW NLL、80/18、原动作和评分保持。第一真实训练批有梯度；原144665点初始复现、最佳恢复、冻结状态、源哈希与三列评分通过。',
        '新采用SW完整磁盘重建：'+str(replay)+'。',
        '', '## 收益与不继续方向',
        f"anchored-free：含回退SW差{contrast['channel_delta']:+.12f}，训练后最佳差{upcontrast['channel_delta']:+.12f}。采用={mode if adopted else None}。",
        f"相同轮次描述性对照：{trajectory['positive_epochs']}/{len(matched)}轮约束组更高，中位SW差{trajectory['median_delta']:+.12f}，最后差{trajectory['last_delta']:+.12f}。各轮不独立，不能当作多次复现实验。",
        '原点约束限制参数漂移不等价于超过父模型。若匹配消融为正，可将该机制带入后续独立假设，但不据此替换当前配方。第0轮回退不是训练收益；仍为重复使用的第五开发折，不是独立新井证据。',
        '两项均未改善父模型时，不延伸本系数/学习率扫描。结论限当前SW末端残差块与固定设置，不泛化为所有骨干微调或正则化无效。' if not adopted else '保留实测有效候选，仍须另行考察泛化稳定性，不自动补折。',
        '', '## 耗时、目标与未做',f"同等工作量总墙钟{timing['first_launch_to_last_completion_seconds']:.2f}秒，实际峰值并发{peak}；单项耗时及显存见timing.json。加载/预热/训练/评估计入；编码和收尾不计入。",
        '完整原五折OOF>82未完成。POR/PERM、旧交付不变，无其他折或旧任务重启。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v42末端块原权重对照完成';msg=marker+f"\n两项完成，TOTAL={scores['TOTAL']:.12f}，增量{scores['TOTAL']-refs['TOTAL']:+.12f}，采用={mode if adopted else None}。无其他折或交付修改；见anchored_backbone_v42_20260929/FINAL.md。"
    for p in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md',STUDY/'README.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):append(p,marker,msg)
    print(json.dumps(native(dict(scores=scores,adopted=adopted,mode=mode if adopted else None,post_update_contrast=upcontrast['channel_delta'],timing=timing))),flush=True)

if __name__=='__main__':main()
