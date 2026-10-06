"""Finalize both SW normalization scopes and score the retained three-channel recipe."""
from datetime import datetime
import gc,hashlib,json,time
import numpy as np
import torch
from backbone_norm_v39 import ROOT,STUDY,PREVIOUS,MODES,Network,Evaluator,load_data,reference_prediction
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
        out=ROOT/f'outputs/pure_dl_backbone_norm_v39_SW_{mode}_fold5';cfg=read(out/'config.json');rep=read(out/'fold_5_report.json')
        for n,d in cfg['source_hashes'].items():assert hashlib.sha256((ROOT/'experiments/pure_dl'/n).read_bytes()).hexdigest()==d,n
        pred=np.load(out/'channel_pred.npy');updated=np.load(out/'best_updated_pred.npy')
        diag=diagnostics(truth,reference,pred,wells,'SW');up=diagnostics(truth,reference,updated,wells,'SW')
        assert abs(diag['scores']['SW']-rep['target_score'])<1e-9 and abs(up['scores']['SW']-rep['best_after_update']['target_score'])<1e-9
        assert rep['all144665_selected_predictions_restored_exactly'] and rep['all_frozen_state_unchanged']
        if rep['best_epoch']==0:assert np.array_equal(pred,reference[:,2])
        records.append(dict(mode=mode,report=rep,selected=diag,post_update=up));predictions[mode]=pred;updates[mode]=updated
        launch=read(out/'launch.json');costs.append(dict(mode=mode,start=launch['started_at'],end=rep['finished_at'],seconds=rep['seconds'],peak_allocated_MiB=rep['peak_allocated_MiB'],median_epoch_seconds=float(np.median([x['seconds'] for x in rep['epoch_timing']]))))
    best=max(records,key=lambda x:x['report']['target_score']);mode=best['mode'];adopted=best['report']['target_score']>refs['SW']+1e-9
    combined=reference.copy();choices=[dict(c) for c in previous['choices']];replay=False
    if adopted:
        torch.set_num_threads(3);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
        data,master=load_data();model=Network('SW',master,mode).cuda()
        cp=torch.load(ROOT/best['report']['checkpoint'],map_location='cpu',weights_only=False)
        model.load_trainable_state(cp['normalization_state'])
        pred,score,_=Evaluator(data,master,2,reference).many(model,(mode,))[mode]
        assert np.array_equal(pred,predictions[mode]) and abs(score-best['report']['target_score'])<1e-9
        combined[:,2]=pred;replay=True
        new=dict(target='SW',key='v39_SW_'+mode,mode=mode,architecture='retained_private_SW_norm_affine_adaptation',
                 model_module='backbone_norm_v39',checkpoint=best['report']['checkpoint'],best_epoch=best['report']['best_epoch'],target_score=score,
                 parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),added_parameters=0,
                 builder='load_data(); Network(SW,master,mode); load_trainable_state(checkpoint[normalization_state]); output dict[mode]')
        choices=[new if c['target']=='SW' else c for c in choices]
        model.close();del model,cp,data,master;gc.collect();torch.cuda.empty_cache()
    scores=metric(truth,combined);ctrl=reference.copy();ctrl[:,2]=predictions['decoder_norm']
    contrast=diagnostics(truth,ctrl,predictions['all_norm'],wells,'SW')
    upctrl=reference.copy();upctrl[:,2]=updates['decoder_norm'];upcontrast=diagnostics(truth,upctrl,updates['all_norm'],wells,'SW')
    diag=diagnostics(truth,reference,combined[:,2],wells,'SW')
    result=native(dict(reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],candidates=records,
        adopted=adopted,selected_mode=mode if adopted else None,new_target_full_disk_replay=replay,
        all_minus_decoder_selected=contrast,all_minus_decoder_post_update=upcontrast,combined_diagnostics=diag,
        parameter_matched=False,update_scope_comparison=True,all_two_complete=True,actual_fixed_three_column_scoring=True,
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
    lines=['# v39 SW骨干归一化仿射：最终结果','',f"第五折TOTAL {refs['TOTAL']:.12f} → {scores['TOTAL']:.12f}，差值 {scores['TOTAL']-refs['TOTAL']:+.12f}。",'',
        '|更新范围|可训练参数|训练后最高SW|含原模型回退SW|最佳/停止轮|','|---|---:|---:|---:|---|']
    for r in records:
        p=r['report'];lines.append(f"|{r['mode']}|{p['trainable_parameters']}|{p['best_after_update']['target_score']:.9f}|{p['target_score']:.9f}|{p['best_epoch']}/{p['stopped_epoch']}|")
    lines+=['','## 探索重点与验证','从当前已验证SW权重起步，仅更新两个私有专家的归一化仿射。卷积、头、路由权重不变，没有新增容量。训练前向保持可微，原梯度分离捕获被替换；仅自身NLL和原Q50。原动作、输入和评分保持。',
        '两个更新范围参数量不同，不能解释成同容量结构消融。第一真实训练批确认每个所选参数有梯度、冻结参数无梯度；最终冻结状态及144665点最佳恢复通过。',
        '新采用SW完整磁盘重建：'+str(replay)+'；实际固定POR/PERM后的三列评分完成。',
        '', '## 收益与不继续方向',
        f"全骨干-解码端：含回退SW差{contrast['channel_delta']:+.12f}，训练后最佳差{upcontrast['channel_delta']:+.12f}。采用={mode if adopted else None}。",
        '第0轮回退不是训练收益；正比较仍是反复使用的第五开发折证据，不是独立新井泛化。',
        '若两个范围均无新增收益，关闭本次SW归一化仿射微调设置，不继续扩大同范围学习率扫描；不能据此否定所有骨干适配。' if not adopted else '保留实际获益的更新范围，后续可在新独立实验测试其适用范围；不自动补折。',
        '', '## 耗时、目标及未做',f"同工作量墙钟{timing['first_launch_to_last_completion_seconds']:.2f}秒，实际峰值并发{peak}；单项耗时/显存见timing.json。",
        'PyTorch本地CUDA线性上采样反向不保证确定性，已记录warn_only；不声称训练逐位可重复。验证和权重恢复按原确定性路径核对。',
        '完整原五折OOF>82未完成；本批未动POR/PERM、未启动其他折/旧任务、未覆盖交付。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v39骨干归一化对照完成';msg=marker+f"\n两项完成，TOTAL={scores['TOTAL']:.12f}，增量{scores['TOTAL']-refs['TOTAL']:+.12f}，采用={mode if adopted else None}。无其他折或交付修改；见backbone_norm_v39_20260929/FINAL.md。"
    for p in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md',STUDY/'README.md'):append(p,marker,msg)
    print(json.dumps(native(dict(scores=scores,adopted=adopted,mode=mode if adopted else None,post_update_contrast=upcontrast['channel_delta'],timing=timing))),flush=True)

if __name__=='__main__':main()
