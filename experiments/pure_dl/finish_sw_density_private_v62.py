"""Finalize matched full-well context versus fixed reference and score assembly."""
import gc,hashlib,json,time
from datetime import datetime
import numpy as np
import torch
from sw_density_private_v62 import ROOT,STUDY,PREVIOUS,MODES,Network,Evaluator,load_data,reference_prediction
from run_retrieval_smooth_v36 import jobs,append
from finish_all_heads_v26 import diagnostics,native
from train_dual8222_heads80 import atomic_json,atomic_array
from train import metric

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def main():
    if jobs():raise RuntimeError('Training active')
    start=time.perf_counter();old=read(PREVIOUS/'CURRENT_RECIPE.json');reference=reference_prediction()
    with np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/oof.npz') as z:
        truth=z['truth'][z['fold']==4].copy();wells=z['well'][z['fold']==4].copy()
    refs=metric(truth,reference);assert truth.shape==reference.shape==(144665,3)
    assert all(abs(refs[k]-v)<1e-9 for k,v in old['scores'].items())
    records=[];preds={};histories={};timings=[];hashes=[]
    for mode in MODES:
        out=ROOT/f'outputs/pure_dl_sw_density_private_v62_SW_{mode}_fold5'
        cfg=read(out/'config.json');rep=read(out/'fold_5_report.json');hist=read(out/'validation_history.json')
        for n,h in cfg['source_hashes'].items():assert hashlib.sha256((ROOT/'experiments/pure_dl'/n).read_bytes()).hexdigest()==h,n
        assert len(cfg['bank_manifest']['train_wells'])==64 and not cfg['bank_manifest']['validation_fit']
        assert read(out/'initial.json')['all144665_parent_exact']
        assert rep['all144665_selected_predictions_restored_exactly']
        assert rep['fixed_branch_unchanged'] and rep['gradient_scope']['fixed_branch_no_grad']
        assert all(rep['gradient_scope']['gradient_norms'][g]>0 for g in ('density.body.','density.output.','density.condition.'))
        pred=np.load(out/'channel_pred.npy');diag=diagnostics(truth,reference,pred,wells,'SW')
        assert abs(diag['scores']['SW']-rep['target_score'])<1e-9
        updated=max(hist,key=lambda x:x['target_score'])
        records.append(dict(mode=mode,report=rep,diagnostics=diag,best_after_update=updated));preds[mode]=pred;histories[mode]=hist;hashes.append(dict(state=cfg['initial_state_sha256'],core=cfg['initial_core_sha256'],head=cfg['initial_head_sha256'],prediction=read(out/'initial.json')['initial_prediction_sha256']))
        launch=read(out/'launch.json');resume=read(out/'resume_1_launch.json')
        failed_at=datetime.fromtimestamp((out/'failure.json').stat().st_mtime).astimezone().isoformat()
        timings.append(dict(mode=mode,start=launch['started_at'],end=rep['finished_at'],seconds=rep['seconds'],peak_allocated_MiB=rep['peak_allocated_MiB'],
            initial_failure_at=failed_at,failure='CUDA OOM',resumed_from_completed_epoch=7,
            attempts=[dict(start=launch['started_at'],end=failed_at,outcome='failed_CUDA_OOM'),dict(start=resume['started_at'],end=rep['finished_at'],outcome='completed_serial_resume')],
            seconds_scope='saved completed-epoch elapsed plus resumed execution; failed in-flight epoch work not included, see attempt wall times'))
    assert hashes[0]['state']==hashes[1]['state'] and hashes[0]['core']==hashes[1]['core'] and hashes[0]['head']==hashes[1]['head'] and hashes[0]['prediction']==hashes[1]['prediction']
    best=max(records,key=lambda x:x['report']['target_score']);adopted=best['report']['target_score']>refs['SW']+1e-9
    combined=reference.copy();choices=[dict(c) for c in old['choices']];replay=False
    if adopted:
        torch.set_num_threads(3);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
        d,meta=load_data();mode=best['mode'];model=Network(mode,meta).cuda();saved=torch.load(ROOT/best['report']['checkpoint'],map_location='cpu',weights_only=False)
        model.load_state_dict(saved['state_dict'],strict=True);p,score,_=Evaluator(d,meta,2,reference)(model)
        assert np.array_equal(p,preds[mode]) and abs(score-best['report']['target_score'])<1e-9
        combined[:,2]=p;replay=True
        choice=dict(target='SW',mode=mode,key='v62_SW_'+mode,architecture='private_SW_density_backbone_fixed_default',model_module='sw_density_private_v62',
            checkpoint=best['report']['checkpoint'],target_score=score,best_epoch=best['report']['best_epoch'],
            builder='load_data builds input-only banks and train64 reference; Network(mode,metadata); load_state_dict(state_dict); Evaluator supplies own-well bank',
            parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
        choices=[choice if c['target']=='SW' else c for c in choices]
        del model,saved,d,meta;gc.collect();torch.cuda.empty_cache()
    scores=metric(truth,combined);ctrl=reference.copy();ctrl[:,2]=preds['gate_aux'];contrast=diagnostics(truth,ctrl,preds['numeric_only'],wells,'SW')
    by={x['epoch']:x for x in histories['gate_aux']}
    matched=[dict(epoch=x['epoch'],numeric_minus_aux_SW=x['target_score']-by[x['epoch']]['target_score']) for x in histories['numeric_only'] if x['epoch'] in by]
    result=native(dict(reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],candidates=records,
        adopted=adopted,selected_mode=best['mode'] if adopted else None,new_target_full_disk_replay=replay,numeric_minus_aux=contrast,
        post_update_numeric_minus_aux=records[1]['best_after_update']['target_score']-records[0]['best_after_update']['target_score'],
        matched_trajectory=matched,initial_state_hash_equal=True,initial_core_hash_equal=True,initial_head_hash_equal=True,initial_prediction_hash_equal=True,actual_fixed_three_column_scoring=True,rows=144665,screening_fold=5,
        strict805_qualified=scores['TOTAL']>80.5,complete_five_folds=False,remaining_folds_started=False,old_deliveries_unchanged=True,
        finished_at=datetime.now().astimezone().isoformat()))
    atomic_array(STUDY/'assembled_pred.npy',combined);atomic_json(STUDY/'assembled_report.json',result)
    recipe=dict(old);recipe.update(scores=native(scores),choices=choices,reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
    atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    events=[]
    for t in timings:
        for attempt in t['attempts']:events.extend([(datetime.fromisoformat(attempt['start']),1),(datetime.fromisoformat(attempt['end']),-1)])
    live=peak=0
    for _,d in sorted(events):live+=d;peak=max(peak,live)
    timing=dict(candidates=timings,first_launch_to_last_completion_seconds=(max(t for t,_ in events)-min(t for t,_ in events)).total_seconds(),peak_process_concurrency=peak,
        finalization_seconds=time.perf_counter()-start,scope='initial concurrent attempts failed at epoch8 with CUDA OOM; serial explicit resume from epoch7; wall includes failed attempts, recovery gap, loading and warmup; finalization separate')
    atomic_json(STUDY/'timing.json',timing)
    lines=['# v62密度私有骨干结果','',f"固定第五折TOTAL {refs['TOTAL']:.12f} → {scores['TOTAL']:.12f}。",'',
        '|模式|训练后最高SW|含回退SW|最佳/停止轮|','|---|---:|---:|---|']
    for row in records:
        r=row['report'];lines.append(f"|{row['mode']}|{row['best_after_update']['target_score']:.12f}|{r['target_score']:.12f}|{r['best_epoch']}/{r['stopped_epoch']}|")
    lines+=['','## 探索重点与完成验证','同1678962总参数/839481优化器参数和完整初始函数；固定v56默认分支，另一个完整密度骨干可训练，推理用固定gate和私有密度。gate_aux用密度自己的gate作辅助监督，numeric_only系数0且该129参数训练无梯度、推理不用。其余原Q50/1024随机上下文/Gaussian损失/官方评分不变。',
        '完整初始状态/输出头/144665点预测一致，私有骨干/头/FiLM实际梯度有效；固定分支权重及无梯度、初始/最佳恢复、源哈希和实际三列评分核验。新采用模型磁盘重建='+str(replay)+'。',
        '', '## 收益与方向边界',f"numeric-aux含回退SW差{contrast['channel_delta']:+.12f}；训练后最佳差{result['post_update_numeric_minus_aux']:+.12f}。采用={result['selected_mode']}。",
        '第0轮回退不算训练收益。若仅数值监督无匹配收益，不扫描本辅助系数或支路数；结论限当前固定默认/私有密度结构，不否定全部任务解耦。正差仍为重复开发折证据，未作独立新井确认。',
        '', '## 耗时与未做',json.dumps(timing,ensure_ascii=False,indent=2),
        '首次两项并发在第8轮CUDA OOM，保留failure.json与原日志；各自完整状态在第7轮，从原子模型/优化器/调度/RNG状态显式串行续跑。源码和配置未改，不是从头重训；失败不算负结论或成功。CUDA反向仍不保证逐位重训。完整五折OOF>82未完成，无其他折/交付修改。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v62密度私有骨干对照完成';note=marker+f"\n两项完成，实际TOTAL={scores['TOTAL']:.12f}，采用{result['selected_mode']}，训练后numeric-aux={result['post_update_numeric_minus_aux']:+.12f}。详见sw_density_private_v62_20260929/FINAL.md；无其他折或交付修改。"
    for p in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md',STUDY/'README.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):append(p,marker,note)
    print(json.dumps(native(dict(scores=scores,adopted=adopted,mode=result['selected_mode'],post_update_delta=result['post_update_numeric_minus_aux'],timing=timing))),flush=True)

if __name__=='__main__':main()
