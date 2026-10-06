"""Finalize the actual task-memory ablation; no fifth-fold rowwise selection."""
import gc,hashlib,json,time
from datetime import datetime
import numpy as np
import torch
from well_memory_v47 import ROOT,STUDY,PREVIOUS,MODES,Network,Evaluator,load_data,reference_prediction
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
        out=ROOT/f'outputs/pure_dl_well_memory_v47_SW_{mode}_fold5'
        cfg=read(out/'config.json');rep=read(out/'fold_5_report.json');hist=read(out/'validation_history.json')
        for n,h in cfg['source_hashes'].items():assert hashlib.sha256((ROOT/'experiments/pure_dl'/n).read_bytes()).hexdigest()==h,n
        assert len(cfg['bank_manifest']['train_wells'])==64 and not cfg['bank_manifest']['validation_fit']
        assert read(out/'initial.json')['all144665_parent_exact'] and rep['all144665_selected_predictions_restored_exactly']
        pred=np.load(out/'channel_pred.npy');diag=diagnostics(truth,reference,pred,wells,'SW')
        assert abs(diag['scores']['SW']-rep['target_score'])<1e-9
        updated=max(hist,key=lambda x:x['target_score'])
        records.append(dict(mode=mode,report=rep,diagnostics=diag,best_after_update=updated));preds[mode]=pred;histories[mode]=hist;hashes.append(cfg['initial_state_sha256'])
        launch=read(out/'launch.json');timings.append(dict(mode=mode,start=launch['started_at'],end=rep['finished_at'],seconds=rep['seconds'],peak_allocated_MiB=rep['peak_allocated_MiB']))
    assert hashes[0]==hashes[1]
    best=max(records,key=lambda x:x['report']['target_score']);adopted=best['report']['target_score']>refs['SW']+1e-9
    combined=reference.copy();choices=[dict(c) for c in old['choices']];replay=False
    if adopted:
        torch.set_num_threads(3);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
        d,meta=load_data();mode=best['mode'];model=Network(mode,meta).cuda();saved=torch.load(ROOT/best['report']['checkpoint'],map_location='cpu',weights_only=False)
        model.load_state_dict(saved['state_dict'],strict=True);p,score,_=Evaluator(d,meta,2,reference)(model)
        assert np.array_equal(p,preds[mode]) and abs(score-best['report']['target_score'])<1e-9
        combined[:,2]=p;replay=True
        choice=dict(target='SW',mode=mode,key='v47_SW_'+mode,architecture='task_queries_whole_well_memory',model_module='well_memory_v47',
            checkpoint=best['report']['checkpoint'],target_score=score,best_epoch=best['report']['best_epoch'],
            builder='load_data builds input-only banks; Network(mode,metadata); load_state_dict(state_dict); original Evaluator supplies own-well bank',
            parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
        choices=[choice if c['target']=='SW' else c for c in choices]
        del model,saved,d,meta;gc.collect();torch.cuda.empty_cache()
    scores=metric(truth,combined);ctrl=reference.copy();ctrl[:,2]=preds['global_query'];contrast=diagnostics(truth,ctrl,preds['section_query'],wells,'SW')
    by={x['epoch']:x for x in histories['global_query']}
    matched=[dict(epoch=x['epoch'],section_minus_global_SW=x['target_score']-by[x['epoch']]['target_score']) for x in histories['section_query'] if x['epoch'] in by]
    result=native(dict(reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],candidates=records,
        adopted=adopted,selected_mode=best['mode'] if adopted else None,new_target_full_disk_replay=replay,section_minus_global=contrast,
        post_update_section_minus_global=records[1]['best_after_update']['target_score']-records[0]['best_after_update']['target_score'],
        matched_trajectory=matched,initial_state_hash_equal=True,actual_fixed_three_column_scoring=True,rows=144665,screening_fold=5,
        strict805_qualified=scores['TOTAL']>80.5,complete_five_folds=False,remaining_folds_started=False,old_deliveries_unchanged=True,
        finished_at=datetime.now().astimezone().isoformat()))
    atomic_array(STUDY/'assembled_pred.npy',combined);atomic_json(STUDY/'assembled_report.json',result)
    recipe=dict(old);recipe.update(scores=native(scores),choices=choices,reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
    atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    events=[]
    for t in timings:events.extend([(datetime.fromisoformat(t['start']),1),(datetime.fromisoformat(t['end']),-1)])
    live=peak=0
    for _,d in sorted(events):live+=d;peak=max(peak,live)
    timing=dict(candidates=timings,first_launch_to_last_completion_seconds=(max(t for t,_ in events)-min(t for t,_ in events)).total_seconds(),peak_process_concurrency=peak,
        finalization_seconds=time.perf_counter()-start,scope='loading/input-bank construction/warmup/training/evaluation plus launch gap; authoring/final replay separate')
    atomic_json(STUDY/'timing.json',timing)
    lines=['# v47 任务查询整井记忆结果','',f"固定第五折TOTAL {refs['TOTAL']:.12f} → {scores['TOTAL']:.12f}。",'',
        '|模式|训练后最高SW|含回退SW|最佳/停止轮|','|---|---:|---:|---|']
    for row in records:
        r=row['report'];lines.append(f"|{row['mode']}|{row['best_after_update']['target_score']:.12f}|{r['target_score']:.12f}|{r['best_epoch']}/{r['stopped_epoch']}|")
    lines+=['','## 探索重点与验证','两组同900985参数和初始化，状态/数值各自查询输入记忆；全骨干和FiLM均继续训练。唯一设计差别为井段查询是否平均为窗口固定查询。原训练井隔离、Q50、1024上下文及官方解码评分保持。',
        '初始父函数与恢复全144665点一致，真实训练梯度、源哈希和同初始化哈希已核验；记忆无标签或井ID特征。新采用目标额外磁盘重建='+str(replay)+'；实际三列评分完成。',
        '', '## 收益与不继续的方向',f"section-global含回退SW差{contrast['channel_delta']:+.12f}；训练后最佳差{result['post_update_section_minus_global']:+.12f}。采用={result['selected_mode']}。",
        '第0轮回退不算训练收益。两组都含任务查询，不能用本对照单独证明任务分离有益。若均不提高则停止本配置扩展，不扫描查询维数/记忆点数/头数；不泛化为全部神经记忆无效。正差仍是重复开发折证据，尚无独立新井确认。',
        '', '## 耗时及未做',json.dumps(timing,ensure_ascii=False,indent=2),
        '完整原五折OOF>82未完成；未补其他折或覆盖交付。CUDA注意力/上采样反向可能非确定，不宣称重新训练逐位复现。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v47任务查询对照完成';note=marker+f"\n两项完成，实际TOTAL={scores['TOTAL']:.12f}，采用{result['selected_mode']}，训练后section-global={result['post_update_section_minus_global']:+.12f}。详见well_memory_v47_20260929/FINAL.md；无其他折或交付修改。"
    for p in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md',STUDY/'README.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):append(p,marker,note)
    print(json.dumps(native(dict(scores=scores,adopted=adopted,mode=result['selected_mode'],post_update_delta=result['post_update_section_minus_global'],timing=timing))),flush=True)

if __name__=='__main__':main()
