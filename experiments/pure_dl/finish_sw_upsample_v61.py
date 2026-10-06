"""Finalize matched full-well context versus fixed reference and score assembly."""
import gc,hashlib,json,time
from datetime import datetime
import numpy as np
import torch
from sw_upsample_v61 import ROOT,STUDY,PREVIOUS,MODES,Network,Evaluator,load_data,reference_prediction
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
        out=ROOT/f'outputs/pure_dl_sw_upsample_v61_SW_{mode}_fold5'
        cfg=read(out/'config.json');rep=read(out/'fold_5_report.json');hist=read(out/'validation_history.json')
        for n,h in cfg['source_hashes'].items():assert hashlib.sha256((ROOT/'experiments/pure_dl'/n).read_bytes()).hexdigest()==h,n
        assert len(cfg['bank_manifest']['train_wells'])==64 and not cfg['bank_manifest']['validation_fit']
        assert read(out/'initial.json')['all144665_parent_exact']
        assert rep['all144665_selected_predictions_restored_exactly']
        assert all(rep['gradient_scope']['gradient_norms'][g]>0 for g in ('upsample.query.','upsample.key.'))
        pred=np.load(out/'channel_pred.npy');diag=diagnostics(truth,reference,pred,wells,'SW')
        assert abs(diag['scores']['SW']-rep['target_score'])<1e-9
        updated=max(hist,key=lambda x:x['target_score'])
        records.append(dict(mode=mode,report=rep,diagnostics=diag,best_after_update=updated));preds[mode]=pred;histories[mode]=hist;hashes.append(dict(state=cfg['initial_state_sha256'],core=cfg['initial_core_sha256'],head=cfg['initial_head_sha256'],prediction=read(out/'initial.json')['initial_prediction_sha256']))
        launch=read(out/'launch.json');timings.append(dict(mode=mode,start=launch['started_at'],end=rep['finished_at'],seconds=rep['seconds'],peak_allocated_MiB=rep['peak_allocated_MiB']))
    assert hashes[0]['state']==hashes[1]['state'] and hashes[0]['core']==hashes[1]['core'] and hashes[0]['head']==hashes[1]['head'] and hashes[0]['prediction']==hashes[1]['prediction']
    best=max(records,key=lambda x:x['report']['target_score']);adopted=best['report']['target_score']>refs['SW']+1e-9
    combined=reference.copy();choices=[dict(c) for c in old['choices']];replay=False
    if adopted:
        torch.set_num_threads(3);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
        d,meta=load_data();mode=best['mode'];model=Network(mode,meta).cuda();saved=torch.load(ROOT/best['report']['checkpoint'],map_location='cpu',weights_only=False)
        model.load_state_dict(saved['state_dict'],strict=True);p,score,_=Evaluator(d,meta,2,reference)(model)
        assert np.array_equal(p,preds[mode]) and abs(score-best['report']['target_score'])<1e-9
        combined[:,2]=p;replay=True
        choice=dict(target='SW',mode=mode,key='v61_SW_'+mode,architecture='private_SW_neural_depth_upsampling',model_module='sw_upsample_v61',
            checkpoint=best['report']['checkpoint'],target_score=score,best_epoch=best['report']['best_epoch'],
            builder='load_data builds input-only banks and train64 reference; Network(mode,metadata); load_state_dict(state_dict); Evaluator supplies own-well bank',
            parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
        choices=[choice if c['target']=='SW' else c for c in choices]
        del model,saved,d,meta;gc.collect();torch.cuda.empty_cache()
    scores=metric(truth,combined);ctrl=reference.copy();ctrl[:,2]=preds['coarse_query'];contrast=diagnostics(truth,ctrl,preds['local_query'],wells,'SW')
    by={x['epoch']:x for x in histories['coarse_query']}
    matched=[dict(epoch=x['epoch'],local_minus_coarse_SW=x['target_score']-by[x['epoch']]['target_score']) for x in histories['local_query'] if x['epoch'] in by]
    result=native(dict(reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],candidates=records,
        adopted=adopted,selected_mode=best['mode'] if adopted else None,new_target_full_disk_replay=replay,local_minus_coarse=contrast,
        post_update_local_minus_coarse=records[1]['best_after_update']['target_score']-records[0]['best_after_update']['target_score'],
        matched_trajectory=matched,initial_state_hash_equal=True,initial_core_hash_equal=True,initial_head_hash_equal=True,initial_prediction_hash_equal=True,actual_fixed_three_column_scoring=True,rows=144665,screening_fold=5,
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
    lines=['# v61神经深度还原结果','',f"固定第五折TOTAL {refs['TOTAL']:.12f} → {scores['TOTAL']:.12f}。",'',
        '|模式|训练后最高SW|含回退SW|最佳/停止轮|','|---|---:|---:|---|']
    for row in records:
        r=row['report'];lines.append(f"|{row['mode']}|{row['best_after_update']['target_score']:.12f}|{r['target_score']:.12f}|{r['best_epoch']}/{r['stopped_epoch']}|")
    lines+=['','## 探索重点与完成验证','同842585参数和完整初始函数，v56完整SW模型新增3104参数rank16双邻居插值；coarse_query读取插值粗特征，local_query读取精细TCN特征。query全零初始，key相同，全网络统一lr3e-5；原训练井/Q50/1024随机上下文/Gaussian损失及官方评分保持。',
        '两组完整初始状态/核心/输出头/144665点预测相同；真实骨干/头/FiLM与新query/key梯度均有效，各自最佳恢复一致，源哈希与三列评分已核验。输入库无标签。新采用目标磁盘重建='+str(replay)+'。',
        '', '## 收益与方向边界',f"local-coarse含回退SW差{contrast['channel_delta']:+.12f}；训练后最佳差{result['post_update_local_minus_coarse']:+.12f}。采用={result['selected_mode']}。",
        '第0轮回退不算训练收益。若局部查询无匹配收益，不扫描本插值rank/邻居数/温度；结论限当前双邻居结构，不否定所有神经上采样。正差仍为重复开发折证据，未作独立新井确认。',
        '', '## 耗时与未做',json.dumps(timing,ensure_ascii=False,indent=2),
        '完整原五折OOF>82未完成；未补其他折或覆盖交付。原注意力/上采样反向可能非确定，不声称训练逐位复现。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v61深度还原对照完成';note=marker+f"\n两项完成，实际TOTAL={scores['TOTAL']:.12f}，采用{result['selected_mode']}，训练后local-coarse={result['post_update_local_minus_coarse']:+.12f}。详见sw_upsample_v61_20260929/FINAL.md；无其他折或交付修改。"
    for p in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md',STUDY/'README.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):append(p,marker,note)
    print(json.dumps(native(dict(scores=scores,adopted=adopted,mode=result['selected_mode'],post_update_delta=result['post_update_local_minus_coarse'],timing=timing))),flush=True)

if __name__=='__main__':main()
