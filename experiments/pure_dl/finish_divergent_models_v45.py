"""Verify new architectures on fixed fold5, adopt only real channel gains."""
from datetime import datetime
import gc,hashlib,json,time
import numpy as np
import torch
from divergent_models_v45 import ROOT,STUDY,PREVIOUS,MODES,Network,Evaluator,load_data,reference_prediction
from run_retrieval_smooth_v36 import jobs,append
from finish_all_heads_v26 import diagnostics,native
from train_dual8222_heads80 import atomic_json,atomic_array
from train import metric

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def main():
    if jobs():raise RuntimeError('Training still active')
    start=time.perf_counter();old=read(PREVIOUS/'CURRENT_RECIPE.json');reference=reference_prediction()
    with np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/oof.npz') as z:
        truth=z['truth'][z['fold']==4].copy();wells=z['well'][z['fold']==4].copy()
    assert truth.shape==reference.shape==(144665,3)
    refs=metric(truth,reference);assert all(abs(refs[k]-v)<1e-9 for k,v in old['scores'].items())
    combined=reference.copy();records=[];selected=[];timings=[];choices=[dict(c) for c in old['choices']]
    for mode,spec in MODES.items():
        target=spec['target'];c=spec['c'];out=ROOT/f'outputs/pure_dl_divergent_v45_{target}_{mode}_fold5'
        report=read(out/'fold_5_report.json');cfg=read(out/'config.json')
        for name,digest in cfg['source_hashes'].items():assert hashlib.sha256((ROOT/'experiments/pure_dl'/name).read_bytes()).hexdigest()==digest,name
        pred=np.load(out/'channel_pred.npy');diag=diagnostics(truth,reference,pred,wells,target)
        assert abs(diag['scores'][target]-report['target_score'])<1e-9 and report['all144665_selected_predictions_restored_exactly']
        assert cfg['random_network_initialization'] and not cfg['loaded_network_weights'] and cfg['only_own_target_labels']
        records.append(dict(target=target,mode=mode,report=report,diagnostics=diag))
        if report['target_score']>refs[target]+1e-9:
            combined[:,c]=pred;selected.append(dict(target=target,c=c,mode=mode,target_score=report['target_score'],checkpoint=report['checkpoint'],best_epoch=report['best_epoch']))
        launch=read(out/'launch.json');timings.append(dict(mode=mode,target=target,start=launch['started_at'],end=report['finished_at'],seconds=report['seconds'],peak_allocated_MiB=report['peak_allocated_MiB']))
    replays=[]
    if selected:
        torch.set_num_threads(3);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
        data,metadata=load_data()
        for s in selected:
            model=Network(s['mode'],metadata).cuda();saved=torch.load(ROOT/s['checkpoint'],map_location='cpu',weights_only=False)
            model.load_state_dict(saved['state_dict'],strict=True)
            pred,score,_=Evaluator(data,metadata,s['c'],reference)(model)
            assert np.array_equal(pred,combined[:,s['c']]) and abs(score-s['target_score'])<1e-9
            for ch in choices:
                if ch['target']==s['target']:
                    ch.clear();ch.update({k:v for k,v in s.items() if k!='c'},key='v45_'+s['mode'],model_module='divergent_models_v45',
                        architecture='random_full_private_'+s['mode'],input_variant=saved['config']['input_variant'],window_length=model.length,
                        preprocessing='results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',
                        builder='load_data metadata; Network(mode,metadata); load_state_dict(checkpoint[state_dict]); Evaluator with original action grid and model.length')
            replays.append(s['target']);del model,saved;gc.collect();torch.cuda.empty_cache()
        del data,metadata
    scores=metric(truth,combined)
    result=native(dict(reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],
        candidates=records,selections=selected,full_disk_replays=replays,rows=144665,screening_fold=5,strict805_qualified=scores['TOTAL']>80.5,
        actual_fixed_three_column_scoring=True,matched_ablation=False,complete_five_folds=False,remaining_folds_started=False,
        old_deliveries_unchanged=True,finished_at=datetime.now().astimezone().isoformat()))
    atomic_array(STUDY/'assembled_pred.npy',combined);atomic_json(STUDY/'assembled_report.json',result)
    recipe=dict(old);recipe.update(scores=native(scores),choices=choices,reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
    atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    events=[]
    for t in timings:events.extend([(datetime.fromisoformat(t['start']),1),(datetime.fromisoformat(t['end']),-1)])
    live=peak=0
    for _,d in sorted(events):live+=d;peak=max(peak,live)
    timing=dict(candidates=timings,first_launch_to_last_completion_seconds=(max(t for t,_ in events)-min(t for t,_ in events)).total_seconds(),
        peak_process_concurrency=peak,finalization_seconds=time.perf_counter()-start,
        scope='loads/preparation/warmup/training/validation and staged launch gap included; source authoring/final replay separate')
    atomic_json(STUDY/'timing.json',timing)
    lines=['# v45发散新架构：结果','',f"固定原第五折TOTAL {refs['TOTAL']:.12f} → {scores['TOTAL']:.12f}，差值{scores['TOTAL']-refs['TOTAL']:+.12f}。",'',
        '|目标/路线|参数|自身最高分|相对原目标|最佳/停止轮|','|---|---:|---:|---:|---|']
    for row in records:
        r=row['report'];lines.append(f"|{row['target']}/{row['mode']}|{r['trainable_parameters']}|{r['target_score']:.12f}|{r['delta_reference_target']:+.12f}|{r['best_epoch']}/{r['stopped_epoch']}|")
    lines+=['','## 探索重点与已做','SW采用1024点多尺度全局/局部网络；POR采用512点九曲线二维轴向网络和联合遮挡重建。两个独立新预测器均从随机网络训练，不加载旧网络权重或旧预测输入；原64井/Q50/标签变换与官方解码/评分保持。POR九曲线子集为明示新变体。',
        '每项恢复自身验证最佳checkpoint，全144665点单覆盖、源哈希与评分核对通过；新采用目标额外磁盘重建='+str(replays)+'。实际固定三列评分完成，保留旧模型/交付。',
        '', '## 收益与边界','采用='+json.dumps(selected,ensure_ascii=False),
        '本批是两条完整路线，不是同容量单因素消融；不能把分数差单独归因于长窗口、轴向注意力或重建。仍是重复使用的第五开发折，不是完整OOF或独立新井收益。',
        '若路线未超过原目标，保留负结果与学习曲线，下一步依据误差或收敛状态另提假设；不退回连续冻结小头修补，不把从默认先验学会预测的提升算成超过旧基线。',
        '', '## 耗时及未完成',json.dumps(timing,ensure_ascii=False,indent=2),
        '完整原五折OOF>82未完成。本批没有启动其他折或覆盖交付；第五>80.5也需明显收益和后续明确推进才补折。CUDA上采样/注意力反向可能非确定，已记录warn_only，不声称训练逐位可复现。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v45两条新架构完成';note=marker+f"\nSW长井段与POR九曲线轴向完成，实际TOTAL={scores['TOTAL']:.12f}，增量{scores['TOTAL']-refs['TOTAL']:+.12f}，采用{[(s['target'],s['mode']) for s in selected]}。详见divergent_models_v45_20260929/FINAL.md；未补其他折，旧交付不变。"
    for p in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md',STUDY/'README.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):append(p,marker,note)
    print(json.dumps(native(dict(scores=scores,selections=selected,timing=timing))),flush=True)

if __name__=='__main__':main()
