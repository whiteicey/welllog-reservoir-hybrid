"""Restore improved channel heads, score fixed assembly, preserve compact evidence."""
from datetime import datetime
import gc,hashlib,json,time
import numpy as np
import torch
from state_messages_v44 import ROOT,STUDY,PREVIOUS,MODES,TARGETS,Network,Evaluator,load_data,reference_prediction
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
    combined=reference.copy();records=[];chosen=[];contrasts=[];timings=[];choices=[dict(x) for x in old['choices']]
    for t in ('SW',):
        c=TARGETS.index(t);out=ROOT/f'outputs/pure_dl_state_messages_v44_{t}_fold5'
        cfg=read(out/'config.json');r=read(out/'fold_5_report.json');own=[]
        for name,digest in cfg['source_hashes'].items():assert hashlib.sha256((ROOT/'experiments/pure_dl'/name).read_bytes()).hexdigest()==digest,name
        for m in MODES:
            arm=out/m;rep=read(arm/'fold_5_report.json');p=np.load(arm/'channel_pred.npy')
            diag=diagnostics(truth,reference,p,wells,t)
            updated=np.load(arm/'best_updated_pred.npy')
            up=diagnostics(truth,reference,updated,wells,t)
            assert abs(up['scores'][t]-rep['best_after_update']['target_score'])<1e-9
            assert abs(diag['scores'][t]-rep['target_score'])<1e-9
            assert rep['all144665_selected_predictions_restored_exactly'] and rep['all_frozen_state_unchanged']
            if rep['best_epoch']==0:assert np.array_equal(p,reference[:,c])
            assert rep['factor_scope']['conditional_density_exact']
            row=dict(target=t,mode=m,report=rep,diagnostics=diag,post_update=up,
                     post_update_changed_predictions=int((updated!=reference[:,c]).sum()))
            records.append(row);own.append((rep['target_score'],m,p,rep))
        best=max(own,key=lambda x:x[0]);adopt=best[0]>refs[t]+1e-9
        ctrl=reference.copy();ctrl[:,c]=own[0][2]
        contrasts.append(dict(target=t,neighbor_minus_self=diagnostics(truth,ctrl,own[1][2],wells,t),
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
                    choice.update(s,key='v44_'+t+'_'+m,architecture='frozen_SW_state_messages8',model_module='state_messages_v44',
                                  parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),
                                  builder='load_data(); Network(target,master,modes=(mode,)); heads[mode].load_state_dict(delta_state)',additional_parameters=10273)
            replay.append(t);model.close();del model,cp;gc.collect();torch.cuda.empty_cache()
        del data,master
    scores=metric(truth,combined)
    result=native(dict(reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],
                       candidates=records,selections=chosen,contrasts=contrasts,full_disk_replays=replay,
                       actual_fixed_three_column_scoring=True,rows=144665,screening_fold=5,strict805_qualified=scores['TOTAL']>80.5,
                       complete_five_folds=False,remaining_folds_started=False,old_deliveries_unchanged=True,finished_at=datetime.now().astimezone().isoformat()))
    atomic_array(STUDY/'assembled_pred.npy',combined);atomic_json(STUDY/'assembled_report.json',result)
    recipe=dict(old);recipe.update(scores=native(scores),choices=choices,reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
    atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    timing=dict(targets=timings,first_launch_to_last_completion_seconds=(max(datetime.fromisoformat(x['end']) for x in timings)-min(datetime.fromisoformat(x['start']) for x in timings)).total_seconds(),
                finalization_seconds=time.perf_counter()-start,scope='one GPU process contains two independently optimized heads sharing frozen forward; loading/warmup/evaluation included; code authoring excluded')
    atomic_json(STUDY/'timing.json',timing)
    lines=['# v44 SW默认状态消息：最终结果','',f"原第五折TOTAL {refs['TOTAL']:.12f} → {scores['TOTAL']:.12f}；差值 {scores['TOTAL']-refs['TOTAL']:+.12f}。",'',
           '|模式|训练后最高SW|含回退SW|最佳/停止轮|','|---|---:|---:|---|']
    for x in records:
        r=x['report'];lines.append(f"|{x['mode']}|{r['best_after_update']['target_score']:.12f}|{r['target_score']:.12f}|{r['best_epoch']}/{r['stopped_epoch']}|")
    lines+=['','## 探索重点与完成验证','两项同10273参数、初始化、训练窗和门控BCE/Q50，只改变状态消息来源。条件密度和旧参数固定，原9曲线/分折/动作/评分保持。',
            '固定8步消息，不是整井精确CRF。self沿用同一输入条件化边耦合和原上下文骨干，不是完全无上下文模型。',
            '全144665点初始复现、最佳恢复、冻结状态及条件密度检查通过。新采用目标额外磁盘重建：'+str(replay)+'；实际三列评分完成。',
            '', '## 收益与不继续方向']
    for x in contrasts:
        d=x['neighbor_minus_self'];lines.append(f"neighbor-self含回退SW差{d['channel_delta']:+.12f}；训练后最高差{x['updated_only_delta']:+.12f}。")
    for row in records:lines.append(row['mode']+'训练后最佳改变预测点数：'+str(row['post_update_changed_predictions'])+'；梯度/参数变化见报告。')
    lines+=['采用：'+json.dumps(chosen,ensure_ascii=False),
            '第0轮回退不是学习收益。若neighbor不优于self和原模型，不扩展消息步数/耦合界扫描；结论限当前冻结骨干有限半径设置，不泛化为所有状态序列模型无效。原第五折被反复用于开发，尚无独立新井确认。',
            '', '## 耗时与未完成目标',str(timing),
            '完整原五折OOF>82仍未完成；未运行其他折、旧任务或覆盖交付。']
    txt='\n'.join(lines)+'\n';(STUDY/'FINAL.md').write_text(txt,encoding='utf-8')
    append(STUDY/'README.md','## 最终报告','## 最终报告\n见[FINAL.md](FINAL.md)，'+f"TOTAL={scores['TOTAL']:.12f}。")
    marker='## v44默认状态消息完成';msg=marker+f"\n两项完成，TOTAL={scores['TOTAL']:.12f}，相对80.336513增加{scores['TOTAL']-refs['TOTAL']:+.12f}；采用{[(x['target'],x['mode']) for x in chosen]}。完整记录state_messages_v44_20260929/FINAL.md；无其他折或交付改动。"
    for p in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):append(p,marker,msg)
    print(json.dumps(dict(scores=scores,chosen=chosen,contrasts=[dict(target=x['target'],delta=x['neighbor_minus_self']['channel_delta'],updated_delta=x['updated_only_delta']) for x in contrasts],timing=timing)),flush=True)

if __name__=='__main__':main()
