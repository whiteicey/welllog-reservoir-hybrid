"""Finalize the matched two-pass dropout-scope pair and retain only demonstrated gains."""
import gc,hashlib,json,time
from datetime import datetime
import numpy as np
import torch
from sw_dropout_scope_v75 import ROOT,STUDY,PREVIOUS,MODES,Network,Evaluator,load_data,reference_prediction
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
    rows=[];preds={};times=[];hashes=[]
    for mode in MODES:
        out=ROOT/f'outputs/pure_dl_sw_dropout_scope_v75_SW_{mode}_fold5'
        cfg=read(out/'config.json');rep=read(out/'fold_5_report.json');initial=read(out/'initial.json')
        for n,h in cfg['source_hashes'].items():assert hashlib.sha256((ROOT/'experiments/pure_dl'/n).read_bytes()).hexdigest()==h,n
        assert len(cfg['bank_manifest']['train_wells'])==64 and not cfg['bank_manifest']['validation_fit']
        assert initial['initial_head_prior_only'] and not initial['all144665_parent_exact'] and rep['all144665_selected_predictions_restored_exactly']
        assert all(v>0 for v in rep['gradient_scope']['gradient_norms'].values())
        pred=np.load(out/'channel_pred.npy');diag=diagnostics(truth,reference,pred,wells,'SW')
        assert abs(diag['scores']['SW']-rep['target_score'])<1e-9
        history=read(out/'validation_history.json')
        rows.append(dict(mode=mode,report=rep,initial=initial,diagnostics=diag,best_after_update=max(history,key=lambda x:x['target_score'])))
        hashes.append((cfg['initial_state_sha256'],initial['initial_prediction_sha256']));preds[mode]=pred
        launch=read(out/'launch.json');times.append(dict(mode=mode,start=launch['started_at'],end=rep['finished_at'],seconds=rep['seconds'],peak_allocated_MiB=rep['peak_allocated_MiB']))
    assert hashes[0]==hashes[1]
    best=max(rows,key=lambda x:x['report']['target_score']);adopt=best['report']['target_score']>refs['SW']+1e-9
    control=reference.copy();control[:,2]=preds['both'];contrast=diagnostics(truth,control,preds['gate_only'],wells,'SW')
    retained={best['mode']} if adopt else set()
    if contrast['channel_delta']>0:retained.add('gate_only')
    combined=reference.copy();choices=[dict(c) for c in old['choices']];verification=[]
    if retained:
        torch.set_num_threads(3);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
        data,meta=load_data()
        for mode in sorted(retained):
            rep=next(r['report'] for r in rows if r['mode']==mode);file=ROOT/rep['checkpoint']
            model=Network(mode,meta).cuda();saved=torch.load(file,map_location='cpu',weights_only=False);model.load_state_dict(saved['state_dict'],strict=True)
            pred,score,_=Evaluator(data,meta,2,reference)(model)
            assert np.array_equal(pred,preds[mode]) and abs(score-rep['target_score'])<1e-9
            verification.append(dict(mode=mode,rows=len(pred),raw_disk_replay_exact=True,checkpoint_sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
            if adopt and mode==best['mode']:
                combined[:,2]=pred
                choice=dict(target='SW',mode=mode,key='v75_SW_'+mode,model_module='sw_dropout_scope_v75',checkpoint=rep['checkpoint'],target_score=score,best_epoch=rep['best_epoch'],builder='load_data; Network(mode,metadata); saved state_dict; Evaluator c=2; unchanged Gaussian decoding',parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
                choices=[choice if c['target']=='SW' else c for c in choices]
            del model,saved;gc.collect();torch.cuda.empty_cache()
    scores=metric(truth,combined)
    result=native(dict(reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],candidates=rows,adopted=adopt,selected_mode=best['mode'] if adopt else None,
        gate_only_minus_both=contrast,post_update_gate_only_minus_both=rows[1]['best_after_update']['target_score']-rows[0]['best_after_update']['target_score'],
        initial_state_and_prediction_equal=True,retained=sorted(retained),verification=verification,actual_fixed_three_column_scoring=True,rows=144665,screening_fold=5,
        strict805_qualified=scores['TOTAL']>80.5,complete_five_folds=False,remaining_folds_started=False,finished_at=datetime.now().astimezone().isoformat()))
    atomic_array(STUDY/'assembled_pred.npy',combined);atomic_json(STUDY/'assembled_report.json',result)
    recipe=dict(old);recipe.update(scores=native(scores),choices=choices,reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)));atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    events=[]
    for t in times:events.extend([(datetime.fromisoformat(t['start']),1),(datetime.fromisoformat(t['end']),-1)])
    live=peak=0
    for _,d in sorted(events):live+=d;peak=max(peak,live)
    timing=dict(candidates=times,first_launch_to_last_completion_seconds=(max(t for t,_ in events)-min(t for t,_ in events)).total_seconds(),peak_process_concurrency=peak,finalization_seconds=time.perf_counter()-start,scope='loading/preprocessing/warmup/training/evaluation/launch gap included; authoring and finalization separate')
    atomic_json(STUDY/'timing.json',timing)
    deleted=[];manifest=STUDY/'deletion_manifest.json'
    if manifest.exists():deleted=read(manifest)['deleted']
    for mode in MODES:
        out=(ROOT/f'outputs/pure_dl_sw_dropout_scope_v75_SW_{mode}_fold5').resolve()
        assert out.parent==(ROOT/'outputs').resolve() and read(out/'runtime_status.json')['status']=='complete'
        if mode not in retained:
            for name in ('model_best.pt','last_state.pt'):
                file=(out/name).resolve();assert file.parent==out
                if file.exists():
                    deleted.append(dict(path=str(file.relative_to(ROOT)),bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
                    atomic_json(manifest,dict(deleted=deleted,status='planned'));file.unlink()
        atomic_json(out/'candidate_status.json',dict(retained=mode in retained,no_automatic_resume=True,reason='absolute or matched gain' if mode in retained else 'no demonstrated gain; original v56 preserved'))
    atomic_json(manifest,dict(deleted=deleted,bytes=sum(x['bytes'] for x in deleted),status='complete',retained=sorted(retained)))
    lines=['# v75 SW失活监督范围结果','',f"固定第五折TOTAL {refs['TOTAL']:.12f} → {scores['TOTAL']:.12f}。",'',
        '|模式|初始SW|训练后最高SW|含起点SW|最佳/停止轮|','|---|---:|---:|---:|---|']
    for row in rows:
        r=row['report'];lines.append(f"|{row['mode']}|{row['initial']['target_score']:.12f}|{row['best_after_update']['target_score']:.12f}|{r['target_score']:.12f}|{r['best_epoch']}/{r['stopped_epoch']}|")
    lines+=['','## 探索与验证','同v56 additive第11轮骨干及FiLM，输出重置训练先验，同839481参数与完整初始函数。两组同两次前向/图反向和随机消耗：门控BCE均读取dropkey10%，密度NLL分别读dropkey或完整上下文。共享骨干仍有梯度耦合，不宣称门控/密度完全独立；推理仅一次完整上下文。新1e-4训练，原数据/Q50/损失定义/解码保持。',
        '同初始张量与144665预测、实际骨干/FF/FiLM/输出梯度、最佳全点恢复、源码哈希、真实三列评分已核验。正模型原始输入磁盘重建：'+str(verification),
        '', '## 收益与不继续',f"gate_only-both含起点差{contrast['channel_delta']:+.12f}；训练后最高差{result['post_update_gate_only_minus_both']:+.12f}。采用={result['selected_mode']}；保留={sorted(retained)}。",
        f"子群实测：默认SW差{contrast['populations']['default']['channel_delta']:+.12f}，数值SW差{contrast['populations']['numeric']['channel_delta']:+.12f}，井胜/负{contrast['well_wins']}/{contrast['well_losses']}。整体配对收益不代表数值预测也改善，不把开发折调参诊断当独立显著性证明。",
        '无最终配对收益则停止本两图损失混合比例/监督范围扫描，v74正机制保留，不泛化全部目标解耦。局部收益与当前冠军绝对收益区分，反复开发折不是独立泛化证明。',
        '', '## 耗时与未完成',json.dumps(timing,ensure_ascii=False,indent=2),f"删除{len(deleted)}负检查点共{sum(x['bytes'] for x in deleted)}字节，清单先写后删，原v56正控制和交付不改。",
        'max80/pat18，早停实际轮数见表；未启动其他折/旧队列。严格第五>80.5与完整原五折OOF>82仍待达成；后续换思路，目标完成后独立井验证和新目录交付。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v75最终结果';note=marker+f"\n两项完成，TOTAL={scores['TOTAL']:.12f}；gate_only-both SW={contrast['channel_delta']:+.12f}，训练后差={result['post_update_gate_only_minus_both']:+.12f}；采用={result['selected_mode']}，保留={sorted(retained)}。负权重已清理，原模型/交付保持，无其他折，目标未完成。详见sw_dropout_scope_v75_20260929/FINAL.md。"
    for path in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md',STUDY/'README.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):append(path,marker,note)
    for mode in MODES:
        for name in ('PROJECTS.md','README.md'):append(ROOT/f'outputs/pure_dl_sw_dropout_scope_v75_SW_{mode}_fold5'/name,marker,note)
    append(ROOT.parent/'AGENTS.md','## v75完成状态',note.replace(marker,'## v75完成状态'))
    append(ROOT/'EXPLORATIONS.md','v75完成：','- v75完成：'+note+' [结果](results/deep_learning/sw_dropout_scope_v75_20260929/FINAL.md)')
    print(json.dumps(result|dict(candidates='see assembled_report.json',timing=timing)),flush=True)

if __name__=='__main__':main()
