"""Finalize the two matched SW warm-start trials; keep real gains, not duplicate parents."""
import gc,hashlib,json,time
from datetime import datetime
import numpy as np
import torch
from sw_input_products_v79 import ROOT,STUDY,PREVIOUS,MODES,SOURCE,Network,Evaluator,load_data,reference_prediction
from run_retrieval_smooth_v36 import jobs,append
from finish_all_heads_v26 import diagnostics,native
from train_dual8222_heads80 import atomic_json,atomic_array
from train import metric

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def main():
    if jobs():raise RuntimeError('Training still active')
    tick=time.perf_counter();old=read(PREVIOUS/'CURRENT_RECIPE.json');reference=reference_prediction()
    with np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/oof.npz') as z:
        truth=z['truth'][z['fold']==4].copy();wells=z['well'][z['fold']==4].copy()
    assert truth.shape==reference.shape==(144665,3)
    refs=metric(truth,reference);assert all(abs(refs[k]-v)<1e-9 for k,v in old['scores'].items())
    rows=[];preds={};configs=[];hashes=[];times=[]
    for mode in MODES:
        out=ROOT/f'outputs/pure_dl_sw_input_products_v79_SW_{mode}_fold5'
        cfg=read(out/'config.json');rep=read(out/'fold_5_report.json');initial=read(out/'initial.json')
        assert read(out/'runtime_status.json')['status']=='complete'
        for name,digest in cfg['source_hashes'].items():
            assert hashlib.sha256((ROOT/'experiments/pure_dl'/name).read_bytes()).hexdigest()==digest,name
        assert cfg['initialization']['sha256']==hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        assert initial['all144665_parent_exact'] and not initial['initial_head_prior_only']
        assert initial['target_score']==refs['SW'] and rep['all144665_selected_predictions_restored_exactly']
        assert rep['all_original_weights_frozen_exactly'] and rep['gradient_scope']['frozen_gradients_absent']
        assert cfg['trainable_parameters']==4320 and cfg['total_parameters']==843801
        assert len(cfg['bank_manifest']['train_wells'])==64 and not cfg['bank_manifest']['validation_fit']
        assert all(v>0 for v in rep['gradient_scope']['gradient_norms'].values())
        pred=np.load(out/'channel_pred.npy');diag=diagnostics(truth,reference,pred,wells,'SW')
        assert abs(diag['scores']['SW']-rep['target_score'])<1e-9
        history=read(out/'validation_history.json')
        rows.append(dict(mode=mode,report=rep,initial=initial,diagnostics=diag,best_after_update=max(history,key=lambda x:x['target_score'])))
        preds[mode]=pred;configs.append(cfg);hashes.append((cfg['initial_state_sha256'],initial['initial_prediction_sha256']))
        launch=read(out/'launch.json');times.append(dict(mode=mode,start=launch['started_at'],end=rep['finished_at'],seconds=rep['seconds'],peak_allocated_MiB=rep['peak_allocated_MiB']))
    assert hashes[0]==hashes[1]
    assert all(k in ('mode','input_adapter_view','initialization') or configs[0][k]==configs[1][k] for k in configs[0])
    marginal=reference.copy();marginal[:,2]=preds['marginal']
    contrast=diagnostics(truth,marginal,preds['cross'],wells,'SW')
    best=max(rows,key=lambda row:row['report']['target_score']);adopt=best['report']['target_score']>refs['SW']+1e-9
    retained={row['mode'] for row in rows if row['report']['target_score']>refs['SW']+1e-9}
    if contrast['channel_delta']>1e-9 and rows[1]['report']['best_epoch']>0:retained.add('cross')
    combined=reference.copy();choices=[dict(c) for c in old['choices']];proof=[]
    if retained:
        torch.set_num_threads(3);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
        data,meta=load_data()
        for mode in sorted(retained):
            rep=next(row['report'] for row in rows if row['mode']==mode);file=ROOT/rep['checkpoint']
            model=Network(mode,meta).cuda();saved=torch.load(file,map_location='cpu',weights_only=False)
            model.load_state_dict(saved['state_dict'],strict=True);pred,score,_=Evaluator(data,meta,2,reference)(model)
            assert np.array_equal(pred,preds[mode]) and abs(score-rep['target_score'])<1e-9
            proof.append(dict(mode=mode,raw_disk_replay_exact=True,rows=len(pred),checkpoint_sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
            if adopt and mode==best['mode']:
                combined[:,2]=pred
                choice=dict(target='SW',mode=mode,key='v79_SW_'+mode,model_module='sw_input_products_v79',checkpoint=rep['checkpoint'],target_score=score,best_epoch=rep['best_epoch'],builder='load_data; Network(mode,metadata); saved state_dict; original Evaluator; no inference dropout',parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
                choices=[choice if c['target']=='SW' else c for c in choices]
            del model,saved;gc.collect();torch.cuda.empty_cache()
    scores=metric(truth,combined);finished=datetime.now().astimezone()
    result=native(dict(reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],candidates=rows,
        cross_minus_marginal=contrast,post_update_cross_minus_marginal=rows[1]['best_after_update']['target_score']-rows[0]['best_after_update']['target_score'],
        adopted=adopt,selected_mode=best['mode'] if adopt else None,retained=sorted(retained),verification=proof,initial_state_and_prediction_equal=True,
        actual_fixed_three_column_scoring=True,screening_fold=5,rows=144665,strict805_qualified=scores['TOTAL']>80.5,
        complete_five_folds=False,remaining_folds_started=False,finished_at=finished.isoformat()))
    atomic_array(STUDY/'assembled_pred.npy',combined);atomic_json(STUDY/'assembled_report.json',result)
    recipe=dict(old);recipe.update(scores=native(scores),choices=choices,reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
    atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    events=[]
    for item in times:events.extend([(datetime.fromisoformat(item['start']),1),(datetime.fromisoformat(item['end']),-1)])
    live=peak=0
    for _,change in sorted(events):live+=change;peak=max(peak,live)
    timing=dict(candidates=times,first_launch_to_last_completion_seconds=(max(t for t,_ in events)-min(t for t,_ in events)).total_seconds(),
        first_launch_to_verified_wall_seconds=(finished-min(t for t,_ in events)).total_seconds(),
        last_training_completion_to_verified_wall_seconds=(finished-max(t for t,_ in events)).total_seconds(),
        peak_process_concurrency=peak,finalization_seconds=time.perf_counter()-tick,
        scope='loading/preprocessing/warmup/training/evaluation/persistence/launch gap included; authoring before launch and atlas after verification separate')
    atomic_json(STUDY/'timing.json',timing)
    deleted=[];manifest=STUDY/'deletion_manifest.json'
    if manifest.exists():deleted=read(manifest)['deleted']
    for mode in MODES:
        out=(ROOT/f'outputs/pure_dl_sw_input_products_v79_SW_{mode}_fold5').resolve()
        assert out.parent==(ROOT/'outputs').resolve() and read(out/'runtime_status.json')['status']=='complete'
        if mode not in retained:
            for name in ('model_best.pt','last_state.pt'):
                file=(out/name).resolve();assert file.parent==out
                if file.exists():
                    deleted.append(dict(path=str(file.relative_to(ROOT)),bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
                    atomic_json(manifest,dict(deleted=deleted,status='planned'));file.unlink()
        atomic_json(out/'candidate_status.json',dict(retained=mode in retained,no_automatic_resume=True,reason='demonstrated absolute or matched gain' if mode in retained else 'no new gain; retained complete v77 parent preserved'))
    atomic_json(manifest,dict(deleted=deleted,bytes=sum(x['bytes'] for x in deleted),status='complete',retained=sorted(retained)))
    lines=['# v79 SW输入曲线交互结果','',f"当前独立通道路线第五折TOTAL {refs['TOTAL']:.12f} → {scores['TOTAL']:.12f}。",'',
        '|模式|完整起点SW|训练后最高SW|含起点最佳SW|最佳/停止轮|','|---|---:|---:|---:|---|']
    for row in rows:
        rep=row['report'];lines.append(f"|{row['mode']}|{row['initial']['target_score']:.12f}|{row['best_after_update']['target_score']:.12f}|{rep['target_score']:.12f}|{rep['best_epoch']}/{rep['stopped_epoch']}|")
    lines+=['','## 探索与验证','完整v77 p40函数同起点，冻结839481旧参数且eval，新增4320参数的零初始化45→96有界stem适配器；仅单曲线非线性或显式跨曲线乘积基不同。新AdamW3e-4/80cosine，原输入/数据/Q50/损失/解码保持。源码哈希、初始张量/预测、实际新层梯度/旧层无梯度、原权重逐位冻结和最佳恢复通过；不单独重复预检。',
        '保留正候选的原始输入磁盘重建：'+str(proof),
        '', '## 实际收益与不继续',f"cross-marginal 含起点差{contrast['channel_delta']:+.12f}，训练后最高差{result['post_update_cross_minus_marginal']:+.12f}；绝对和机制收益分别报告，采用={result['selected_mode']}，保留={sorted(retained)}。",
        f"配对子群默认{contrast['populations']['default']['channel_delta']:+.12f}，数值{contrast['populations']['numeric']['channel_delta']:+.12f}，井胜/负{contrast['well_wins']}/{contrast['well_losses']}；开发折选择不当独立泛化证明。",
        '若cross不优于同批marginal，不继续当前乘积基阶数/组合细扫；两项均无绝对收益也不继续本stem适配器幅度/容量扫描。只限此输入基与连接位置，不泛化全部特征工程。',
        '', '## 时间、清理与未完成',json.dumps(timing,ensure_ascii=False,indent=2),f"删除{len(deleted)}无新增收益/重复父模型checkpoint，共{sum(x['bytes'] for x in deleted)}字节；清单先写后删，原v77和历史正候选/交付保留。",
        'max80/pat18/min10，实际最佳/停止轮见表。无其他折/旧队列，严格第五>80.5及完整OOF>82目标仍按实际结果判定，未完成不宣称达成。目标后独立井确认与新目录交付。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v79最终结果';note=marker+f"\n两项完成，当前独立通道路线TOTAL={scores['TOTAL']:.12f}，相对起点{scores['TOTAL']-refs['TOTAL']:+.12f}；cross-marginal SW={contrast['channel_delta']:+.12f}，训练后差{result['post_update_cross_minus_marginal']:+.12f}。采用={result['selected_mode']}，保留={sorted(retained)}；负或重复权重清理，原来源/交付保留，无其他折。完整OOF>82未完成。见sw_input_products_v79_20260930/FINAL.md。"
    for path in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md',STUDY/'README.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):append(path,marker,note)
    for mode in MODES:
        for name in ('PROJECTS.md','README.md'):append(ROOT/f'outputs/pure_dl_sw_input_products_v79_SW_{mode}_fold5'/name,marker,note)
    append(ROOT.parent/'AGENTS.md','## v79完成状态',note.replace(marker,'## v79完成状态'))
    append(ROOT/'EXPLORATIONS.md','v79完成：','- v79完成：'+note+' [结果](results/deep_learning/sw_input_products_v79_20260930/FINAL.md)')
    print(json.dumps(result|dict(candidates='see assembled_report.json',timing=timing)),flush=True)

if __name__=='__main__':main()
