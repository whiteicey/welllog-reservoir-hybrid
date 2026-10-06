"""POR complete same-capacity utility comparison, actual assembly/replay/cleanup."""
from datetime import datetime
import gc,hashlib,json,time
import numpy as np
import torch
from por_full_utility_v92 import ROOT,STUDY,PREVIOUS,MODES,Network,Evaluator,load_data,SOURCE,reference_prediction
from train_dual8222_heads80 import atomic_json,atomic_array
from finish_all_heads_v26 import diagnostics,native
from run_retrieval_smooth_v36 import append,jobs
import train_invariant as base

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def source_proof(cfg):
    for n,h in cfg['source_hashes'].items():assert hashlib.sha256((ROOT/'experiments/pure_dl'/n).read_bytes()).hexdigest()==h,n
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==cfg['preprocessing_sha256']
    for p,h in [('source_checkpoint','source_sha256'),('actual_source_checkpoint','actual_source_sha256')]:
        assert hashlib.sha256((ROOT/cfg['initialization'][p]).read_bytes()).hexdigest()==cfg['initialization'][h]
    assert hashlib.sha256((ROOT/'package/src/utils.py').read_bytes()).hexdigest()==cfg['scoring_sha256']

def main():
    if jobs():raise RuntimeError('Training remains active')
    tick=time.perf_counter();old=read(PREVIOUS/'CURRENT_RECIPE.json');reference=reference_prediction()
    with np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/oof.npz') as z:
        truth=z['truth'][z['fold']==4].copy();wells=z['well'][z['fold']==4].copy()
    assert reference.shape==truth.shape==(144665,3)
    refs=base.metric(truth,reference);assert all(abs(refs[k]-v)<1e-9 for k,v in old['scores'].items())
    rows=[];preds={};updates={};configs=[];times=[];histories={}
    for mode in MODES:
        out=ROOT/f'outputs/pure_dl_por_full_utility_v92_POR_{mode}_fold5';assert read(out/'runtime_status.json')['status']=='complete'
        cfg=read(out/'config.json');rep=read(out/'fold_5_report.json');initial=read(out/'initial.json');source_proof(cfg)
        assert cfg['only_own_target_training_labels'] and cfg['unused_source_readouts_and_anchor_parameters_frozen']
        assert not any('.anchor_' in n or n.startswith('source.') and '.main_head.' in n for n in cfg['trainable_names'])
        assert rep['frozen_packed_rows_unchanged'] and rep['frozen_source_readouts_and_anchor_parameters_unchanged'] and rep['all144665_selected_predictions_restored_exactly']
        assert all(np.isfinite(v) and v>0 for v in rep['gradient_scope']['gradient_norms'].values())
        assert initial['all144665_parent_exact'] and abs(initial['target_score']-refs['POR'])<1e-9
        pred=np.load(out/'channel_pred.npy');up=np.load(out/'best_updated_pred.npy')
        selected=diagnostics(truth,reference,pred,wells,'POR');updated=diagnostics(truth,reference,up,wells,'POR')
        assert abs(selected['scores']['POR']-rep['target_score'])<1e-9 and abs(selected['scores']['TOTAL']-rep['reference_replacement_scores']['TOTAL'])<1e-9
        assert abs(updated['scores']['POR']-rep['best_after_update']['target_score'])<1e-9
        if rep['best_epoch']==0:assert np.array_equal(pred,reference[:,0])
        history=read(out/'validation_history.json');histories[mode]={r['epoch']:r for r in history}
        rows.append(dict(mode=mode,report=rep,against_parent=selected,post_update=updated,initial=initial));preds[mode]=pred;updates[mode]=up;configs.append(cfg)
        times.append(dict(mode=mode,start=read(out/'launch.json')['started_at'],end=rep['finished_at'],seconds=rep['seconds'],peak_allocated_MiB=rep['peak_allocated_MiB'],peak_reserved_MiB=rep['peak_reserved_MiB']))
    for k in configs[0]:
        if k not in ('mode','utility_coefficient','initialization','trainable_names'):assert configs[0][k]==configs[1][k],k
    for k in configs[0]['initialization']:
        if k!='utility_coefficient':assert configs[0]['initialization'][k]==configs[1]['initialization'][k],k
    assert [n.replace('likelihood','MODE') for n in configs[0]['trainable_names']]==[n.replace('joint','MODE') for n in configs[1]['trainable_names']]
    ctrl=reference.copy();ctrl[:,0]=preds['likelihood'];contrast=diagnostics(truth,ctrl,preds['joint'],wells,'POR')
    post_ctrl=reference.copy();post_ctrl[:,0]=updates['likelihood'];post_contrast=diagnostics(truth,post_ctrl,updates['joint'],wells,'POR');post_delta=float(post_contrast['channel_delta'])
    common=sorted(set(histories['likelihood'])&set(histories['joint']));deltas=[histories['joint'][ep]['target_score']-histories['likelihood'][ep]['target_score'] for ep in common]
    trajectory=dict(common_epochs=common,delta_joint_minus_likelihood=deltas,positive_epochs=sum(d>0 for d in deltas),negative_epochs=sum(d<0 for d in deltas),mean_delta=float(np.mean(deltas)),last_delta=float(deltas[-1]))
    best=max(rows,key=lambda r:r['report']['target_score']);adopt=bool(best['report']['target_score']>refs['POR']+1e-9)
    retained={r['mode'] for r in rows if r['report']['target_score']>refs['POR']+1e-9};matched=set(MODES) if post_delta>1e-9 else set()
    combined=reference.copy();choices=[dict(c) for c in old['choices']];proof=[]
    if retained or matched:
        torch.set_num_threads(3);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
        data,master=load_data();assert np.array_equal(truth,data['truth'][data['validation']]) and np.array_equal(wells,data['wells'][data['validation']])
        for mode in sorted(retained|matched):
            out=ROOT/f'outputs/pure_dl_por_full_utility_v92_POR_{mode}_fold5';use_updated=mode in matched
            cpfile=out/('model_best_updated.pt' if use_updated else 'model_best.pt');saved=torch.load(cpfile,map_location='cpu',weights_only=False)
            model=Network(mode,master).cuda();model.load_state_dict(saved['state_dict'],strict=True)
            fresh,score,_=Evaluator(data,master,0,reference)(model);expected=updates[mode] if use_updated else preds[mode]
            assert np.array_equal(fresh,expected);proof.append(dict(mode=mode,updated_checkpoint=use_updated,all144665_fresh_disk_replay_exact=True,score=score,sha256=hashlib.sha256(cpfile.read_bytes()).hexdigest()))
            if adopt and mode==best['mode']:
                assert np.array_equal(fresh,preds[mode]);combined[:,0]=fresh
                choices[0].update(key='v92_POR_'+mode,mode=mode,model_module='por_full_utility_v92',checkpoint=str((out/'model_best.pt').relative_to(ROOT)),target_score=score,best_epoch=best['report']['best_epoch'],parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),builder='load_data; Network(mode,master); load_state_dict(saved[state_dict]); original POR Evaluator')
            model.close();del model,saved;gc.collect();torch.cuda.empty_cache()
        del data,master;gc.collect();torch.cuda.empty_cache()
    scores=base.metric(truth,combined);assert np.array_equal(combined[:,1:],reference[:,1:])
    assert abs(scores['TOTAL']-.3*scores['POR']-.35*scores['PERM']-.35*scores['SW'])<1e-9
    result=native(dict(reference_scores=refs,selected_scores=scores,candidates=rows,joint_minus_likelihood=contrast,post_update_joint_minus_likelihood=post_delta,post_update_contrast=post_contrast,trajectory_diagnostic=trajectory,
        adopted=adopt,selected_mode=best['mode'] if adopt else None,retained=sorted(retained),retained_updated=sorted(matched),verification=proof,actual_fixed_three_column_scoring=True,
        screening_fold=5,rows=144665,strict805_passed=bool(scores['TOTAL']>80.5),complete_five_folds=False,remaining_folds_started=False,finished_at=datetime.now().astimezone().isoformat()))
    atomic_json(STUDY/'assembled_report.json',result);atomic_array(STUDY/'assembled_pred.npy',combined)
    recipe=dict(old);recipe.update(scores=native(scores),choices=choices,reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)));atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    starts=[datetime.fromisoformat(t['start']) for t in times];ends=[datetime.fromisoformat(t['end']) for t in times]
    overlap=max(0.,(min(ends)-max(starts)).total_seconds())
    for t,a,b in zip(times,starts,ends):t['launch_to_completion_seconds']=(b-a).total_seconds()
    timing=dict(candidates=times,first_launch_to_last_completion_seconds=(max(ends)-min(starts)).total_seconds(),first_launch_to_verified_seconds=(datetime.now().astimezone()-min(starts)).total_seconds(),process_overlap_seconds=overlap,peak_training_processes=2 if overlap>0 else 1,sum_individual_launch_seconds=sum(t['launch_to_completion_seconds'] for t in times),finalize_seconds=time.perf_counter()-tick,scope='loading/warmup/train/eval/save and explicit launch gap included; prelaunch authoring/atlas separate; not capacity evidence')
    atomic_json(STUDY/'timing.json',native(timing))
    manifest=STUDY/'deletion_manifest.json';deleted=read(manifest).get('deleted',[]) if manifest.exists() else [];plan=[]
    for mode in MODES:
        out=(ROOT/f'outputs/pure_dl_por_full_utility_v92_POR_{mode}_fold5').resolve();assert out.parent==(ROOT/'outputs').resolve()
        for name in ('model_best.pt','model_best_updated.pt','last_state.pt'):
            if name=='model_best.pt' and mode in retained:continue
            if name=='model_best_updated.pt' and mode in matched:continue
            p=(out/name).resolve();assert p.parent==out
            if p.exists():plan.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    atomic_json(STUDY/'deletion_plan.json',native(dict(planned=plan,retained=sorted(retained),retained_updated=sorted(matched),status='planned_before_unlink')))
    for item in plan:
        p=(ROOT/item['path']).resolve();assert p.parent.parent==(ROOT/'outputs').resolve();p.unlink();deleted.append(item)
        atomic_json(manifest,native(dict(deleted=deleted,status='in_progress')))
    atomic_json(manifest,native(dict(deleted=deleted,bytes=sum(x['bytes'] for x in deleted),retained=sorted(retained),retained_updated=sorted(matched),status='complete')))
    lines=['# v92 POR完整私有模型动作效用结果','',f"当前独立路线TOTAL {refs['TOTAL']:.12f} → {scores['TOTAL']:.12f}。",'','|方案|POR|固定PERM/SW TOTAL|最佳/停止轮|训练后最高POR|','|---|---:|---:|---:|---:|']
    for row in rows:
        r=row['report'];lines.append(f"|{row['mode']}|{r['target_score']:.12f}|{r['reference_replacement_scores']['TOTAL']:.12f}|{r['best_epoch']}/{r['stopped_epoch']}|{r['best_after_update']['target_score']:.12f}|")
    lines+=['','探索与验证：完整v31 anchored第7轮/v30实际bundle/v25-v8私有POR同起点，共'+str(configs[0]['trainable_parameters'])+'学习参数。解除隐藏/输入路由梯度冻结及旧v25 train override，两骨干/实际输出头/v28-v29更新，未使用源输出副本/route anchor/其他目标packed行固定。原9曲线206+已有幅值63/分折/Q50/t3/log1p-expm1/官方.08/default.1/256支持800均匀动作不改；仅系数0/1，新body1e-5/head3e-5/温度.05/80上限pat18。真实梯度、初始全预测、source/预处理/评分hash、frozen权重/行不变、最佳144665恢复及三列评价通过。原始输入磁盘重载='+str(proof)+'；CUDAwarn_only不声称逐位训练复现。', '',f"收益：含起点joint-likelihood POR{contrast['channel_delta']:+.12f}，完整排除起点的训练后最佳差{post_delta:+.12f}。采用={result['selected_mode']}。epoch0不是新收益；同容量局部对照和绝对采用分开。",'同轮轨迹：'+json.dumps(native(trajectory),ensure_ascii=False)]
    for row in rows:lines.append(row['mode']+' 更新后默认/数值/各井：'+json.dumps(native(row['post_update']['populations']),ensure_ascii=False)+f"；井升/降{row['post_update']['well_wins']}/{row['post_update']['well_losses']}。")
    lines+=['不继续：无绝对或完整匹配收益则关闭本完整POR/v31来源/t3/低LR/系数1/温度.05/预算组合，不临时扫参数，不泛化所有POR效用方法。开发第五折不是完整OOF或独立泛化证明。', '', '成本：'+json.dumps(native(timing),ensure_ascii=False),f"清理{len(deleted)}个冗余状态{sum(x['bytes'] for x in deleted)}字节，清理计划先落盘；正源/当前/匹配依赖和旧交付保留。",'已做：完整两项实测/最佳恢复/实际组合/必要重载/清理/记录。未做：其他折、独立泛化。第五严格>80.5且明确收益后才考虑其余四折，完整原OOF>82未完成；目标后独立井确认与新目录交付。']
    text='\n'.join(lines)+'\n';(STUDY/'FINAL.md').write_text(text,encoding='utf-8');(STUDY/'PROJECTS.md').write_text(text,encoding='utf-8');append(STUDY/'README.md','## 最终结果','## 最终结果\n'+text)
    for mode in MODES:
        out=ROOT/f'outputs/pure_dl_por_full_utility_v92_POR_{mode}_fold5';(out/'PROJECTS.md').write_text(text,encoding='utf-8');append(out/'README.md','## 最终结果','## 最终结果\n'+text)
    note=f"## v92 POR完整网络效用完成\n当前TOTAL{scores['TOTAL']:.12f}，采用={result['selected_mode']}；完整训练后joint-likelihood POR{post_delta:+.12f}，同轮均值{trajectory['mean_delta']:+.12f}。配方results/deep_learning/por_full_utility_v92_20260930/CURRENT_RECIPE.json；无其他折，原OOF>82未完成。"
    for p in (ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):append(p,'## v92 POR完整网络效用完成',note)
    for p in (ROOT/'AGENTS.md',ROOT.parent/'AGENTS.md'):append(p,'## v92当前研究配方',note.replace('## v92 POR完整网络效用完成','## v92当前研究配方'))
    append(ROOT/'EXPLORATIONS.md','v92完成：','- v92完成：[结果](results/deep_learning/por_full_utility_v92_20260930/FINAL.md)，'+note)
    print(json.dumps(native(dict(scores=scores,selected_mode=result['selected_mode'],updated_delta=post_delta,trajectory=trajectory,retained=sorted(retained),retained_updated=sorted(matched),deleted_bytes=sum(x['bytes'] for x in deleted),timing=timing)),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
