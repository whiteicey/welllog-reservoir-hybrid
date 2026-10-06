"""Finalize the two PERM candidates on the original fixed three-column population."""
from datetime import datetime
import gc,hashlib,json,time
import numpy as np
import torch
from perm_full_utility_v87 import ROOT,STUDY,PREVIOUS,MODES,Network,Evaluator,load_data,SOURCE,reference_prediction
from train_dual8222_heads80 import atomic_json,atomic_array
from finish_all_heads_v26 import diagnostics,native
from run_retrieval_smooth_v36 import append,jobs
import train_invariant as base

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    if jobs():raise RuntimeError('Training remains active')
    tick=time.perf_counter();reference=reference_prediction();old=read(PREVIOUS/'CURRENT_RECIPE.json')
    with np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/oof.npz') as z:
        truth=z['truth'][z['fold']==4].copy();wells=z['well'][z['fold']==4].copy()
    assert reference.shape==truth.shape==(144665,3)
    refs=base.metric(truth,reference);assert all(abs(refs[k]-v)<1e-9 for k,v in old['scores'].items())
    rows=[];preds={};configs=[];times=[]
    for mode in MODES:
        out=ROOT/f'outputs/pure_dl_perm_full_utility_v87_PERM_{mode}_fold5'
        assert read(out/'runtime_status.json')['status']=='complete'
        cfg=read(out/'config.json');rep=read(out/'fold_5_report.json');initial=read(out/'initial.json')
        for name,digest in cfg['source_hashes'].items():
            assert hashlib.sha256((ROOT/'experiments/pure_dl'/name).read_bytes()).hexdigest()==digest,name
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==cfg['preprocessing_sha256']
        assert hashlib.sha256((ROOT/cfg['initialization']['source_checkpoint']).read_bytes()).hexdigest()==cfg['initialization']['source_sha256']
        assert hashlib.sha256((ROOT/'package/src/utils.py').read_bytes()).hexdigest()==cfg['scoring_sha256']
        assert cfg['trainable_parameters']==1324879 and cfg['only_own_target_training_labels']
        assert rep['frozen_packed_rows_unchanged'] and rep['all144665_selected_predictions_restored_exactly']
        assert all(np.isfinite(v) and v>0 for v in rep['gradient_scope']['gradient_norms'].values())
        assert initial['all144665_parent_exact'] and abs(initial['target_score']-refs['PERM'])<1e-9
        pred=np.load(out/'channel_pred.npy');d=diagnostics(truth,reference,pred,wells,'PERM')
        assert abs(d['scores']['PERM']-rep['target_score'])<1e-9
        assert abs(d['scores']['TOTAL']-rep['reference_replacement_scores']['TOTAL'])<1e-9
        if rep['best_epoch']==0:assert np.array_equal(pred,reference[:,1])
        hist=read(out/'validation_history.json');best_update=max(hist,key=lambda r:r['target_score'])
        rows.append(dict(mode=mode,report=rep,against_parent=d,initial=initial,best_after_update=best_update,
            history_last=hist[-1],checkpoint=str((out/'model_best.pt').relative_to(ROOT))))
        preds[mode]=pred;configs.append(cfg)
        times.append(dict(mode=mode,start=read(out/'launch.json')['started_at'],end=rep['finished_at'],seconds=rep['seconds'],
            peak_allocated_MiB=rep['peak_allocated_MiB'],peak_reserved_MiB=rep['peak_reserved_MiB']))
    assert configs[0]['initial_state_sha256']==configs[1]['initial_state_sha256']
    for key in configs[0]:
        if key not in ('mode','utility_coefficient','initialization'):assert configs[0][key]==configs[1][key],key
    for key in configs[0]['initialization']:
        if key!='utility_coefficient':assert configs[0]['initialization'][key]==configs[1]['initialization'][key],key
    ctrl=reference.copy();ctrl[:,1]=preds['likelihood'];contrast=diagnostics(truth,ctrl,preds['joint'],wells,'PERM')
    updated_delta=rows[1]['best_after_update']['target_score']-rows[0]['best_after_update']['target_score']
    best=max(rows,key=lambda r:r['report']['target_score']);adopt=best['report']['target_score']>refs['PERM']+1e-9
    retained={r['mode'] for r in rows if r['report']['target_score']>refs['PERM']+1e-9}
    if contrast['channel_delta']>1e-9:retained.add('joint')
    combined=reference.copy();choices=[dict(c) for c in old['choices']];proof=[]
    torch.set_num_threads(3);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
    for mode in sorted(retained):
        data,master=load_data();assert np.array_equal(truth,data['truth'][data['validation']]) and np.array_equal(wells,data['wells'][data['validation']])
        out=ROOT/f'outputs/pure_dl_perm_full_utility_v87_PERM_{mode}_fold5'
        cp=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False);model=Network(mode,master).cuda()
        model.load_state_dict(cp['state_dict'],strict=True);pred,score,_=Evaluator(data,master,1,reference)(model)
        assert np.array_equal(pred,preds[mode]);proof.append(dict(mode=mode,all144665_fresh_disk_replay_exact=True,score=score))
        if adopt and mode==best['mode']:
            combined[:,1]=pred
            choices[1].update(key='v87_PERM_'+mode,mode=mode,model_module='perm_full_utility_v87',checkpoint=str((out/'model_best.pt').relative_to(ROOT)),
                target_score=score,best_epoch=cp['best_epoch'],parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),
                builder='load_data; Network(mode,master); load_state_dict(checkpoint[state_dict]); original Evaluator')
        model.close();del model,cp,data,master;gc.collect();torch.cuda.empty_cache()
    scores=base.metric(truth,combined)
    assert np.array_equal(combined[:,[0,2]],reference[:,[0,2]])
    assert abs(scores['TOTAL']-.3*scores['POR']-.35*scores['PERM']-.35*scores['SW'])<1e-9
    deleted=[]
    for mode in MODES:
        out=(ROOT/f'outputs/pure_dl_perm_full_utility_v87_PERM_{mode}_fold5').resolve()
        assert out.parent==(ROOT/'outputs').resolve()
        for name in ('model_best.pt','last_state.pt'):
            if mode in retained:continue
            p=(out/name).resolve();assert p.parent==out
            if p.exists():
                deleted.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()));p.unlink()
    atomic_json(STUDY/'deletion_manifest.json',dict(deleted=deleted,bytes=sum(r['bytes'] for r in deleted),retained=sorted(retained),status='complete'))
    starts=[datetime.fromisoformat(t['start']) for t in times];ends=[datetime.fromisoformat(t['end']) for t in times]
    overlap=max(0.,(min(ends)-max(starts)).total_seconds())
    for t,a,b in zip(times,starts,ends):t['launch_to_completion_seconds']=(b-a).total_seconds()
    timing=dict(candidates=times,first_launch_to_last_completion_seconds=(max(ends)-min(starts)).total_seconds(),
        first_launch_to_verified_seconds=(datetime.now().astimezone()-min(starts)).total_seconds(),process_overlap_seconds=overlap,
        peak_training_processes=2 if overlap>0 else 1,sum_individual_launch_seconds=sum(t['launch_to_completion_seconds'] for t in times),
        finalize_seconds=time.perf_counter()-tick,scope='input loading/warmup/train/eval/save and explicit launch gap included; prelaunch authoring and atlas separate; not a capacity benchmark')
    atomic_json(STUDY/'timing.json',native(timing))
    result=native(dict(reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],
        candidates=rows,joint_minus_likelihood=contrast,post_update_joint_minus_likelihood=updated_delta,
        adopted=adopt,selected_mode=best['mode'] if adopt else None,retained=sorted(retained),verification=proof,
        actual_fixed_three_column_scoring=True,screening_fold=5,rows=144665,strict805_passed=scores['TOTAL']>80.5,
        complete_five_folds=False,remaining_folds_started=False,finished_at=datetime.now().astimezone().isoformat()))
    atomic_json(STUDY/'assembled_report.json',result);atomic_array(STUDY/'assembled_pred.npy',combined)
    recipe=dict(old);recipe.update(scores=native(scores),choices=choices,reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
    atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    lines=['# v87 PERM完整网络联合效用结果','',f"独立路线TOTAL {refs['TOTAL']:.12f} → {scores['TOTAL']:.12f}。",'',
        '|方案|PERM|固定POR/SW TOTAL|最佳/停止轮|更新后最高PERM|','|---|---:|---:|---:|---:|']
    for r in rows:
        p=r['report'];lines.append(f"|{r['mode']}|{p['target_score']:.12f}|{p['reference_replacement_scores']['TOTAL']:.12f}|{p['best_epoch']}/{p['stopped_epoch']}|{r['best_after_update']['target_score']:.12f}|")
    lines+=['','## 探索与验证','同1324879学习参数、完整当前函数同起点；原骨干/输入路由/本目标头和小修正均实际更新，其他目标packed行仍是buffer。',
        '原log10 Gaussian/默认0.01/官方log误差/原支持动作/分折/Q50不改；新body1e-5、head3e-5、80/18、温度.05效用0/1。实际梯度、初始144665一致、source/预处理/评分hash、buffer、最佳恢复、原行序和实际三列评价通过；正候选重载='+str(proof)+'。',
        'CUDA线性上采样backward只支持warn_only确定性模式，不宣称逐位训练复现。','',
        '## 收益与不继续',f"最终joint-likelihood PERM {contrast['channel_delta']:+.12f}；排除起点的更新后最高差 {updated_delta:+.12f}（诊断独立报告，非当前采用收益）。采用={result['selected_mode']}。恢复第0轮不算新增收益。"]
    for r in rows:lines.append(r['mode']+' 默认/数值差='+json.dumps(native(r['against_parent']['populations']),ensure_ascii=False)+'；实测停止='+str(r['report']['stop_reason']))
    lines+=['无绝对或匹配最终收益时关闭本源/全网络/低LR/系数1/温度.05组合，不继续临时扫效用或延长预算；结论不泛化所有PERM训练。',
        '一个反复开发的第五折不是完整OOF或独立泛化证明。','', '## 成本、清理、未完成与目标后',json.dumps(native(timing),ensure_ascii=False,indent=2),
        f"删除{len(deleted)}个冗余权重共{sum(r['bytes'] for r in deleted)}字节，正源和旧交付不改。",
        '未运行其他折、旧队列。仍需原第五严格>80.5且明确收益再考虑其余四折，完整原OOF>82未完成；目标后独立井确认与新目录交付。']
    final='\n'.join(lines)+'\n';(STUDY/'FINAL.md').write_text(final,encoding='utf-8')
    for p in (STUDY/'README.md',):append(p,'## 最终结果','## 最终结果\n'+final)
    progress='目标：纯DL原完整OOF>82。\n已做：两项实测、完整更新、恢复、实际组合、必要重载、清理。\n计划/未做：转向下一明确机制；原其他折与独立泛化未做。\n'+final
    (STUDY/'PROJECTS.md').write_text(progress,encoding='utf-8')
    for mode in MODES:
        out=ROOT/f'outputs/pure_dl_perm_full_utility_v87_PERM_{mode}_fold5'
        (out/'PROJECTS.md').write_text(progress,encoding='utf-8');append(out/'README.md','## 最终结果','## 最终结果\n'+final)
    note=f"## v87 PERM完整网络效用完成\n当前TOTAL={scores['TOTAL']:.12f}，变化{result['delta_reference_TOTAL']:+.12f}；最终joint-likelihood PERM{contrast['channel_delta']:+.12f}，更新后差{updated_delta:+.12f}。配方results/deep_learning/perm_full_utility_v87_20260930/CURRENT_RECIPE.json；无其他折或旧队列。"
    for p in (ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):append(p,'## v87 PERM完整网络效用完成',note)
    for p in (ROOT/'AGENTS.md',ROOT.parent/'AGENTS.md'):append(p,'## v87当前研究配方',note.replace('## v87 PERM完整网络效用完成','## v87当前研究配方'))
    print(json.dumps(native(dict(scores=scores,delta=result['delta_reference_TOTAL'],matched_delta=contrast['channel_delta'],updated_delta=updated_delta,timing=timing)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
