"""Finalize full matched private-density runs; preserve real fair gains."""
from datetime import datetime
import gc,hashlib,json,time
import numpy as np
import torch
from sw_factor_sam_v96 import ROOT,STUDY,PREVIOUS,MODES,GATE,DENSITY,Network,Evaluator,load_data,reference_prediction
from finish_all_heads_v26 import diagnostics,native
from train_dual8222_heads80 import atomic_json,atomic_array
from run_retrieval_smooth_v36 import jobs,append
import train_invariant as base

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    if jobs():raise RuntimeError('Training remains active')
    if (STUDY/'assembled_report.json').exists():raise RuntimeError('Completed study is immutable')
    tick=time.perf_counter();old=read(PREVIOUS/'CURRENT_RECIPE.json');reference=reference_prediction()
    with np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/oof.npz') as z:
        truth=z['truth'][z['fold']==4].copy();wells=z['well'][z['fold']==4].copy()
    refs=base.metric(truth,reference);assert all(abs(refs[k]-v)<1e-9 for k,v in old['scores'].items())
    fixed=torch.load(GATE,map_location='cpu',weights_only=False)['state_dict']
    expected_fixed=hashlib.sha256(b''.join(v.numpy().tobytes() for v in fixed.values())).hexdigest()
    rows=[];configs=[];preds={};postpreds={};histories={};times=[]
    for mode,spec in MODES.items():
        out=ROOT/f'outputs/pure_dl_sw_factor_sam_v96_SW_{mode}_fold5';assert read(out/'runtime_status.json')['status']=='complete'
        cfg=read(out/'config.json');rep=read(out/'fold_5_report.json');initial=read(out/'initial.json')
        for name,digest in cfg['source_hashes'].items():assert sha(ROOT/'experiments/pure_dl'/name)==digest,name
        assert cfg['initialization']['sha256']==sha(DENSITY) and cfg['initialization']['atom_sha256']==sha(GATE)
        assert cfg['target_normalization']['sha256']==sha(ROOT/cfg['target_normalization']['checkpoint'])
        assert cfg['scoring_kernel_sha256']==sha(ROOT/'package/src/utils.py')
        assert cfg['total_parameters']==1678962 and cfg['trainable_parameters']==839481 and cfg['utility_weight']==spec['utility_weight']
        assert cfg['initial_fixed_hash']==expected_fixed and rep['fixed_current_and_selected_bitwise_unchanged']
        assert cfg['epochs']==80 and cfg['patience']==18 and cfg['sam_rho']==.1 and cfg['lr']==1e-4
        assert cfg['training_rows']==578891 and cfg['validation_rows']==144665 and len(cfg['bank_manifest']['train_wells'])==64 and not cfg['bank_manifest']['validation_fit']
        assert initial['all144665_parent_exact'] and initial['initial_prediction_sha256']==hashlib.sha256(reference[:,2].tobytes()).hexdigest()
        assert rep['all144665_selected_predictions_restored_exactly'] and rep['gradient_scope']['frozen_atom_has_no_gradient'] and rep['gradient_scope']['second_pass_rng_consumption_exact']
        assert abs(rep['gradient_scope']['actual_perturbation_norm']-.1)<1e-6
        for f in ('gradient_norms','decision_gradient_norms'):assert all(np.isfinite(v) and v>0 for v in rep['gradient_scope'][f].values())
        pred=np.load(out/'channel_pred.npy');post=np.load(out/'best_updated_pred.npy');diag=diagnostics(truth,reference,pred,wells,'SW');postdiag=diagnostics(truth,reference,post,wells,'SW')
        assert abs(rep['target_score']-diag['scores']['SW'])<1e-9 and abs(rep['reference_replacement_scores']['TOTAL']-diag['scores']['TOTAL'])<1e-9
        assert abs(rep['best_after_update']['target_score']-postdiag['scores']['SW'])<1e-9
        if rep['best_epoch']==0:assert np.array_equal(pred,reference[:,2])
        else:assert np.array_equal(pred,post)
        rows.append(dict(mode=mode,report=rep,against_parent=diag,updated_against_parent=postdiag));configs.append(cfg);preds[mode]=pred;postpreds[mode]=post
        histories[mode]={r['epoch']:r for r in read(out/'validation_history.json')}
        times.append(dict(mode=mode,start=read(out/'launch.json')['started_at'],end=rep['finished_at'],seconds=rep['seconds'],peak_allocated_MiB=rep['peak_allocated_MiB'],peak_reserved_MiB=rep['peak_reserved_MiB']))
    for k in configs[0]:assert k in ('mode','utility_weight') or configs[0][k]==configs[1][k],k
    other=reference.copy();other[:,2]=preds['likelihood'];contrast=diagnostics(truth,other,preds['joint'],wells,'SW')
    postother=reference.copy();postother[:,2]=postpreds['likelihood'];postcontrast=diagnostics(truth,postother,postpreds['joint'],wells,'SW')
    common=sorted(set(histories['likelihood'])&set(histories['joint']));deltas=[histories['joint'][e]['target_score']-histories['likelihood'][e]['target_score'] for e in common]
    trajectory=dict(epochs=common,deltas=deltas,positive_epochs=sum(d>0 for d in deltas),negative_epochs=sum(d<0 for d in deltas),mean_delta=float(np.mean(deltas)),last_delta=float(deltas[-1]))
    best=max(rows,key=lambda r:r['report']['target_score']);adopt=bool(best['report']['target_score']>refs['SW']+1e-9)
    retained={r['mode'] for r in rows if r['against_parent']['channel_delta']>1e-9}
    if postcontrast['channel_delta']>1e-9:retained.update(MODES)
    keep_files={m:('model_best.pt' if next(r for r in rows if r['mode']==m)['report']['best_epoch']>0 else 'model_best_updated.pt') for m in retained}
    combined=reference.copy();choices=[dict(c) for c in old['choices']];proof=[]
    if retained:
        torch.set_num_threads(3);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
        data,meta=load_data();assert np.array_equal(truth,data['truth'][data['validation']])
        for mode in sorted(retained):
            out=ROOT/f'outputs/pure_dl_sw_factor_sam_v96_SW_{mode}_fold5';file=out/keep_files[mode];saved=torch.load(file,map_location='cpu',weights_only=False)
            model=Network(mode,meta).cuda();model.load_state_dict(saved['state_dict'],strict=True);assert model.fixed_hash()==expected_fixed
            fresh,score,_=Evaluator(data,meta,2,reference)(model);expected=preds[mode] if keep_files[mode]=='model_best.pt' else postpreds[mode]
            assert np.array_equal(fresh,expected);proof.append(dict(mode=mode,file=str(file.relative_to(ROOT)),rows=144665,fresh_disk_replay_exact=True,sha256=sha(file),score=score))
            if adopt and mode==best['mode']:
                combined[:,2]=fresh;choices[2].update(key='v96_SW_'+mode,mode=mode,model_module='sw_factor_sam_v96',checkpoint=str(file.relative_to(ROOT)),source_checkpoints=None,best_epoch=saved['best_epoch'],target_score=score,
                    parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),builder='load_data; Network(mode,metadata); load saved state_dict; frozen atom/private density original Evaluator')
            del model,saved;gc.collect();torch.cuda.empty_cache()
    scores=base.metric(truth,combined);assert np.array_equal(combined[:,:2],reference[:,:2])
    result=native(dict(reference_scores=refs,selected_scores=scores,candidates=rows,adopted=adopt,selected_mode=best['mode'] if adopt else None,retained=sorted(retained),verification=proof,
        joint_minus_likelihood=contrast,post_update_joint_minus_likelihood=postcontrast,trajectory=trajectory,
        screening_fold=5,rows=144665,strict805_passed=bool(scores['TOTAL']>80.5),complete_five_folds=False,remaining_folds_started=False,finished_at=datetime.now().astimezone().isoformat()))
    atomic_json(STUDY/'assembled_report.json',result);atomic_array(STUDY/'assembled_pred.npy',combined)
    recipe=dict(old);recipe.update(scores=native(scores),choices=choices,reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)));atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    starts=[datetime.fromisoformat(t['start']) for t in times];ends=[datetime.fromisoformat(t['end']) for t in times];assert max(starts)>=min(ends)
    cost=dict(candidates=times,first_launch_to_last_completion_seconds=(max(ends)-min(starts)).total_seconds(),sum_candidate_seconds=sum(t['seconds'] for t in times),first_launch_to_verified_seconds=(datetime.now().astimezone()-min(starts)).total_seconds(),actual_process_concurrency=1,overlap_seconds=0.,finalization_seconds=time.perf_counter()-tick,scope='loading/warmup/train/eval/save included; explicit serial launch gap in total wall; authoring/atlas separate')
    atomic_json(STUDY/'timing.json',native(cost));plan=[]
    for mode in MODES:
        out=(ROOT/f'outputs/pure_dl_sw_factor_sam_v96_SW_{mode}_fold5').resolve();assert out.parent==(ROOT/'outputs').resolve()
        for name in ('model_best.pt','model_best_updated.pt','last_state.pt'):
            if keep_files.get(mode)==name:continue
            p=(out/name).resolve();assert p.parent==out
            if p.exists():plan.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
        atomic_json(out/'candidate_status.json',dict(retained=mode in retained,retained_file=keep_files.get(mode),no_automatic_resume=True))
    atomic_json(STUDY/'deletion_plan.json',native(dict(planned=plan,status='planned_before_unlink',retained=keep_files)));deleted=[]
    for item in plan:
        p=(ROOT/item['path']).resolve();assert p.parent.parent==(ROOT/'outputs').resolve();p.unlink();deleted.append(item);atomic_json(STUDY/'deletion_manifest.json',native(dict(deleted=deleted,status='in_progress')))
    atomic_json(STUDY/'deletion_manifest.json',native(dict(deleted=deleted,bytes=sum(x['bytes'] for x in deleted),status='complete',retained=keep_files)))
    lines=['# v96 固定SAM默认atom/私有密度优化结果','',f"当前原第五TOTAL {refs['TOTAL']:.12f} → {scores['TOTAL']:.12f}；采用={result['selected_mode']}。",'', '|目标损失|SW|TOTAL|最佳/停止|更新后最佳SW|对父SW|','|---|---:|---:|---:|---:|---:|']
    for r in rows:
        p=r['report'];lines.append(f"|{r['mode']}|{p['target_score']:.12f}|{p['reference_replacement_scores']['TOTAL']:.12f}|{p['best_epoch']}/{p['stopped_epoch']}|{p['best_after_update']['target_score']:.12f}|{r['against_parent']['channel_delta']:+.12f}|")
    lines+=['','探索与验证：同v95完整函数、固定r100 atom/私有v82密度；1678962总/839481优化器参数，129私有gate输出不用；原数据/9曲线/Q50/评分/动作/80-18/1e-4/SAM.1/DropKey.4相同，仅效用0/1。实际完整初始函数、atom无梯度/当前及最佳hash、密度实际梯度/扰动RNG、源/预处理/原评分/144665最佳恢复及固定三列通过；无独立预检/审查/重复Q。',
        '收益：含起点joint-likelihood SW='+str(contrast['channel_delta'])+'，完整训练后最高差='+str(postcontrast['channel_delta'])+'；同epoch诊断='+json.dumps(native(trajectory),ensure_ascii=False)+'。epoch0不作新收益；保留='+str(keep_files)+'；原始输入必要磁盘重载='+str(proof)+'。',
        '默认/数值/井诊断：'+json.dumps(native({r['mode']:r['against_parent'] for r in rows}),ensure_ascii=False),
        '不继续：若两项都epoch0且更新后joint无正增益，关闭本父函数/私有密度/SAM.1/1e-4/损失配置，不扫LR/人口/来源；不泛化因子模型或SAM。旧v84不同gate/单pass/3e-5负范围保持。',
        '成本：'+json.dumps(native(cost),ensure_ascii=False),f"删除{len(deleted)}个负/重复checkpoint，共{sum(x['bytes'] for x in deleted)}字节，完整计划先落盘，正源/旧交付保留。",'已完成：两项实际训练/选模/组合/必要重载/清理/记录；未做：其他折/独立泛化，开发折非完整OOF。第五严格>80.5且有收益后才考虑补折，不自动启动；完整原OOF>82未完成，目标后独立确认/新目录交付。']
    text='\n'.join(lines)+'\n';(STUDY/'FINAL.md').write_text(text,encoding='utf-8');(STUDY/'PROJECTS.md').write_text(text,encoding='utf-8');append(STUDY/'README.md','## 最终结果','## 最终结果\n'+text)
    for mode in MODES:
        out=ROOT/f'outputs/pure_dl_sw_factor_sam_v96_SW_{mode}_fold5';(out/'PROJECTS.md').write_text(text,encoding='utf-8');append(out/'README.md','## 最终结果','## 最终结果\n'+text)
    note=f"## v96私有密度SAM完成\n当前原第五TOTAL{scores['TOTAL']:.12f}，采用={result['selected_mode']}；两项真实恢复/评分/清理完成，未补折/完整OOF。配方results/deep_learning/sw_factor_sam_v96_20260930/CURRENT_RECIPE.json。\n"
    for p in (ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):append(p,'## v96私有密度SAM完成',note)
    for p in (ROOT/'AGENTS.md',ROOT.parent/'AGENTS.md'):append(p,'## v96当前研究配方',note.replace('## v96私有密度SAM完成','## v96当前研究配方'))
    append(ROOT/'EXPLORATIONS.md','v96完成：','- v96完成：[结果](results/deep_learning/sw_factor_sam_v96_20260930/FINAL.md)；'+note)
    print(json.dumps(dict(scores=native(scores),adopted=adopt,selected_mode=result['selected_mode'],post_update_delta=postcontrast['channel_delta'],retained=keep_files,deleted_bytes=sum(x['bytes'] for x in deleted)),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
