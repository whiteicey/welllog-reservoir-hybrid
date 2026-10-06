"""Finalize all 11 heads, choose at most one per target, replay newly adopted models."""
from pathlib import Path
from datetime import datetime
import gc,hashlib,json,subprocess,sys,time
import numpy as np
import torch
from champion_all_heads_v26 import ROOT,STUDY,PREVIOUS,FACTORS,TARGETS,Network,Evaluator,load_data,reference_prediction
from train_all_heads_v26 import factor_scope
from train_dual8222_heads80 import atomic_json,atomic_array
from launch_latent_gate_v17 import active_jobs,append_once
from gpu_job_guard_v2 import check_slot
import train_invariant as base

WEIGHTS={'POR':.30,'PERM':.35,'SW':.35}
LABELS={'gate':'默认状态门控','weight':'分量混合权重','location':'分量位置','scale':'分量尺度'}


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def native(value):
    """Convert NumPy diagnostics to strict, portable JSON primitive types."""
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return native(value.tolist())
    if isinstance(value,dict):return {k:native(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [native(v) for v in value]
    return value


def diagnostics(truth,reference,prediction,wells,target):
    c=TARGETS.index(target);weight=WEIGHTS[target]
    changed=reference.copy();changed[:,c]=prediction
    old=base.metric(truth,reference);new=base.metric(truth,changed)
    defaults=base.dmask(truth)[:,c];pop={}
    for name,mask in (('default',defaults),('numeric',~defaults)):
        a=base.metric(truth[mask],reference[mask])[target];b=base.metric(truth[mask],changed[mask])[target]
        pop[name]=dict(n=int(mask.sum()),reference=a,candidate=b,channel_delta=b-a,TOTAL_contribution=weight*(b-a)*mask.mean())
    perwell=[]
    for well in np.unique(wells):
        mask=wells==well
        a=base.metric(truth[mask],reference[mask])[target];b=base.metric(truth[mask],changed[mask])[target]
        perwell.append(dict(well=int(well),n=int(mask.sum()),channel_delta=b-a))
    counts=np.array([r['n'] for r in perwell]);delta=np.array([r['channel_delta'] for r in perwell])
    indexes=np.random.default_rng(20261010).integers(0,len(perwell),(10000,len(perwell)))
    bootstrap=weight*(counts[indexes]*delta[indexes]).sum(1)/counts[indexes].sum(1)
    return dict(scores=new,channel_delta=new[target]-old[target],TOTAL_delta=new['TOTAL']-old['TOTAL'],populations=pop,
        well_wins=sum(r['channel_delta']>0 for r in perwell),well_losses=sum(r['channel_delta']<0 for r in perwell),
        per_well=perwell,exploratory_selection_unadjusted_TOTAL_95=np.quantile(bootstrap,[.025,.975]).tolist())


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if active_jobs():raise RuntimeError('Final replay requires no active trainers')
    for target in FACTORS:
        if not (ROOT/'outputs'/f'pure_dl_all_heads_v26_{target}_fold5/fold_5_report.json').exists():raise RuntimeError('Incomplete '+target)
    subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).with_name('run_all_heads_v26.py')),'status','--record'],check=True,stdout=subprocess.DEVNULL)
    progress=read(STUDY/'progress.json');assert progress['all_complete'] and progress['completed_arms']==11
    previous=read(PREVIOUS/'CURRENT_RECIPE.json');reference=reference_prediction()
    with np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/oof.npz') as z:
        truth=z['truth'][z['fold']==4].copy();wells=z['well'][z['fold']==4].copy()
    assert truth.shape==reference.shape==(144665,3)
    reference_scores=base.metric(truth,reference)
    assert all(abs(reference_scores[k]-v)<1e-9 for k,v in previous['scores'].items())
    records=[];selections=[];starts=[];ends=[];costs={}
    for target,factors in FACTORS.items():
        out=ROOT/'outputs'/f'pure_dl_all_heads_v26_{target}_fold5';cfg=read(out/'config.json');group=read(out/'fold_5_report.json')
        for name,digest in cfg['source_hashes'].items():
            assert hashlib.sha256((ROOT/'experiments/pure_dl'/name).read_bytes()).hexdigest()==digest,name
        starts.append(datetime.fromisoformat(read(out/'launch.json')['started_at']));ends.append(datetime.fromisoformat(group['finished_at']))
        costs[target]=dict(group_seconds=group['seconds'],peak_allocated_MiB=group['peak_allocated_MiB'],
            group_epoch_median_seconds=float(np.median([r['seconds'] for r in group['epoch_timing']])),
            per_arm_epochs={r['factor']:r['stopped_epoch'] for r in group['arms']},
            scope='target process including shared frozen inference; not separate per-arm wall time')
        own=[]
        for factor in factors:
            arm=out/factor;report=read(arm/'fold_5_report.json')
            selected=np.load(arm/'channel_pred.npy');updated=np.load(arm/'best_updated_pred.npy')
            d=diagnostics(truth,reference,selected,wells,target)
            u=diagnostics(truth,reference,updated,wells,target)
            assert abs(d['scores'][target]-report['target_score'])<1e-9
            assert abs(d['scores']['TOTAL']-report['reference_replacement_total'])<1e-9
            assert abs(u['scores'][target]-report['best_after_update']['target_score'])<1e-9
            cp=torch.load(arm/'model_best.pt',map_location='cpu',weights_only=False)
            assert cp['target']==target and cp['factor']==factor and cp['best_epoch']==report['best_epoch']
            if report['best_epoch']==0:
                assert np.array_equal(selected,reference[:,TARGETS.index(target)])
                assert all(bool((v==0).all()) for n,v in cp['delta_state'].items() if '.output.' in n)
            row=dict(target=target,factor=factor,trainable_parameters=report['trainable_parameters'],
                best_epoch=report['best_epoch'],stopped_epoch=report['stopped_epoch'],stop_reason=report['stop_reason'],
                best_updated_epoch=report['best_after_update']['epoch'],selected=d,post_update=u,
                checkpoint=str((arm/'model_best.pt').relative_to(ROOT)),source_checkpoint=cfg['source_checkpoint'],
                all_original_state_unchanged=report['all_original_state_unchanged'],
                in_run_all144665_restored=report['all144665_selected_predictions_restored_exactly'],
                gradient_scope=report['gradient_scope'],factor_scope=report['factor_scope'])
            records.append(row);own.append(row)
        winner=max(own,key=lambda r:r['selected']['scores'][target])
        adopt=winner['selected']['scores'][target]>reference_scores[target]+1e-9
        selections.append(dict(target=target,factor=winner['factor'] if adopt else None,adopted=adopt,
            best_new_factor=winner['factor'],target_score=winner['selected']['scores'][target] if adopt else reference_scores[target],
            checkpoint=winner['checkpoint'] if adopt else next(c['checkpoint'] for c in previous['choices'] if c['target']==target),
            source_checkpoint=winner['source_checkpoint'],best_epoch=winner['best_epoch'] if adopt else None,
            additional_parameters=winner['trainable_parameters'] if adopt else 0))
    # Assemble the actual selected columns, then independently rebuild each newly adopted target.
    started=time.perf_counter();torch.set_num_threads(3);check_slot(min_free_gib=3.)
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
    data,master=load_data()
    assert np.array_equal(truth,data['truth'][data['validation']]) and np.array_equal(wells,data['wells'][data['validation']])
    combined=reference.copy();choices=[dict(c) for c in previous['choices']]
    for s in selections:
        if not s['adopted']:
            s['verification']='unchanged target reuses prior verified predictions';continue
        target=s['target'];f=s['factor'];c=TARGETS.index(target)
        cp=torch.load(ROOT/s['checkpoint'],map_location='cpu',weights_only=False)
        model=Network(target,master,factors=(f,)).cuda()
        model.heads[f].load_state_dict(cp['delta_state'],strict=True)
        pred,score,total=Evaluator(data,master,c,reference).many(model,(f,))[f]
        saved=np.load((ROOT/s['checkpoint']).parent/'channel_pred.npy')
        assert np.array_equal(pred,saved) and abs(score-s['target_score'])<1e-9
        combined[:,c]=pred;s['all144665_fresh_disk_replay_exact']=True;s['factor_scope_replayed']=factor_scope(model,data,f)
        s['verification']='fresh independent frozen source + selected delta rebuild; all144665 exact'
        for choice in choices:
            if choice['target']==target:
                choice.update(key='v26_'+target+'_'+f,architecture='private_frozen_factor_head',factor=f,
                    checkpoint=s['checkpoint'],source_checkpoint=s['source_checkpoint'],model_module='champion_all_heads_v26',
                    builder='Network(target,master,factors=(factor,)); heads[factor].load_state_dict(checkpoint[delta_state])',
                    target_score=score,best_epoch=s['best_epoch'])
                choice.pop('mode',None)
        model.close();del model,cp;gc.collect();torch.cuda.empty_cache()
    scores=base.metric(truth,combined)
    expected=sum(WEIGHTS[s['target']]*s['target_score'] for s in selections)
    assert abs(scores['TOTAL']-expected)<1e-9
    total_delta=scores['TOTAL']-reference_scores['TOTAL']
    perwell=[]
    for w in np.unique(wells):
        mask=wells==w
        delta=base.metric(truth[mask],combined[mask])['TOTAL']-base.metric(truth[mask],reference[mask])['TOTAL']
        perwell.append(dict(well=int(w),n=int(mask.sum()),TOTAL_delta=delta))
    counts=np.array([r['n'] for r in perwell]);deltas=np.array([r['TOTAL_delta'] for r in perwell])
    indexes=np.random.default_rng(20261010).integers(0,16,(10000,16))
    boots=(counts[indexes]*deltas[indexes]).sum(1)/counts[indexes].sum(1)
    elapsed=time.perf_counter()-started
    assembled=dict(fold=5,rows=144665,reference_scores=reference_scores,selected_scores=scores,delta_reference_TOTAL=total_delta,
        candidates=records,selections=selections,all11_candidates_complete=True,actual_combined_prediction_scored=True,
        every_adopted_target_independently_disk_replayed=True,unchanged_targets_not_fresh_replayed=True,
        original_population_and_order_verified=True,per_well=perwell,
        combined_well_wins=sum(r['TOTAL_delta']>0 for r in perwell),combined_well_losses=sum(r['TOTAL_delta']<0 for r in perwell),
        combined_exploratory_selection_unadjusted_TOTAL_95=np.quantile(boots,[.025,.975]).tolist(),
        complete_five_folds=False,remaining_folds_started=False,finished_at=datetime.now().astimezone().isoformat())
    recipe=dict(scores=scores,choices=choices,source_preprocessing='results/deep_learning/champion_scaling_v21_20260928/data_manifest.json',
        amplitude_metadata='results/deep_learning/feature_structure_v20_20260928/feature_manifest.json',
        reference_recipe='results/deep_learning/champion_shape_v25_20260928/CURRENT_RECIPE.json',
        original_fifth_development_selection=True,complete_five_folds=False,remaining_folds_started=False,old_delivery_unchanged=True)
    assembled=native(assembled);recipe=native(recipe)
    records=assembled['candidates'];selections=assembled['selections'];scores=assembled['selected_scores']
    atomic_array(STUDY/'assembled_pred.npy',combined)
    atomic_json(STUDY/'assembled_report.json',assembled);atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    timing=dict(maximum_parallel_target_jobs=2,first_launch_to_last_finish_seconds=(max(ends)-min(starts)).total_seconds(),
        targets=costs,final_diagnostic_and_replay_seconds=elapsed,
        scope='target processes share frozen inference between alternatives; wall includes staged-launch gaps, imports, training and restoration, excludes source implementation and final replay')
    atomic_json(STUDY/'timing.json',timing)
    lines=['# v26：三个目标的多输出头优化','',f"完成：{assembled['finished_at']}",'',
        f"原配方TOTAL={reference_scores['TOTAL']:.12f}；实际组合TOTAL={scores['TOTAL']:.12f}；增量={total_delta:+.12f}。",'',
        '## 覆盖范围','',
        '已完成11项：POR门控/混合权重/尺度，PERM和SW各门控/混合权重/位置/尺度。POR已有效的v25位置修正始终保留。没有从头训练或扩容；原网络和已有输出参数全部冻结。每项独立小头、优化器、调度和早停。',
        '本批新候选固定一个幅值条件MLP设计。正负结果不代表穷尽某类输出头的所有可能结构。','',
        '## 全部结果（更新后最佳与第0轮回退分开）','',
        '|目标|头|新增参数|训练后最好通道分|选定通道分|选定对TOTAL贡献|最佳/停止轮次|',
        '|---|---|---:|---:|---:|---:|---|']
    for r in records:
        t=r['target'];p=r['post_update'];d=r['selected']
        lines.append(f"|{t}|{LABELS[r['factor']]}|{r['trainable_parameters']}|{p['scores'][t]:.9f}|{d['scores'][t]:.9f}|{d['TOTAL_delta']:+.9f}|{r['best_epoch']}/{r['stopped_epoch']}|")
    lines+=['','选定第0轮代表保留原预测，不计新增训练收益。候选之间不共享可训练参数；组内冻结推理复用仅为计算优化。不同头的输出维度不同，gate12354参数，其余12816，未声称严格同容量因果比较。','',
        '## 实际采用与收益来源','']
    for s in selections:
        t=s['target'];delta=s['target_score']-reference_scores[t]
        lines.append(f"{t}：采用{s['factor'] if s['adopted'] else '原模型'}，通道{reference_scores[t]:.9f}→{s['target_score']:.9f}；对TOTAL贡献{WEIGHTS[t]*delta:+.9f}。新增推理参数{s['additional_parameters']}。")
        if s['adopted']:
            r=next(r for r in records if r['target']==t and r['factor']==s['factor']);pop=r['selected']['populations']
            lines.append(f"默认段通道变化{pop['default']['channel_delta']:+.9f}，TOTAL贡献{pop['default']['TOTAL_contribution']:+.9f}；非默认段通道变化{pop['numeric']['channel_delta']:+.9f}，TOTAL贡献{pop['numeric']['TOTAL_contribution']:+.9f}。{r['selected']['well_wins']}/16井改善。")
    lines+=['',f"最终组合{assembled['combined_well_wins']}井提高、{assembled['combined_well_losses']}井下降；选择未校正整井重采样区间{assembled['combined_exploratory_selection_unadjusted_TOTAL_95']}。",
        '同一目标只选一个完整新候选，未将不同因子增量直接叠加；三个目标独立组合后实际调用原metric确认总分。原第五折已经反复选模，本批另比较11项，分差和区间均不是独立泛化确认。','',
        '## 结构与协议','',
        '每位专家原隐藏h∈R^128与幅值a∈R^63拼接，MLP191→32→1/8，输出层零初始化。非位置偏移0.25*tanh(u/0.25)，位置偏移0.25*sigma*tanh(u/(0.25*sigma))，sigma为固定原分量标准差。单个候选只改一个专家输出因子；gate改变可以通过生存质量重新分配最终混合权重。',
        '原64训练井/16验证井、九曲线206维主输入、Q50及训练井统计、256窗口/中央128监督、原Student-t或Gaussian似然/3和256格/800动作单覆盖解码不变。新增头lr3e-4、80/pat18/min10、bs64×64步、AdamW wd.01、clip2、cosine到1e-6。',
        '每项初始144665点预测完全匹配当前源模型；全部源参数和缓冲区冻结，实际反向有新梯度。除目标因子外的专家输出逐元素检查通过，原源码指纹保持。源模型始终eval。','',
        '## 验证与交付边界','',
        '11项选定输出在训练进程内均全量恢复验证。最终采用的新目标另从磁盘独立构建源模型和小头并推理，全部144665点精确复现；未变目标沿用此前验证预测，未重复声称新GPU重载。',
        '小头检查点只保存delta_state；源检查点、原初始化依赖和幅值侧输入仍是完整运行依赖，详见CURRENT_RECIPE.json及各config。现有源权重和交付不覆盖。',
        '本批全部结束，无新队列/其他折或冷启动扩容。完整原五折OOF>82及第五折严格>80.5仍需按实际新分数判定，未自动启动补折。','',
        '## 尚未执行','',
        '同目标多个正因子联合训练及其相互影响、跨种子/独立划分确认和其他四折均未执行。本批负因子仅在当前设计下停止，不追加边界系数或宽度扫描。','',
        '## 记录','',
        '本目录progress.json/METHOD.md/assembled_report.json/CURRENT_RECIPE.json；各outputs/pure_dl_all_heads_v26_<目标>_fold5下的因子子目录保存独立结果和小头，父目录保存可恢复整组训练状态。',
        f"三目标显式分批，首次启动到最后完成{timing['first_launch_to_last_finish_seconds']:.3f}秒；组内共享推理，不将组总耗时冒充每项独立耗时。最终诊断和重载{elapsed:.3f}秒另计。"]
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v26 三目标输出头优化完成'
    note=marker+'\n'+datetime.now().astimezone().isoformat()+f"。11项完成，实际组合TOTAL={scores['TOTAL']:.12f}，相对v25{total_delta:+.12f}；采用"+','.join(s['target']+':'+str(s['factor']) for s in selections)+'。新采纳目标全144665点独立磁盘重载，旧源权重与交付不改，无其他折。证据all_heads_v26_20260928/FINAL.md。'
    for p in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):append_once(p,marker,note)
    current='# v26状态：已完成\n\n目标：三目标输出头独立改进及真实组合。\n\n已完成：11项训练、各自早停、原权重冻结、初始函数检查、最终组合和新目标重载。\n\n'+note+'\n\n计划／未做：同目标多因子联合、独立重复验证和其他四折。本批不自动延伸。\n\n完成后：保留新研究配方及原对照，已交付包不覆盖。\n'
    (STUDY/'PROJECTS.md').write_text(current,encoding='utf-8')
    for t,fs in FACTORS.items():
        out=ROOT/'outputs'/f'pure_dl_all_heads_v26_{t}_fold5'
        append_once(out/'README.md',marker,note);(out/'PROJECTS.md').write_text(current,encoding='utf-8')
        for f in fs:
            r=next(r for r in records if r['target']==t and r['factor']==f)
            arm_note=note+f"\n本项{t}/{f}选定通道分{r['selected']['scores'][t]:.9f}，最佳第{r['best_epoch']}轮，停止第{r['stopped_epoch']}轮。"
            append_once(out/f/'README.md',marker,arm_note);(out/f/'PROJECTS.md').write_text(current+'\n'+arm_note+'\n',encoding='utf-8')
    policy='## v26最新完成配方\n'+f"原第五折TOTAL={scores['TOTAL']:.12f}，相对v25{total_delta:+.12f}。当前每目标选择见results/deep_learning/all_heads_v26_20260928/CURRENT_RECIPE.json。11项有限小头已完成，旧来源和交付保留；不自动叠加同目标因子、启动其他折或恢复冷启动/扩容。维护PROJECTS和图谱。"
    for p in (ROOT.parent/'AGENTS.md',ROOT/'AGENTS.md'):append_once(p,'## v26最新完成配方',policy)
    print('FINAL_SCORES',json.dumps(scores));print('DELTA',total_delta)
    print('SELECTIONS',json.dumps(selections,ensure_ascii=False))
    print('ALL_ARMS',json.dumps([dict(target=r['target'],factor=r['factor'],selected=r['selected']['scores'][r['target']],post_update=r['post_update']['scores'][r['target']],best_epoch=r['best_epoch'],stopped_epoch=r['stopped_epoch']) for r in records],ensure_ascii=False))
    print('ACTIVE',active_jobs());print('REPORT',STUDY/'FINAL.md')


if __name__=='__main__':main()
