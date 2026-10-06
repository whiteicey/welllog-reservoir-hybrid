"""Finalize six feature trials, replay positive selections and preserve old recipes."""
from pathlib import Path
from datetime import datetime
import gc,hashlib,json,time
import numpy as np
import torch
from crosscurve_v32 import ROOT,STUDY,PREVIOUS,TARGETS,MODES,Network,Evaluator,load_data,reference_prediction,factor_scope
from run_crosscurve_v32 import record
from launch_champion_posterior_v27 import jobs,append_once
from finish_all_heads_v26 import diagnostics,native
from train_dual8222_heads80 import atomic_json,atomic_array
import train_invariant as base

W={'POR':.3,'PERM':.35,'SW':.35}
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    progress=record()
    if not progress['all_complete']:raise RuntimeError('All six predefined candidates must finish')
    if jobs():raise RuntimeError('Active GPU jobs prevent final replay')
    previous=read(PREVIOUS/'CURRENT_RECIPE.json');reference=reference_prediction()
    with np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/oof.npz') as z:
        truth=z['truth'][z['fold']==4].copy();wells=z['well'][z['fold']==4].copy()
    assert truth.shape==reference.shape==(144665,3)
    refs=base.metric(truth,reference)
    assert all(abs(refs[k]-v)<1e-9 for k,v in previous['scores'].items())
    records=[];selections=[];comparisons=[];costs={};events=[];combined=reference.copy()
    for group in progress['rows']:
        t=group['target'];c=TARGETS.index(t);out=ROOT/'outputs'/group['id'];cfg=read(out/'config.json')
        for name,digest in cfg['source_hashes'].items():
            assert hashlib.sha256((ROOT/'experiments/pure_dl'/name).read_bytes()).hexdigest()==digest,name
        assert hashlib.sha256((STUDY/'feature_manifest.json').read_bytes()).hexdigest()==cfg['metadata']['feature_manifest_sha256']
        own=[];preds={};updates={}
        for m in MODES:
            arm=out/m;rep=read(arm/'fold_5_report.json')
            pred=np.load(arm/'channel_pred.npy');up=np.load(arm/'best_updated_pred.npy')
            d=diagnostics(truth,reference,pred,wells,t);u=diagnostics(truth,reference,up,wells,t)
            assert abs(d['scores'][t]-rep['target_score'])<1e-9
            assert abs(d['scores']['TOTAL']-rep['reference_replacement_total'])<1e-9
            assert abs(u['scores'][t]-rep['best_after_update']['target_score'])<1e-9
            if rep['best_epoch']==0:assert np.array_equal(pred,reference[:,c])
            row=dict(target=t,mode=m,report=rep,selected=d,post_update=u,
                initial=read(arm/'initial_verification.json'),checkpoint=str((arm/'model_best.pt').relative_to(ROOT)))
            own.append(row);records.append(row);preds[m]=pred;updates[m]=up
        ctrl=reference.copy();ctrl[:,c]=preds['marginal']
        updated_control=reference.copy();updated_control[:,c]=updates['marginal']
        comparisons.append(dict(target=t,crosscorr_minus_marginal=diagnostics(truth,ctrl,preds['crosscorr'],wells,t),
            post_update_contrast=diagnostics(truth,updated_control,updates['crosscorr'],wells,t),
            same_parameter_count=True,same_initial_tensors=True,scope='same head and protocol; changes 18 extra features and their train-only normalization'))
        best=max(own,key=lambda r:r['selected']['scores'][t]);adopt=best['selected']['scores'][t]>refs[t]+1e-9
        old=next(x for x in previous['choices'] if x['target']==t)
        selections.append(dict(target=t,adopted=adopt,mode=best['mode'] if adopt else None,target_score=best['selected']['scores'][t],
            best_epoch=best['report']['best_epoch'],checkpoint=best['checkpoint'] if adopt else old['checkpoint'],
            added_parameters=best['report']['trainable_parameters'] if adopt else 0))
        combined[:,c]=preds[best['mode']]
        launch=datetime.fromisoformat(read(out/'launch.json')['started_at']);end=datetime.fromisoformat(group['report']['finished_at'])
        events.extend([(launch,1),(end,-1)])
        costs[t]=dict(seconds=group['report']['seconds'],peak_allocated_MiB=group['report']['peak_allocated_MiB'],
            median_epoch_seconds=float(np.median([x['seconds'] for x in group['report']['epoch_timing']])),
            scope='one target process includes two candidates sharing frozen forward, not two independent run times')
    tick=time.perf_counter();replayed=[];choices=[dict(x) for x in previous['choices']]
    if any(s['adopted'] for s in selections):
        torch.set_num_threads(3);torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True,warn_only=True)
        data,master=load_data()
        assert np.array_equal(truth,data['truth'][data['validation']]) and np.array_equal(wells,data['wells'][data['validation']])
        for s in selections:
            if not s['adopted']:continue
            t=s['target'];m=s['mode'];c=TARGETS.index(t)
            cp=torch.load(ROOT/s['checkpoint'],map_location='cpu',weights_only=False)
            assert cp['target']==t and cp['mode']==m
            model=Network(t,master,modes=(m,)).cuda();model.heads[m].load_state_dict(cp['delta_state'],strict=True)
            pred,score,_=Evaluator(data,master,c,reference).many(model,(m,))[m]
            assert np.array_equal(pred,combined[:,c]) and abs(score-s['target_score'])<1e-9
            combined[:,c]=pred;s['all144665_fresh_disk_replay_exact']=True
            s['replayed_scope']=factor_scope(model,data,m);replayed.append(t)
            for choice in choices:
                if choice['target']==t:
                    choice.update(key='v32_'+t+'_'+m,mode=m,factor=model.factor,architecture='frozen_predictor_crosscurve_feature_correction',
                        checkpoint=s['checkpoint'],model_module='crosscurve_v32',target_score=score,best_epoch=s['best_epoch'],
                        parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),
                        builder='Network(target,master,modes=(mode,)); heads[mode].load_state_dict(checkpoint[delta_state]); aligned feature views via load_data()',
                        feature_manifest=str((STUDY/'feature_manifest.json').relative_to(ROOT)))
            model.close();del model,cp;gc.collect();torch.cuda.empty_cache()
        del data,master;gc.collect();torch.cuda.empty_cache()
    scores=base.metric(truth,combined)
    assert abs(scores['TOTAL']-sum(W[s['target']]*s['target_score'] for s in selections))<1e-9
    perwell=[]
    for well in np.unique(wells):
        mask=wells==well
        delta=base.metric(truth[mask],combined[mask])['TOTAL']-base.metric(truth[mask],reference[mask])['TOTAL']
        perwell.append(dict(well=int(well),n=int(mask.sum()),TOTAL_delta=delta))
    counts=np.array([x['n'] for x in perwell]);deltas=np.array([x['TOTAL_delta'] for x in perwell])
    ix=np.random.default_rng(20261010).integers(0,len(counts),(10000,len(counts)))
    bootstrap=(counts[ix]*deltas[ix]).sum(1)/counts[ix].sum(1)
    result=native(dict(reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],
        candidates=records,selections=selections,comparisons=comparisons,per_well=perwell,
        well_wins=sum(x['TOTAL_delta']>0 for x in perwell),well_losses=sum(x['TOTAL_delta']<0 for x in perwell),
        exploratory_selection_unadjusted_TOTAL_95=np.quantile(bootstrap,[.025,.975]),fold=5,rows=144665,
        all_six_complete=True,actual_combined_prediction_scored=True,new_target_full_disk_replays=replayed,
        strict805_qualified=bool(scores['TOTAL']>80.5),complete_five_folds=False,remaining_folds_started=False,
        finished_at=datetime.now().astimezone().isoformat()))
    recipe=dict(previous);recipe.update(scores=native(scores),choices=choices,reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
    atomic_array(STUDY/'assembled_pred.npy',combined);atomic_json(STUDY/'assembled_report.json',result);atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    concurrent=peak=0
    for _,change in sorted(events):concurrent+=change;peak=max(peak,concurrent)
    timing=dict(target_processes=costs,actual_peak_launch_interval_concurrency=peak,maximum_allowed_training_jobs=2,
        feature_preparation_seconds=read(STUDY/'feature_manifest.json')['seconds'],
        first_launch_to_last_completion_seconds=(max(t for t,_ in events)-min(t for t,_ in events)).total_seconds(),
        final_replay_seconds=time.perf_counter()-tick,scope='training launch interval includes explicit staging gaps; coding, feature preparation and final replay listed separately')
    atomic_json(STUDY/'timing.json',timing)
    lines=['# v32：跨曲线局部相关的同结构对照','',f"完成：{result['finished_at']}",'',
        f"原TOTAL={refs['TOTAL']:.12f}；实际组合TOTAL={scores['TOTAL']:.12f}；增量={result['delta_reference_TOTAL']:+.12f}。",'',
        '|目标|附加视图|参数|训练后最高通道分|最终通道分|最佳/停止轮次|','|---|---|---:|---:|---:|---|']
    for row in records:
        t=row['target'];rep=row['report']
        lines.append(f"|{t}|{row['mode']}|{rep['trainable_parameters']}|{row['post_update']['scores'][t]:.9f}|{rep['target_score']:.9f}|{rep['best_epoch']}/{rep['stopped_epoch']}|")
    lines+=['','第0轮回退不计为训练收益。六项独立优化/早停，只复用冻结前向。','', '## 实际选择与来源','']
    for s in selections:
        t=s['target'];lines.append(f"{t}：{s['mode'] if s['adopted'] else '保留v31'}；通道{s['target_score']:.12f}；TOTAL贡献{W[t]*(s['target_score']-refs[t]):+.12f}；新增参数{s['added_parameters']}。")
        if s['adopted']:
            row=next(r for r in records if r['target']==t and r['mode']==s['mode'])
            lines.append('子集分解：'+json.dumps(native(row['selected']['populations']),ensure_ascii=False))
    lines+=['','## 跨曲线信息相对冗余统计对照','']
    for comp in comparisons:
        d=comp['crosscorr_minus_marginal'];u=comp['post_update_contrast']
        lines.append(f"{comp['target']} crosscorr-marginal：选定通道差{d['channel_delta']:+.12f}，TOTAL差{d['TOTAL_delta']:+.12f}；排除第0轮的最高通道差{u['channel_delta']:+.12f}。")
    lines+=['',f"组合{result['well_wins']}井升、{result['well_losses']}井降；探索性未校正区间{result['exploratory_selection_unadjusted_TOTAL_95']}。",
        '六项与早停均依赖反复开发的第五折，区间未校正选择偏差；没有独立井或种子确认。','',
        '## 机制与验证范围','',
        '原269维路径保持，六对曲线三尺度生成18维局部相关；分母有训练IQR下限，全部缩放仅在64训练井拟合。marginal的18维由已有幅值统计派生，故没有新增观测信息。',
        '每目标两候选同337→32→输出的头及完整初始张量，POR改16位置、PERM改最终默认logit、SW改16权重。父模型/原路由/既有修正全部冻结。该头相对父模型增加了新参数，不能将所有相对父模型的增量归因于跨曲线特征。',
        '特征校验包含原行序及预定窗口中心化协方差，未读取目标标签。六项第0轮全144665点复现，实际梯度及冻结状态检查通过，选定参数训练内完整恢复。新采纳目标独立磁盘重建并全量预测：'+str(replayed)+'；最终实际三列合并评分。',
        '不变目标复用其先前验证，不声称新增重载。原权重、源码和交付保留；无其它折、冷启动或骨干扩容。显式低秩交互本轮未做。','',
        '## 停止边界与待做','',
        '不按本次分数追加曲线配对、窗口、头宽度或学习率网格。若无跨曲线信息收益，不据此扩展交互网络；结论仅限当前特征与接入方式，不否定所有跨曲线建模。',
        '完整OOF>82仍未达成，其他四折未授权启动。是否有值得保留的特征机制，以本报告绝对分、同结构差值及子集证据共同判断。','',
        '## 文件','',
        'METHOD.md、feature_manifest.json、progress.json、assembled_report.json、CURRENT_RECIPE.json、timing.json、trainer_provenance.json；源码prepare_crosscurve_v32.py/crosscurve_v32.py/train_crosscurve_v32.py。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v32跨曲线特征对照完成'
    note=marker+'\n'+f"TOTAL={scores['TOTAL']:.12f}，相对v31{result['delta_reference_TOTAL']:+.12f}；"+', '.join(s['target']+':'+str(s['mode']) for s in selections)+'。六项完成，无其他折。证据crosscurve_features_v32_20260929/FINAL.md。'
    current='# v32已完成\n\n目标：检验当前强模型的跨曲线信息增量。\n\n已完成：训练侧特征准备、六项训练/最佳恢复、冻结验证、正选模型磁盘重建和实际组合。\n\n'+note+'\n\n未完成：独立新井/种子确认及显式交互模块；不自动扫描本批负设置。\n\n达成后：仅依据已验证机制讨论下一轮；旧交付保留，其他四折仍暂停。\n'
    (STUDY/'PROJECTS.md').write_text(current,encoding='utf-8')
    for path in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md']:
        append_once(path,marker,note)
    for row in records:
        arm=(ROOT/row['checkpoint']).parent;t=row['target'];rep=row['report']
        detail=f"\n## 最终结果\n{t}/{row['mode']}更新后最高{row['post_update']['scores'][t]:.12f}，最终{rep['target_score']:.12f}；最佳{rep['best_epoch']}/停止{rep['stopped_epoch']}。见研究目录FINAL.md。\n"
        append_once(arm/'README.md','## 最终结果',detail)
        (arm/'PROJECTS.md').write_text(current+detail,encoding='utf-8');(arm.parent/'PROJECTS.md').write_text(current,encoding='utf-8')
    for path in [ROOT.parent/'AGENTS.md',ROOT/'AGENTS.md']:
        append_once(path,'## v32当前研究配方','## v32当前研究配方\n'+f"TOTAL={scores['TOTAL']:.12f}；配方results/deep_learning/crosscurve_features_v32_20260929/CURRENT_RECIPE.json。六项结束，无自动后续实验或补折，旧模型及交付保留。")
    compact=dict(scores=scores,delta=result['delta_reference_TOTAL'],selections=selections,
        comparisons=[dict(target=c['target'],selected_delta=c['crosscorr_minus_marginal']['channel_delta'],post_update_delta=c['post_update_contrast']['channel_delta']) for c in comparisons],
        well_wins=result['well_wins'],well_losses=result['well_losses'],interval=result['exploratory_selection_unadjusted_TOTAL_95'],timing=timing)
    print('FINAL',json.dumps(native(compact),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
