"""Finalize four v31 trials with actual original-population scores and replay."""
from pathlib import Path
from datetime import datetime
import gc,hashlib,importlib,json,time
import numpy as np
import torch
from run_mechanism_v31 import ROOT,STUDY,MODES,record
from launch_champion_posterior_v27 import jobs,append_once
from finish_all_heads_v26 import diagnostics,native
from train_dual8222_heads80 import atomic_json,atomic_array
import train_invariant as base

PREVIOUS=ROOT/'results/deep_learning/champion_coadapt_v30_20260929'
TARGETS=('POR','PERM','SW');WEIGHTS={'POR':.3,'PERM':.35,'SW':.35}
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    p=record()
    if not p['all_complete']:raise RuntimeError('Both target studies must finish first')
    if jobs():raise RuntimeError('Training remains active')
    previous=read(PREVIOUS/'CURRENT_RECIPE.json');reference=np.load(PREVIOUS/'assembled_pred.npy')
    with np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/oof.npz') as z:
        truth=z['truth'][z['fold']==4].copy();wells=z['well'][z['fold']==4].copy()
    assert truth.shape==reference.shape==(144665,3)
    refs=base.metric(truth,reference)
    assert all(abs(refs[k]-v)<1e-9 for k,v in previous['scores'].items())
    records=[];selections=[];comparisons=[];costs={};events=[]
    combined=reference.copy();choices=[dict(c) for c in previous['choices']]
    for group in p['rows']:
        t=group['target'];c=TARGETS.index(t);out=ROOT/'outputs'/group['id'];cfg=read(out/'config.json')
        for name,digest in cfg['source_hashes'].items():
            assert hashlib.sha256((ROOT/'experiments/pure_dl'/name).read_bytes()).hexdigest()==digest,name
        own=[];preds={}
        for mode in MODES[t]:
            arm=out/mode;report=read(arm/'fold_5_report.json');pred=np.load(arm/'channel_pred.npy');up=np.load(arm/'best_updated_pred.npy')
            selected=diagnostics(truth,reference,pred,wells,t);updated=diagnostics(truth,reference,up,wells,t)
            assert abs(selected['scores'][t]-report['target_score'])<1e-9
            assert abs(updated['scores'][t]-report['best_after_update']['target_score'])<1e-9
            assert abs(selected['scores']['TOTAL']-report['reference_replacement_total'])<1e-9
            if report['best_epoch']==0:assert np.array_equal(pred,reference[:,c])
            row=dict(target=t,mode=mode,report=report,selected=selected,post_update=updated,
                initial=read(arm/'initial_verification.json'),checkpoint=str((arm/'model_best.pt').relative_to(ROOT)),
                prediction=str((arm/'channel_pred.npy').relative_to(ROOT)))
            records.append(row);own.append(row);preds[mode]=pred
        ctrl=reference.copy();ctrl[:,c]=preds[MODES[t][0]]
        comparisons.append(dict(target=t,treatment=MODES[t][1],control=MODES[t][0],
            selected_contrast=diagnostics(truth,ctrl,preds[MODES[t][1]],wells,t),
            best_post_update_channel_contrast=own[1]['post_update']['channel_delta']-own[0]['post_update']['channel_delta']))
        best=max(own,key=lambda r:r['selected']['scores'][t]);adopt=best['selected']['scores'][t]>refs[t]+1e-9
        old=next(c for c in previous['choices'] if c['target']==t)
        selections.append(dict(target=t,adopted=adopt,mode=best['mode'] if adopt else None,target_score=best['selected']['scores'][t],
            checkpoint=best['checkpoint'] if adopt else old['checkpoint'],best_epoch=best['report']['best_epoch'],
            trainable_parameters=best['report']['trainable_parameters'] if adopt else 0,module=cfg['module']))
        combined[:,c]=preds[best['mode']]
        launch=datetime.fromisoformat(read(out/'launch.json')['started_at']);end=datetime.fromisoformat(group['report']['finished_at'])
        events.extend([(launch,1),(end,-1)])
        costs[t]=dict(seconds=group['report']['seconds'],peak_allocated_MiB=group['report']['peak_allocated_MiB'],
            median_epoch_seconds=float(np.median([v['seconds'] for v in group['report']['epoch_timing']])),
            scope='one target process includes both candidates; fixed backbone computation shared, not independent per-arm wall time')
    start=time.perf_counter();replayed=[];torch.set_num_threads(3)
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
    for s in selections:
        if not s['adopted']:
            s['verification']='retained prior verified prediction; in-run candidate restoration only';continue
        t=s['target'];c=TARGETS.index(t);m=s['mode'];module=importlib.import_module(s['module'])
        data,master=module.load_data()
        assert np.array_equal(truth,data['truth'][data['validation']]) and np.array_equal(wells,data['wells'][data['validation']])
        cp=torch.load(ROOT/s['checkpoint'],map_location='cpu',weights_only=False)
        assert cp['target']==t and cp['mode']==m
        if t=='SW':
            # Deployment can read knots from saved metadata, with no training labels.
            master=dict(master);master['v31_knots']=cp['config']['metadata']['knots']
        model=module.Network(t,master,modes=(m,)).cuda();model.heads[m].load_state_dict(cp['delta_state'],strict=True)
        pred,score,_=module.Evaluator(data,master,c,reference).many(model,(m,))[m]
        assert np.array_equal(pred,combined[:,c]) and abs(score-s['target_score'])<1e-9
        s['all144665_fresh_disk_replay_exact']=True;s['replayed_scope']=module.factor_scope(model,data,m)
        replayed.append(t);combined[:,c]=pred
        for choice in choices:
            if choice['target']==t:
                choice.update(key='v31_'+t+'_'+m,mode=m,checkpoint=s['checkpoint'],model_module=s['module'],
                    target_score=score,best_epoch=s['best_epoch'],parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),
                    builder='Network(target,master,modes=(mode,)); heads[mode].load_state_dict(checkpoint[delta_state]); SW master[v31_knots] from checkpoint config metadata',
                    architecture='same_bound_reparameterized_routes' if t=='POR' else 'conditional_monotone_warp',
                    factor='route_parameterization' if t=='POR' else 'conditional_distribution')
        model.close();del model,cp,data,master;gc.collect();torch.cuda.empty_cache()
    scores=base.metric(truth,combined)
    assert abs(scores['TOTAL']-sum(WEIGHTS[t]*scores[t] for t in TARGETS))<1e-10
    assert np.array_equal(combined[:,1],reference[:,1])
    wells_out=[]
    for w in np.unique(wells):
        mask=wells==w;delta=base.metric(truth[mask],combined[mask])['TOTAL']-base.metric(truth[mask],reference[mask])['TOTAL']
        wells_out.append(dict(well=int(w),n=int(mask.sum()),TOTAL_delta=delta))
    counts=np.array([v['n'] for v in wells_out]);deltas=np.array([v['TOTAL_delta'] for v in wells_out])
    ix=np.random.default_rng(20261010).integers(0,len(counts),(10000,len(counts)))
    boots=(counts[ix]*deltas[ix]).sum(1)/counts[ix].sum(1)
    concurrency=0;maximum=0
    for _,v in sorted(events):concurrency+=v;maximum=max(maximum,concurrency)
    result=native(dict(reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],
        candidates=records,selections=selections,comparisons=comparisons,all_four_complete=True,fold=5,rows=144665,
        actual_combined_prediction_scored=True,new_target_full_disk_replays=replayed,PERM_unchanged=True,
        per_well=wells_out,well_wins=sum(v['TOTAL_delta']>0 for v in wells_out),well_losses=sum(v['TOTAL_delta']<0 for v in wells_out),
        exploratory_selection_unadjusted_TOTAL_95=np.quantile(boots,[.025,.975]),
        finished_at=datetime.now().astimezone().isoformat(),complete_five_folds=False,remaining_folds_started=False))
    recipe=dict(previous);recipe.update(scores=native(scores),choices=choices,reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
    atomic_array(STUDY/'assembled_pred.npy',combined);atomic_json(STUDY/'assembled_report.json',result);atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    timing=dict(targets=costs,maximum_allowed_training_jobs=2,peak_launch_interval_concurrency=maximum,
        launch_to_last_completion_seconds=(max(t for t,v in events)-min(t for t,v in events)).total_seconds(),
        final_replay_seconds=time.perf_counter()-start,scope='launch intervals include staged gaps; implementation and final replay separate')
    atomic_json(STUDY/'timing.json',timing)
    lines=['# v31：路由重参数化与连续分布单调修正','',f"完成：{result['finished_at']}",'',
        f"基线TOTAL={refs['TOTAL']:.12f}；最终实际TOTAL={scores['TOTAL']:.12f}；变化={result['delta_reference_TOTAL']:+.12f}。",'',
        '|目标|方案|可训练参数|更新后最高通道分|最终通道分|最佳/停止轮次|','|---|---|---:|---:|---:|---|']
    for r in records:
        t=r['target'];v=r['report'];lines.append(f"|{t}|{r['mode']}|{v['trainable_parameters']}|{r['post_update']['scores'][t]:.9f}|{v['target_score']:.9f}|{v['best_epoch']}/{v['stopped_epoch']}|")
    lines+=['','第0轮回退代表保留旧模型，不算训练收益。PERM始终保留当前预测。','', '## 最终选择与子集分解','']
    for s in selections:
        t=s['target'];lines.append(f"{t}：{s['mode'] if s['adopted'] else '保留父模型'}；通道{s['target_score']:.12f}。")
        if s['adopted']:
            r=next(r for r in records if r['target']==t and r['mode']==s['mode'])
            lines.append(json.dumps(native(r['selected']['populations']),ensure_ascii=False))
    lines+=['','## 匹配对照','']
    for comp in comparisons:
        d=comp['selected_contrast'];lines.append(f"{comp['target']} {comp['treatment']}-{comp['control']}：最终通道{d['channel_delta']:+.12f}；排除epoch0的最高通道差{comp['best_post_update_channel_contrast']:+.12f}。")
    lines+=['',f"组合{result['well_wins']}井升、{result['well_losses']}井降，探索性未校正区间{result['exploratory_selection_unadjusted_TOTAL_95']}。",
        '一个反复用于开发的第五折与本次种子；同折区间未消除候选选择偏差，不作为独立泛化证据。','',
        '## 验证与限制','',
        'POR两组同30819学习参数、同原每级边界；anchored保存额外冻结锚点，训练/推理开销增加。初始函数全量一致，导数探针见initial_verification.json。clip可能在越界后梯度为零，未扩大范围。',
        'SW两组共享初始化输入投影，输出2/3参数，容量略异；PWL为四段分段线性单调变换，三内部位移、训练井分位点、固定边界及恒等尾部。affine全域仿射，尾部约束也不同，非纯容量一致的因果比较。',
        'F_new(z)=F0(T(z))，f_new=f0(T(z))*Tprime；完整Jacobian已进入训练，默认概率固定。所有支持格/动作/评分保持，变换的是模型条件分布。节点部署可从检查点读取，无需推理标签。',
        '全部候选训练内最佳恢复逐元素一致；新采用目标独立磁盘重建：'+str(replayed)+'；原始行序核对并实际三列评分。旧状态保持不变。',
        '不启动其他折、冷启动或扩容；原权重和交付不覆盖。负结果停止本设置延伸，不将任何增益宣称为完整OOF或达成82。跨曲线特征仍未执行。','',
        '## 文件','', 'AGENTS.md/METHOD.md/PROJECTS.md、progress.json、assembled_report.json、CURRENT_RECIPE.json、timing.json；各目标子目录保留初始探针、更新后最佳、最终评分、梯度及学习曲线。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v31有限机制验证完成'
    note=marker+'\n'+f"TOTAL={scores['TOTAL']:.12f}，相对v30{result['delta_reference_TOTAL']:+.12f}；"+', '.join(s['target']+':'+str(s['mode']) for s in selections)+'。四项结束，无其他折；证据mechanism_checks_v31_20260929/FINAL.md。'
    current='# v31已完成\n\n目标：验证两项有诊断支持的机制。\n\n已完成：四项训练、初始函数/冻结/最佳恢复、实际组合与采用目标磁盘重载。\n\n'+note+'\n\n计划/未做：跨曲线特征及独立重复验证尚未执行，不自动延伸本批负项。\n\n达成后：固定有收益配方；由用户授权决定完整OOF。原交付保持。\n'
    (STUDY/'PROJECTS.md').write_text(current,encoding='utf-8')
    for pth in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md']:
        append_once(pth,marker,note)
    for r in records:
        arm=(ROOT/r['checkpoint']).parent;part=f"\n## 最终结果\n{r['target']}/{r['mode']}更新后{r['post_update']['scores'][r['target']]:.12f}，最终{r['report']['target_score']:.12f}。详见研究目录FINAL.md。\n"
        append_once(arm/'README.md','## 最终结果',part);(arm/'PROJECTS.md').write_text(current+part,encoding='utf-8');(arm.parent/'PROJECTS.md').write_text(current,encoding='utf-8')
    for pth in [ROOT.parent/'AGENTS.md',ROOT/'AGENTS.md']:
        append_once(pth,'## v31当前研究配方','## v31当前研究配方\n'+f"实际TOTAL={scores['TOTAL']:.12f}；当前配方results/deep_learning/mechanism_checks_v31_20260929/CURRENT_RECIPE.json。四项已结束，无其他折或队列，旧交付保留。")
    print('FINAL',json.dumps(native(dict(scores=scores,delta=result['delta_reference_TOTAL'],selections=selections,well_wins=result['well_wins'],well_losses=result['well_losses'],interval=result['exploratory_selection_unadjusted_TOTAL_95'],timing=timing)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
