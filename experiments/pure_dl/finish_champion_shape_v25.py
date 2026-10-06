"""Finish v25 from actual predictions; no extra training or parameter search."""
from datetime import datetime
from pathlib import Path
import gc, hashlib, json, subprocess, sys, time
import numpy as np
import torch
from champion_shape_v25 import ROOT, STUDY, FEATURES, MODES, REFERENCE, load_data, make_model
from train_scaling_v21 import Evaluator
from train_champion_shape_v25 import factor_check
from train_dual8222_heads80 import atomic_json, atomic_array
from evaluate_scratch_scaling_v23 import paired
from launch_latent_gate_v17 import active_jobs, append_once
from gpu_job_guard_v2 import check_slot
import train_invariant as base


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if active_jobs(): raise RuntimeError('Wait for the two explicitly started jobs to finish')
    for mode in MODES:
        if not (ROOT/'outputs'/f'pure_dl_champion_shape_v25_{mode}_fold5/fold_5_report.json').exists():
            raise RuntimeError('Unfinished candidate '+mode)
    subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).with_name('run_champion_shape_v25.py')),'status','--record'],check=True,stdout=subprocess.DEVNULL)
    progress=read(STUDY/'progress.json')
    assert progress['same_initial_state_and_trainable_parameters']
    torch.set_num_threads(3); check_slot(min_free_gib=3.)
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
    started=time.perf_counter()
    reference=np.load(REFERENCE/'fold_5_pred.npy')
    with np.load(REFERENCE/'oof.npz') as archive:
        truth=archive['truth'][archive['fold']==4].copy();wells=archive['well'][archive['fold']==4].copy()
    assert truth.shape==reference.shape==(144665,3)
    defaults=base.dmask(truth)[:,0];refscore=base.metric(truth,reference)
    candidates=[];predictions={};updated_predictions={};starts=[];ends=[]
    for mode in MODES:
        out=ROOT/'outputs'/f'pure_dl_champion_shape_v25_{mode}_fold5'
        report=read(out/'fold_5_report.json');cfg=read(out/'config.json')
        starts.append(datetime.fromisoformat(read(out/'launch.json')['started_at']))
        ends.append(datetime.fromisoformat(report['finished_at']))
        for name,digest in cfg['source_hashes'].items():
            assert hashlib.sha256((ROOT/'experiments/pure_dl'/name).read_bytes()).hexdigest()==digest,name
        data,master=load_data(mode)
        assert np.array_equal(truth,data['truth'][data['validation']])
        assert np.array_equal(wells,data['wells'][data['validation']])
        model=make_model(mode,master).cuda()
        fixed={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if '.delta_' not in n}
        # Replay the best checkpoint after an actual update even when selected=epoch0.
        cp=torch.load(out/'model_best_updated.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(cp['state_dict'],strict=True)
        assert all(torch.equal(model.state_dict()[n].detach().cpu(),v) for n,v in fixed.items())
        evaluator=Evaluator(data,master,0,reference)
        updated,score,total=evaluator(model)
        assert np.array_equal(updated,np.load(out/'best_updated_pred.npy'))
        assert abs(score-report['best_after_update']['target_score'])<1e-9
        factors=factor_check(model,data)
        selected=np.load(out/'channel_pred.npy')
        selected_cp=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False)
        assert all(torch.equal(selected_cp['state_dict'][n].cpu(),v) for n,v in fixed.items())
        if report['best_epoch']==0:
            assert np.array_equal(selected,reference[:,0])
            assert all(bool((v==0).all()) for n,v in selected_cp['state_dict'].items() if '.delta_out.' in n)
        else:
            assert np.array_equal(selected,updated)
            assert all(torch.equal(v,cp['state_dict'][n]) for n,v in selected_cp['state_dict'].items())
        combined=reference.copy();combined[:,0]=selected
        after_update=reference.copy();after_update[:,0]=updated
        selected_scores=base.metric(truth,combined);updated_scores=base.metric(truth,after_update)
        assert np.array_equal(combined[:,1:],reference[:,1:])
        assert abs(selected_scores['TOTAL']-report['reference_replacement_total'])<1e-9
        assert abs(updated_scores['TOTAL']-total)<1e-9
        atomic_array(STUDY/f'assembled_{mode}_pred.npy',combined)
        atomic_array(STUDY/f'assembled_{mode}_updated_pred.npy',after_update)
        row=dict(mode=mode,scores=selected_scores,post_update_scores=updated_scores,
            delta_reference_TOTAL=selected_scores['TOTAL']-refscore['TOTAL'],
            post_update_delta_reference_TOTAL=updated_scores['TOTAL']-refscore['TOTAL'],
            best_epoch=report['best_epoch'],best_updated_epoch=report['best_after_update']['epoch'],stopped_epoch=report['stopped_epoch'],
            trainable_parameters=12816,total_parameters=1303306,
            post_update_all144665_predictions_disk_replayed_exactly=True,
            selected_verification='same updated checkpoint' if report['best_epoch']>0 else 'zero-output state and exact baseline array',
            all_old_states_unchanged=True,factor_identity=factors,
            fixed_PERM_SW_arrays_reused=True,fixed_PERM_SW_fresh_GPU_replay=False,
            population=paired(truth[:,0],reference[:,0],selected,defaults,wells),
            updated_population=paired(truth[:,0],reference[:,0],updated,defaults,wells),
            checkpoint=str((out/'model_best.pt').relative_to(ROOT)))
        candidates.append(row);predictions[mode]=selected;updated_predictions[mode]=updated
        print('VERIFIED',json.dumps({k:v for k,v in row.items() if k not in ('population','updated_population')},ensure_ascii=False),flush=True)
        del data,master,model,evaluator,cp,selected_cp,fixed;gc.collect();torch.cuda.empty_cache()
    comparison=paired(truth[:,0],predictions['amplitude'],predictions['shape'],defaults,wells)
    updated_comparison=paired(truth[:,0],updated_predictions['amplitude'],updated_predictions['shape'],defaults,wells)
    best=max(candidates,key=lambda c:c['scores']['TOTAL'])
    positive=best['scores']['TOTAL']>refscore['TOTAL']+1e-9
    choices=[dict(c) for c in read(REFERENCE/'fold_5_report.json')['choices']]
    if positive:
        for choice in choices:
            if choice['target']=='POR':choice.update(key='POR_v25_'+best['mode'],architecture='champion_location_side',
                checkpoint=best['checkpoint'],model_module='champion_shape_v25',builder='make_model(mode,master)',mode=best['mode'],
                target_score=best['scores']['POR'],best_epoch=best['best_epoch'])
    selected_scores=best['scores'] if positive else refscore
    assembled=dict(fold=5,rows=144665,reference_scores=refscore,candidates=candidates,selected_scores=selected_scores,
        selected_new_POR=positive,shape_minus_amplitude=comparison,post_update_shape_minus_amplitude=updated_comparison,
        all_updated_checkpoints_disk_replayed=True,original_validation_order_verified=True,
        complete_five_folds=False,remaining_folds_started=False,seconds=time.perf_counter()-started,
        finished_at=datetime.now().astimezone().isoformat())
    recipe=dict(scores=selected_scores,choices=choices,selected_new_POR=positive,best_v25_candidate=best['mode'],
        feature_manifest='results/deep_learning/feature_structure_v20_20260928/feature_manifest.json' if positive else None,
        original_fifth_development_selection=True,complete_five_folds=False,remaining_folds_started=False,old_delivery_unchanged=True)
    atomic_json(STUDY/'assembled_report.json',assembled);atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    timing=dict(explicit_concurrent_jobs=2,launch_to_last_finish_seconds=(max(ends)-min(starts)).total_seconds(),
        per_run_seconds={r['mode']:r['final']['seconds'] for r in progress['rows']},
        peak_allocated_MiB={r['mode']:r['final']['peak_allocated_MiB'] for r in progress['rows']},
        final_updated_checkpoint_replay_seconds=assembled['seconds'],
        scope='launch/setup/cached inputs/training/validation/selected restore; excludes implementation and final replay; no throughput speedup claim')
    atomic_json(STUDY/'timing.json',timing)
    text=['# v25：回到80.274135的局部改进结果','',f"完成：{datetime.now().astimezone().isoformat()}",'',
        '## 当前决定','',
        '本批直接加载并固定原冠军全部权重，只训练新增位置头。冷启动替换、扩容及专家监督的自动延伸已停止，原文件和旧交付没有删除或覆盖。',
        f"本批最高实测TOTAL={best['scores']['TOTAL']:.12f}；相对原冠军变化={best['delta_reference_TOTAL']:+.12f}；研究配方采纳新POR={positive}。单折微小正分差不代表稳定收益或交付升级。",'',
        '|方案|更新后最高POR|更新后实际TOTAL|包含第0轮的选定TOTAL|最佳/停止轮次|',
        '|---|---:|---:|---:|---:|',
        f"|原冠军|{refscore['POR']:.9f}|{refscore['TOTAL']:.9f}|{refscore['TOTAL']:.9f}|历史|"]
    for c in candidates:
        text.append(f"|{c['mode']}|{c['post_update_scores']['POR']:.9f}|{c['post_update_scores']['TOTAL']:.9f}|{c['scores']['TOTAL']:.9f}|{c['best_epoch']}/{c['stopped_epoch']}|")
    text+=['','## 只改了哪里','',
        '每个专家的128维原隐藏特征与63维幅值/形态侧输入经191→32→8小头，生成受0.25倍固定分量标准差约束的新增位置偏移。两组只差输入表示，配置差异严格为mode/input_view；全部初始权重及144665点预测相同。总新参数12816，全部旧1290490参数固定。原两个骨干、校准、路由、默认logit、混合权重、尺度和point32残差不变。',
        '形态特征来自原9曲线的9/51/201点局部标准化残差、尺度差及变化量；幅值对照为等维幅值统计。全部全局常数由64训练井拟合，原206主特征不改，侧输入不进入旧骨干。',
        '原64/16井、Q50、POR最终混合似然/3、256格/800动作单覆盖解码；新增头lr3e-4、AdamW wd.01、clip2、cosine80×64至1e-6、80/18/min10、bs64×64步。没有额外学习率或窗口扫描。','',
        '## 实测归因','',
        f"最终选定的shape减amplitude：TOTAL={comparison['TOTAL_delta']:+.12f}。排除第0轮的更新后最好模型比较：TOTAL={updated_comparison['TOTAL_delta']:+.12f}。第0轮回退不称训练收益。"]
    for c in candidates:
        p=c['updated_population']
        text.append(f"{c['mode']}更新后相对冠军：TOTAL{p['TOTAL_delta']:+.12f}，默认段贡献{p['default_TOTAL_contribution']:+.12f}，非默认段贡献{p['numeric_TOTAL_contribution']:+.12f}；{p['well_wins']}井提高/{p['well_losses']}井下降；选择未校正整井重采样95%区间{p['exploratory_selection_unadjusted_TOTAL_95']}。")
    text+=['同一开发折已反复选模，两个侧输入和新头的整体效果不能外推为所有形态特征有效；本次只采用已经完成的固定协议，不再追加此小头的宽度/限制系数扫描。','',
        '## 验证与保留','',
        '两份真实更新后的最佳检查点从磁盘重建，全部144665点精确复现；非位置输出在固定训练输入上对照新支路关闭时逐元素不变。旧参数在最终训练与选定状态都未更新。最终输出动作仍可能因位置改变而变化。',
        '新POR与已验证PERM/SW数组实际合成后调用原metric；未声称重新加载未变的其他目标。没有改变行序、逐井选模或其他原始折。',
        '无第三个候选、无自动续跑，旧交付保持不变。停止的历史路线仍保留紧凑证据，未删除文件。后续只围绕保留冠军与有直接实测价值的局部调整；新实验尚未启动。',
        '完整OOF>82和第五折严格>80.5仍未达成。','',
        '## 本地证据','',
        'assembled_report.json、CURRENT_RECIPE.json、progress.json、METHOD.md及各outputs中的config/gradient_scope/validation_history/fold_5_report；模型代码champion_shape_v25.py，训练代码train_champion_shape_v25.py。',
        f"两项启动至结束墙钟{timing['launch_to_last_finish_seconds']:.3f}秒，单项{timing['per_run_seconds']}，最终重载{assembled['seconds']:.3f}秒另计。",
        '图谱由run_champion_shape_v25.py status --record增量维护并build_atlas.py刷新；历史训练代码保持不变。']
    final='\n'.join(text)+'\n';(STUDY/'FINAL.md').write_text(final,encoding='utf-8')
    marker='## v25 冠军原权重上的位置修正已完成'
    brief=marker+'\n'+datetime.now().astimezone().isoformat()+f"。amplitude={candidates[0]['scores']['TOTAL']:.12f}，shape={candidates[1]['scores']['TOTAL']:.12f}，保留研究TOTAL={selected_scores['TOTAL']:.12f}。仅12816新参数，旧模型冻结和更新后最佳全量重载已验证。冷启动/扩容停止延伸，无其他折/新队列/交付更改。详见champion_shape_v25_20260928/FINAL.md。"
    for path in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):
        append_once(path,marker,brief)
    current='# v25状态：已完成\n\n目标：直接改进80.274135，停止无收益替换路线。\n\n已完成：两项小型位置头、初始函数保持、真实梯度、冻结范围、最佳重载及实际评分。\n\n'+brief+'\n\n计划：保留完成证据，后续只从当前有效配方出发。\n\n未做：其他通道/原其他折/独立重复验证。\n\n完成后选项：只在新授权下做有限局部改进；无自动启动。\n'
    (STUDY/'PROJECTS.md').write_text(current,encoding='utf-8')
    for mode in MODES:
        out=ROOT/'outputs'/f'pure_dl_champion_shape_v25_{mode}_fold5'
        append_once(out/'README.md',marker,brief);(out/'PROJECTS.md').write_text(current,encoding='utf-8')
    atomic_json(STUDY/'integrity_summary.json',dict(training_failures=[r['failure'] for r in progress['rows'] if r['failure']],
        original_parameters_frozen=True,post_update_disk_replays_exact=True,other_targets_unchanged=True,
        cold_scaling_routes_not_restarted=True,remaining_folds_started=False,old_deliveries_changed=False))
    print('FINAL_REPORT',STUDY/'FINAL.md')
    print('SELECTED',json.dumps(recipe,ensure_ascii=False))
    print('COMPARISON',json.dumps({k:v for k,v in comparison.items() if k!='per_well'},ensure_ascii=False))
    print('UPDATED_COMPARISON',json.dumps({k:v for k,v in updated_comparison.items() if k!='per_well'},ensure_ascii=False))
    print('ACTIVE_TRAINERS',active_jobs())


if __name__=='__main__':main()
