"""Finalize v22 using actual saved results; never mutate trained source/weights."""
from pathlib import Path
from datetime import datetime
import json, subprocess, sys
from report_frozen_growth_v22 import ROOT, STUDY, MODES, read_json
from launch_latent_gate_v17 import active_jobs, append_once


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if active_jobs(): raise RuntimeError('Training still active')
    subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).with_name('report_frozen_growth_v22.py')),'--record'],check=True,stdout=subprocess.DEVNULL)
    progress=read_json(STUDY/'progress.json');assembled=read_json(STUDY/'assembled_report.json')
    assert progress['all_complete'] and assembled['all_updated_checkpoints_disk_replayed']
    rows={r['mode']:r for r in progress['rows']};candidates={c['mode']:c for c in assembled['candidates']}
    start=min(datetime.fromisoformat(read_json(ROOT/'outputs'/r['id']/'launch.json')['started_at']) for r in progress['rows'])
    end=max(datetime.fromisoformat(r['final']['finished_at']) for r in progress['rows'])
    timing=dict(explicit_parallel_jobs=2,launch_to_both_finish_seconds=(end-start).total_seconds(),
        per_run_seconds={m:rows[m]['final']['seconds'] for m in MODES},
        peak_allocated_MiB={m:rows[m]['final']['peak_allocated_MiB'] for m in MODES},
        new_updated_checkpoint_replay_seconds=assembled['seconds'],
        scope='Wall includes launch/import/cache load/training/epoch evaluation/selected restore; excludes implementation and final independent updated-checkpoint replay. Concurrent overlap differs from v21; no speedup claim.')
    (STUDY/'timing.json').write_text(json.dumps(timing,ensure_ascii=False,indent=2),encoding='utf-8')
    text=['# v22：固定原权重后的Scaling-up结果','',f"完成记录：{datetime.now().astimezone().isoformat()}",'',
        '## 本轮接续状态','',
        '本轮开始时，本地已有v21三档完整训练记录，结束于2026-09-28 21:18（UTC+8）。没有重复启动v21。本轮新实现并执行v22两项：只训练新增容量，保留全部原1290490参数。',
        '此前聊天附件仅提出FFN扩宽；实际本地v21还扩大了POR位置残差32→64/128。本轮采用与已完成v21完全相同的扩容模型，参数以真实实例计数为准。','',
        '## 关键结论','',
        f"当前实际保留TOTAL={assembled['selected_scores']['TOTAL']:.12f}；新增POR被采纳={assembled['selected_new_POR']}。两个v22候选都在第18轮早停，最佳为第0轮。只训练新增容量减轻了全量微调的退步，但没有超过原冠军。",'',
        '## 训练后最好成绩：排除第0轮','',
        '|来源|更新范围|内部宽度|模型总参数|可训练参数|训练后最好POR|实际TOTAL|相对原冠军TOTAL|','|---|---|---:|---:|---:|---:|---:|---:|']
    for mode in ('base1','ffn2','ffn4'):
        prev=read_json(ROOT/'outputs'/f'pure_dl_champion_scaling_v21_{mode}_fold5/fold_5_report.json')
        cfg=read_json(ROOT/'outputs'/f'pure_dl_champion_scaling_v21_{mode}_fold5/config.json')
        s=prev['best_after_update']
        text.append(f"|本地已完成v21|全部参数|{mode}|{cfg['trainable_parameters']}|{cfg['trainable_parameters']}|{s['target_score']:.9f}|{s['reference_replacement_total']:.9f}|{s['reference_replacement_total']-assembled['reference_scores']['TOTAL']:+.9f}|")
    for mode in MODES:
        c=candidates[mode];s=c['post_update_scores']
        text.append(f"|本轮新训练v22|仅新增参数|{mode}|{c['total_parameters']}|{c['trainable_parameters']}|{s['POR']:.9f}|{s['TOTAL']:.9f}|{c['updated_delta_reference_TOTAL']:+.9f}|")
    text+=['','全部五项的训练后峰值都在第1轮；若把初始模型纳入选择，全部回退80.274135。回退分数不是新容量的训练收益。基线1倍在全冻结条件下无新增参数，因此v22不虚构一个有更新的1倍实验。','',
        '## 本轮模型及匹配范围','',
        '复用v21双专家、原校准、路由、18个残差块、原位置残差与原数据缓存。仅.extra_参数进入优化器，旧参数不接收梯度，不受权重衰减影响。没有通过no_grad切断冻结层，因此梯度可以穿过原层到达较早的新增通路。',
        '2/4倍是FFN内部宽度，模型总参数分别1981498/3363514；新增可训练参数分别691008/2073024。位置残差也随之变宽，两处容量共同变化，尚未分离各自贡献。外部通道、注意力和数据规模未扩大。',
        '与v21同尺寸保持相同原权重和新参数初始化、Q50、64/16井、输入/损失/解码、学习率和80/18预算及train/eval行为。只改变可训练参数集合；训练器的状态记录新实现，不保证逐位训练轨迹相同。不同尺寸本身也改变随机运算形状。','',
        '## 结果解释','']
    for mode in MODES:
        c=candidates[mode];d=c['post_update_population']
        text.append(f"{mode}：比对应v21训练后最高TOTAL提高{c['updated_delta_matching_v21_TOTAL']:+.9f}，但相对原冠军仍为{c['updated_delta_reference_TOTAL']:+.9f}。真实默认段POR差{d['default_POR_delta']:+.9f}，非默认段POR差{d['numeric_POR_delta']:+.9f}；{d['well_wins']}井提高、{d['well_losses']}井下降。")
    text+=['','固定原参数减轻了退步，但新增通路仍会改变中间特征和最终分布；不能因为参数被冻结就认为原表示/默认概率/尺度完全不变。当前结果只限定于这个热启动、更新范围和预算，不能证明更大模型从头训练或其他扩容位置没有价值。',
        '本次只有一个反复使用的开发折与一个种子，不是独立泛化验证，也没有拟合Scaling Law。不以第0轮相同分数声称不同容量等效。','',
        '## 实际验证','',
        '两项训练前的144665点预测完全复现原冠军。真实反向确认原参数梯度全为None，新增骨干、新增读出及新增输入投影有非零梯度。训练结束前和最佳恢复后，都确认原参数/缓冲区逐元素不变。',
        '两份训练后第1轮检查点已从磁盘独立构建并推理，全部144665点精确复现。包含第0轮的选定检查点仍为原功能，输出投影零状态和最终预测数组也已检查。',
        '实际调用原metric计算已更新与选定三列预测的总分。PERM/SW复用此前已验证数组，本轮未重复加载未变旧模型。原验证行序不变，无逐行/逐井真值模型选择。训练代码指纹与配置一致。','',
        '## 保留、未做与下一步','',
        '继续保留原v5/v8配方。未采用零输出扩容网络作为额外推理开销；用户交付包、原检查点及其他四折不变。没有自动追加实验或队列。',
        '本批完成的是末期热启动内部容量增长；从头训练同一双专家的容量对照、整体通道/深度扩展、PERM/SW独立扩容与完整五折均尚未执行。若继续，应先选择一条新的固定对照路线，不因当前负结果无限增宽或只改学习率。',
        '原第五折严格超过80.5、完整原五折OOF超过82仍未完成。','',
        '## 耗时及来源','',
        f"两项显式并发，启动到均结束墙钟{timing['launch_to_both_finish_seconds']:.3f}秒；单项{timing['per_run_seconds']}；峰值分配显存MiB={timing['peak_allocated_MiB']}。最终独立重载及数据准备{assembled['seconds']:.3f}秒另计，不作并发加速比。",
        '源码：frozen_growth_v22.py、train_frozen_growth_v22.py、launch_frozen_growth_v22.py；沿用champion_scaling_v21.py及train_scaling_v21.py中评价/损失，旧源码不改。',
        '原始证据：outputs/pure_dl_frozen_growth_v22_ffn2_fold5和ffn4_fold5下config、initial_verification、gradient_scope、validation_history、fold_5_report及检查点。',
        '本目录：METHOD.md、progress.json、assembled_report.json、CURRENT_RECIPE.json、timing.json。图谱节点由report_frozen_growth_v22.py --record维护，未修改历史curate.py。']
    final='\n'.join(text)+'\n'
    (STUDY/'FINAL.md').write_text(final,encoding='utf-8')
    marker='## v22 固定原权重扩容实验已完成'
    entry=marker+'\n'+datetime.now().astimezone().isoformat()+f"。2倍更新后TOTAL={candidates['ffn2']['post_update_scores']['TOTAL']:.9f}；4倍={candidates['ffn4']['post_update_scores']['TOTAL']:.9f}。均18轮停止/最佳第0轮，保留{assembled['selected_scores']['TOTAL']:.9f}。新通路梯度、旧参数冻结、训练后检查点全144665点磁盘重载通过。未运行其他目标/原其他折，不改交付；详见frozen_growth_v22_20260928/FINAL.md。"
    for path in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):
        append_once(path,marker,entry)
    for mode in MODES:
        out=ROOT/'outputs'/rows[mode]['id']
        for name in ('README.md','PROJECTS.md'):append_once(out/name,marker,entry)
    summary=dict(training_failures=[r['failure'] for r in progress['rows'] if r['failure']],
        original_parameters_exact=True,trained_checkpoint_replays_exact=True,all_finished=True,
        historical_v21_not_rerun=True,other_folds_started=False,delivery_changed=False)
    (STUDY/'integrity_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    graphpath=ROOT/'docs/deep_learning/atlas/curation.json';graph=read_json(graphpath)
    deadid='dead:frozen_growth_v22_warm_internal_scaling'
    dead=dict(id=deadid,title='v22末期内部扩容：本设置未获增益',stage='architecture',parents=[r['id'] for r in progress['rows']],
        summary='固定原129万参数，2/4倍新增容量训练后TOTAL80.269258/80.263250，低于80.274135；均回退第0轮。',
        reason='关闭当前末期热启动配方的进一步增宽；不否定从头训练规模效应或不同扩容位置。')
    graph['deadends']=[r for r in graph.get('deadends',[]) if r.get('id')!=deadid]+[dead]
    graphpath.write_text(json.dumps(graph,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FINAL_REPORT',STUDY/'FINAL.md');print('SELECTED',assembled['selected_scores']);print('ACTIVE_TRAINERS',active_jobs())
    print('TIMING',json.dumps(timing,ensure_ascii=False))


if __name__=='__main__':main()
