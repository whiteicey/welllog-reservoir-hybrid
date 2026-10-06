"""Finalize only completed, replayed v23 evidence; preserve all older models."""
from pathlib import Path
from datetime import datetime
import json,subprocess,sys
import numpy as np
from report_scratch_scaling_v23 import ROOT,STUDY,MODES,load,save
from launch_latent_gate_v17 import active_jobs,append_once


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if active_jobs():raise RuntimeError('Training still active')
    subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).with_name('report_scratch_scaling_v23.py')),'--record'],check=True,stdout=subprocess.DEVNULL)
    p=load(STUDY/'progress.json');a=load(STUDY/'assembled_report.json')
    assert p['all_complete'] and a['all_new_checkpoints_disk_replayed']
    rows={r['mode']:r for r in p['rows']};cands={r['mode']:r for r in a['candidates']};comparison=a['capacity_comparison']
    starts=[];ends=[];timings={}
    for mode,row in rows.items():
        out=ROOT/'outputs'/row['id'];r=row['final'];h=load(out/'validation_history.json')
        starts.append(datetime.fromisoformat(load(out/'launch.json')['started_at']));ends.append(datetime.fromisoformat(r['finished_at']))
        timings[mode]=dict(seconds=r['seconds'],epochs=r['stopped_epoch'],updates=r['stopped_epoch']*64,
            epoch_median_seconds=float(np.median([x['seconds'] for x in h])),peak_allocated_MiB=r['peak_allocated_MiB'])
    timing=dict(explicit_parallel_jobs=2,first_launch_to_last_finish_seconds=(max(ends)-min(starts)).total_seconds(),runs=timings,
        final_inference_seconds=a['seconds'],
        scope='Process time includes setup/cache/training/each validation/final selected restoration. Final fresh validation+training inference is separate. Same step budget ceiling, not same realized updates/FLOPs or isolated GPU throughput.')
    save(STUDY/'timing.json',timing)
    ref=a['reference_scores']
    lines=['# v23：POR双专家从头训练的容量实验','',f"完成记录：{datetime.now().astimezone().isoformat()}",'',
        '## 本轮回答的问题','',
        'v21/v22测试末期热启动扩容；v23将原尺寸与扩容后的私有POR双专家都从随机网络权重训练，检验新增容量在表示形成阶段的作用。没有加载旧网络参数。只读已有训练侧预处理、Q50及解码统计。','',
        '## 原第五折实测总分','',
        '|模型|网络参数|POR|PERM|SW|TOTAL|相对80.274135|最佳/完成轮次|','|---|---:|---:|---:|---:|---:|---:|---|',
        f"|原v5/v8|1290490（POR）|{ref['POR']:.9f}|{ref['PERM']:.9f}|{ref['SW']:.9f}|{ref['TOTAL']:.9f}|0|历史|"]
    for mode in MODES:
        c=cands[mode];s=c['scores']
        lines.append(f"|v23 {mode}|{c['parameter_count']}|{s['POR']:.9f}|{s['PERM']:.9f}|{s['SW']:.9f}|{s['TOTAL']:.9f}|{c['delta_reference_TOTAL']:+.9f}|{c['best_epoch']}/{c['stopped_epoch']}|")
    lines+=['',f"新增POR采纳={a['selected_new_POR']}；最终保留TOTAL={a['selected_scores']['TOTAL']:.12f}。80轮结束表示预算耗尽，其他轮数按原patience18停止；均使用实际恢复的最佳检查点。",'',
        '## 容量对照','',
        f"扩容减原尺寸：POR{comparison['POR_delta']:+.9f}，TOTAL{comparison['TOTAL_delta']:+.9f}；{comparison['well_wins']}/16井提高，{comparison['well_losses']}井下降。",
        f"真实默认段POR变化{comparison['default_POR_delta']:+.9f}，非默认段变化{comparison['numeric_POR_delta']:+.9f}。折算TOTAL贡献分别{comparison['default_TOTAL_contribution']:+.9f}、{comparison['numeric_TOTAL_contribution']:+.9f}。",
        f"探索性整井重采样TOTAL区间{comparison['exploratory_selection_unadjusted_TOTAL_95']}，未校正本次早停及历史重复开发的选择影响，不作为独立确认。",'',
        '## 训练与验证子集','',
        '|模型|训练POR|验证POR|训练非默认POR|验证非默认POR|训练默认POR|验证默认POR|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for mode in MODES:
        t=cands[mode]['fit'];v=cands[mode]['validation']
        lines.append(f"|{mode}|{t['POR']:.6f}|{v['POR']:.6f}|{t['numeric_POR']:.6f}|{v['numeric_POR']:.6f}|{t['default_POR']:.6f}|{v['default_POR']:.6f}|")
    lines+=['','这些训练集分数来自最终已选检查点的独立完整训练井推理，并非随机训练批次的近似分。没有用它们再选择新模型。训练/验证样本组成不同，不能只看总体差判断泛化。','',
        '## 网络、初始化和优化协议','',
        f"共同随机参数一致={p['common_random_state_and_training_protocol_match']}；初始144665点预测完全一致={p['all144665_initial_predictions_equal']}。整网参数比={p['parameter_ratio']:.9f}，不是2倍整网。",
        '两专家各有9个多尺度残差块，两个视图、输入校准、路由、注意力和外部通道保持相同。ffn2只增加块内FFN与位置残差单元；位置残差32→64，两个部位共同变化，不能分开归因。每块新增输出投影为零，初始函数与随机原尺寸模型一致。',
        '原九曲线206特征+相对深度、原64/16井、Q50、单任务最终混合POR似然/3及256格/800动作/单覆盖解码一致；无其他目标损失。每轮相同抽样生成器，batch64×64步、max80/pat18/min10，AdamW6e-4、wd.01、cosine到1e-5、clip2。扩容增加算力，实验不声称等FLOPs或独立GPU吞吐对照。',
        '原80.22来自多任务训练、每专家自身监督、深层辅助头、独立路由监督和动态Q；本轮保留当前私有双专家的单目标最终混合损失、固定Q50。原模型与本轮存在训练历史差异；不能将相对80.27的差全部归因于容量，也不能把本轮叫完整原训练配方复现。',
        '虽然网络权重从头初始化，Q50和预处理沿用此前同64训练井产物，因此不是整条pipeline完全无训练历史。', '',
        '## 路由诊断（不作额外候选）','']
    for mode in MODES:
        route=cands[mode]['validation']['input_router']
        lines.append(f"{mode}验证集输入路由平均权重={route['mean_weights']}；平均熵={route['mean_entropy_nats']:.6f} nats，最大权重大于0.95的点占{route['max_weight_over_095_fraction']:.4%}。")
        responsibility=cands[mode]['validation']['posterior_responsibility']
        lines.append(f"{mode}在真实非默认标签处的专家后验责任均值={responsibility['nondefault_mean_weights']}，最大责任大于0.95的点占{responsibility['nondefault_max_over_095_fraction']:.4%}。该诊断在训练启动后补充，仅解释已选终点，不用于超参或检查点选择。")
    lines+=['路由权重仅说明输入选择行为，不等于两个专家的因果贡献，也不是目标条件后验责任。未根据该诊断切换模型或添加评分候选。','',
        '## 实际验证与状态','',
        '两份新最佳模型从磁盘独立重建，原144665点预测逐元素复现，按原行序与固定PERM/SW实际组合调用原metric。两条未变目标的旧模型不重复GPU重载，复用既有已验证预测。',
        '实际训练反向确认骨干和新增容量收到梯度；相同共有参数起点与训练窗口设置已核对。训练期间未修改模型/训练源码，完成时再次核对配置中源码指纹。',
        '未修改原数据、原权重或交付。没有自动队列、第三候选或其他折。完整原五折OOF>82与原第五折严格>80.5目标尚未完成。', '',
        '## 下一步边界','',
        '本次以固定单任务训练协议对照两种容量；结果只约束已测内部宽度/优化预算。若需要回答原80.22完整训练配方的规模收益，仍需单独重建其多目标/专家辅助监督与动态Q训练，不能以本结果代替。整体通道、深度、其他目标和跨种子验证本轮均未执行。',
        '下一步优先根据非默认段的训练—验证差及路由行为，在固定规模上检验专家监督的形成方式；不把无收益直接解释为参数不够，不继续无界增宽。该新对照尚未启动。','',
        '## 记录与开销','',
        f"两个任务显式并发，首次启动至最后完成墙钟{timing['first_launch_to_last_finish_seconds']:.3f}秒。各项记录{json.dumps(timings,ensure_ascii=False)}。最终独立推理{a['seconds']:.3f}秒另计。",
        '代码scratch_scaling_v23.py、train_scratch_scaling_v23.py；本目录METHOD.md、progress.json、assembled_report.json、CURRENT_RECIPE.json保存方法与结果。各outputs目录保存config、history、model_best和last_state。',
        '图谱中文节点通过report_scratch_scaling_v23.py --record维护，再build_atlas.py更新静态快照；历史训练代码不改。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v23 从头双专家扩容对照已完成'
    entry=marker+'\n'+datetime.now().astimezone().isoformat()+f"。base1 TOTAL={cands['base1']['scores']['TOTAL']:.9f}；ffn2 TOTAL={cands['ffn2']['scores']['TOTAL']:.9f}；容量差{comparison['TOTAL_delta']:+.9f}，当前保留{a['selected_scores']['TOTAL']:.9f}。两模型从随机权重训练/最佳磁盘重载/训练与验证诊断均完成。其他目标/原其他折及原多任务完整配方未执行；无自动后续。记录scratch_scaling_v23_20260928/FINAL.md。"
    for path in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):
        append_once(path,marker,entry)
    for mode in MODES:
        out=ROOT/'outputs'/rows[mode]['id']
        for name in ('README.md','PROJECTS.md'):append_once(out/name,marker,entry)
    save(STUDY/'integrity_summary.json',dict(training_failures=[r['failure'] for r in p['rows'] if r['failure']],
        initial_functions_equal=p['all144665_initial_predictions_equal'],trained_sources_unchanged=True,
        actual_new_checkpoints_replayed=True,all_complete=True,old_deliveries_unchanged=True,remaining_folds_started=False))
    graphpath=ROOT/'docs/deep_learning/atlas/curation.json';graph=load(graphpath)
    if not a['selected_new_POR']:
        deadid='dead:scratch_scaling_v23_direct_replacement'
        dead=dict(id=deadid,title='v23单目标从头扩容：本配置未替代冠军',stage='architecture',parents=[r['id'] for r in p['rows']],
            summary=f"base1/ffn2实测TOTAL={cands['base1']['scores']['TOTAL']:.6f}/{cands['ffn2']['scores']['TOTAL']:.6f}；继续保留80.274135。",
            reason='关闭本批配置的直接替换；不否定原多任务配方/其他尺度/其他训练条件的容量收益。')
        graph['deadends']=[x for x in graph.get('deadends',[]) if x.get('id')!=deadid]+[dead];save(graphpath,graph)
    print('FINAL_REPORT',STUDY/'FINAL.md');print('SELECTED',a['selected_scores'])
    print('CAPACITY',json.dumps({k:v for k,v in comparison.items() if k!='per_well'},ensure_ascii=False))
    print('TIMING',json.dumps(timing,ensure_ascii=False));print('ACTIVE_TRAINERS',active_jobs())


if __name__=='__main__':main()
