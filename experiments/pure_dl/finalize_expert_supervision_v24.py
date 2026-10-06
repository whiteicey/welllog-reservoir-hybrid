"""Finalize only completed v24 evidence; no changes to trained model code."""
from datetime import datetime
from pathlib import Path
import json,subprocess,sys
import numpy as np
from report_expert_supervision_v24 import ROOT,STUDY,MODES,load,save
from launch_latent_gate_v17 import active_jobs,append_once


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if active_jobs():raise RuntimeError('Trainers still active')
    subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).with_name('report_expert_supervision_v24.py')),'--record'],check=True,stdout=subprocess.DEVNULL)
    progress=load(STUDY/'progress.json');a=load(STUDY/'assembled_report.json')
    assert progress['all_complete'] and a['all_new_checkpoints_disk_replayed']
    rows={r['mode']:r for r in progress['rows']};candidates={r['mode']:r for r in a['candidates']}
    d=a['supervision_comparison'];ref=a['reference_scores']
    starts=[];ends=[];costs={}
    for mode in MODES:
        r=rows[mode];out=ROOT/'outputs'/r['id'];h=load(out/'validation_history.json')
        starts.append(datetime.fromisoformat(load(out/'launch.json')['started_at']))
        ends.append(datetime.fromisoformat(r['final']['finished_at']))
        costs[mode]=dict(seconds=r['final']['seconds'],epochs=r['final']['stopped_epoch'],updates=64*r['final']['stopped_epoch'],
            median_epoch_seconds=float(np.median([x['seconds'] for x in h])),peak_allocated_MiB=r['final']['peak_allocated_MiB'])
    timing=dict(explicit_parallel_jobs=2,first_launch_to_last_finish_seconds=(max(ends)-min(starts)).total_seconds(),
        runs=costs,final_inference_seconds=a['seconds'],
        scope='Launch/import/cache/training/epoch validation/selected restore included. Independent final replay and full training diagnostics separate; not an isolated throughput experiment.')
    save(STUDY/'timing.json',timing)
    positive=d['TOTAL_delta']>0
    conclusion=('专家独立监督在本批相同网络/优化器条件下优于混合责任监督。' if positive else '专家独立监督在本批相同网络/优化器条件下没有优于混合责任监督。')
    continuation=('下一步待验证：采用相同专家独立监督，保持其他设置，再复测原尺寸与扩容尺寸，检查监督方式和容量是否存在交互。当前尚未启动。' if positive else '下一步待验证：在固定规模上单独检验深层辅助监督或训练任务设置，不将本结果直接归为容量不足；当前尚未启动。')
    lines=['# v24：固定容量的专家监督对照','',f"完成时间：{datetime.now().astimezone().isoformat()}",'',
        '## 完成结果','',conclusion,
        f"当前采纳新POR={a['selected_new_POR']}；保留TOTAL={a['selected_scores']['TOTAL']:.12f}。",'',
        '|POR模型|参数量|POR|PERM|SW|实际TOTAL|相对冠军TOTAL|最佳/完成轮次|',
        '|---|---:|---:|---:|---:|---:|---:|---|',
        f"|原v5/v8|1290490|{ref['POR']:.9f}|{ref['PERM']:.9f}|{ref['SW']:.9f}|{ref['TOTAL']:.9f}|0|历史|"]
    for mode in MODES:
        c=candidates[mode];s=c['scores']
        lines.append(f"|{mode}|{c['parameter_count']}|{s['POR']:.9f}|{s['PERM']:.9f}|{s['SW']:.9f}|{s['TOTAL']:.9f}|{c['delta_reference_TOTAL']:+.9f}|{c['best_epoch']}/{c['stopped_epoch']}|")
    lines+=['','## 匹配比较与边界','',
        f"独立减混合：POR={d['POR_delta']:+.9f}，TOTAL={d['TOTAL_delta']:+.9f}，{d['well_wins']}/16井上升、{d['well_losses']}井下降。",
        f"真实默认段POR变化{d['default_POR_delta']:+.9f}，非默认段{d['numeric_POR_delta']:+.9f}；对应TOTAL贡献{d['default_TOTAL_contribution']:+.9f}和{d['numeric_TOTAL_contribution']:+.9f}。",
        f"整井重采样的探索性TOTAL区间={d['exploratory_selection_unadjusted_TOTAL_95']}；未校正早停、单次随机性和多轮使用第五折选模，不代表独立泛化确认。",
        f"参数和初始状态匹配={progress['same_parameters_and_initial_state']}；所有144665点初始预测一致={progress['all144665_initial_predictions_equal']}；配置差异={progress['config_differences']}。",
        '网络不增宽，输入、视图、校准、路由、point32头、Q50、损失基本定义、训练样本和解码保持。两组都有相同专家/路由独立AdamW和分组裁剪，只改变专家目标；v23使用单优化器/全局裁剪，因此不能用其差替代本批匹配比较。', '',
        '## 监督方式','',
        '设i为有效监督点、e为两位专家之一。p_ie是完整专家似然（默认点为概率质量，非默认点为连续密度）；a_i属于R^2为非负且和为1的路由向量；p_i=sum_e a_ie p_ie。',
        'mixture专家用L_mix=-sum_i q_i log(p_i)/(3N)，计算专家损失时detach路由。其对专家e的梯度由责任r_ie=a_ie p_ie/p_i加权。',
        'independent专家用L_ind=-sum_i q_i[log(p_i1)+log(p_i2)]/(6N)，各自接受全体监督并平均。q_i为原Q50，N为有效监督点数，/3沿用原损失尺度。',
        '路由两组均用detach专家输出的L_mix；梯度不能回传给专家。mixture记录的总loss=专家loss+路由loss=2L_mix，但每组参数只接收一次对应梯度，不是双倍优化。不同模式训练loss不可直接横向当同一个似然指标。', '',
        '## 最佳检查点的完整训练与验证诊断','',
        '|模式|训练POR|验证POR|训练非默认POR|验证非默认POR|验证默认POR|','|---|---:|---:|---:|---:|---:|']
    for mode in MODES:
        c=candidates[mode];t=c['fit'];v=c['validation']
        lines.append(f"|{mode}|{t['POR']:.6f}|{v['POR']:.6f}|{t['numeric_POR']:.6f}|{v['numeric_POR']:.6f}|{v['default_POR']:.6f}|")
    lines+=['','以下为选定检查点之后的解释性统计，不用于再选择权重、路由阈值或新模型。']
    for mode in MODES:
        v=candidates[mode]['validation'];r=v['input_router'];p=v['posterior_responsibility']
        lines.append(f"{mode}：验证路由均值{r['mean_weights']}，路由最大权重>0.95占{r['max_weight_over_095_fraction']:.6%}；非默认点后验责任最大值>0.95占{p['nondefault_max_over_095_fraction']:.6%}，非默认责任熵{p['nondefault_mean_entropy_nats']:.6f} nats。")
    lines+=['责任统计描述已学分工，不等价于专家的因果贡献，也不能仅凭高度分工认定坏塌缩。独立监督可能同时影响专家能力、默认门控及路由；本批没有再细分这些中介机制。','',
        '## 实际验证与保留','',
        '真实训练第二步验证两专家和路由均有非零梯度，专家损失到路由参数、路由损失到专家参数的交叉梯度为None。无其他目标损失或深层辅助头。完整原80.22多任务/辅助监督/动态Q训练尚未复现。',
        '两份新最佳检查点已从磁盘独立重建，全部144665验证预测逐元素复现；按原行序拼接固定PERM/SW数组，实际调用原metric。两个未改目标的旧模型不重复GPU重载，沿用原有验证证据。',
        '记录中的训练源码指纹保持一致。网络权重从随机初始化，数据/变换/质量权重沿用训练侧产物，不能称整条pipeline无历史训练。',
        '本批只有两项，没有追加第三候选、原其他四折或旧任务。原冠军/已交付包均未改。第80轮结束应记录预算耗尽，其余在满足patience18时自然早停。','',
        '## 下一步未做','',continuation,
        '其他通道训练、完整原多任务流程、跨种子及完整原五折OOF>82均未验证；当前不自动补折。','',
        '## 开销与证据','',
        f"两项显式并发，首次启动到最后完成墙钟{timing['first_launch_to_last_finish_seconds']:.3f}秒；单项记录{json.dumps(costs,ensure_ascii=False)}；最终独立验证和训练井推理{a['seconds']:.3f}秒另计。",
        '模型/损失：experiments/pure_dl/expert_supervision_v24.py；训练入口train_expert_supervision_v24.py；每项outputs下config、history、gradient_scope、fold_5_report及checkpoint。',
        '本目录progress.json、assembled_report.json、CURRENT_RECIPE.json是实际状态与评分；METHOD.md是预定义条件。图谱由report_expert_supervision_v24.py --record增量维护，build_atlas.py生成当前离线快照，历史训练源码不改。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v24 专家监督对照已完成'
    brief=marker+'\n'+datetime.now().astimezone().isoformat()+f"。mixture TOTAL={candidates['mixture']['scores']['TOTAL']:.9f}，independent TOTAL={candidates['independent']['scores']['TOTAL']:.9f}；匹配差{d['TOTAL_delta']:+.9f}，保留{a['selected_scores']['TOTAL']:.9f}。两模型梯度隔离/最佳磁盘重载/实际评分完成，未改交付或补折。详见expert_supervision_v24_20260928/FINAL.md。"
    for path in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):
        append_once(path,marker,brief)
    current='# v24当前状态：已完成\n\n目标：同容量比较专家监督并核算三通道得分。\n\n已完成：两项训练、相同初始预测、梯度隔离、最佳磁盘重载、全训练/验证井诊断、真实总分。\n\n结果：'+brief+'\n\n计划／尚未执行：'+continuation+'\n\n未完成：其他通道/跨种子/完整OOF。旧模型及交付未改，无自动续跑。\n'
    (STUDY/'PROJECTS.md').write_text(current,encoding='utf-8')
    for mode in MODES:
        out=ROOT/'outputs'/rows[mode]['id']
        append_once(out/'README.md',marker,brief)
        (out/'PROJECTS.md').write_text(current,encoding='utf-8')
    save(STUDY/'integrity_summary.json',dict(training_failures=[r['failure'] for r in progress['rows'] if r['failure']],
        same_initial_predictions=True,gradient_isolation_verified=True,all_new_models_replayed=True,
        historical_sources_unchanged=True,remaining_folds_started=False,delivery_changed=False))
    print('FINAL_REPORT',STUDY/'FINAL.md');print('SELECTED',a['selected_scores'])
    print('SUPERVISION_DELTA',json.dumps({k:v for k,v in d.items() if k!='per_well'},ensure_ascii=False))
    print('TIMING',json.dumps(timing,ensure_ascii=False));print('ACTIVE_TRAINERS',active_jobs())


if __name__=='__main__':main()
