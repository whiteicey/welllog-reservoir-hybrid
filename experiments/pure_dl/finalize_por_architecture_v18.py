"""Finalize v18 from verified artifacts, without changing trained models."""
from pathlib import Path
from datetime import datetime
import json,subprocess,sys
from launch_latent_gate_v17 import active_jobs,append_once

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/por_architecture_v18_20260928'


def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).with_name('report_por_architecture_v18.py')),'--record'],check=True,stdout=subprocess.DEVNULL)
    progress=read(STUDY/'progress.json');assembled=read(STUDY/'assembled_report.json')
    assert progress['all_complete'] and assembled['all_new_POR_reloaded']
    bymode={x['mode']:x for x in progress['rows']};a,b=[bymode[m]['final'] for m in ('tcn104','latent64')]
    cands={x['mode']:x for x in assembled['candidates']};delta=assembled['comparison']
    launches=[read(ROOT/'outputs'/x['id']/'launch.json') for x in progress['rows']]
    start=min(datetime.fromisoformat(x['started_at']) for x in launches)
    end=max(datetime.fromisoformat(x['final']['finished_at']) for x in progress['rows'])
    timing=dict(explicit_parallel_jobs=2,launch_to_both_finish_seconds=(end-start).total_seconds(),
        per_run_seconds={m:x['final']['seconds'] for m,x in bymode.items()},
        peak_allocated_MiB={m:x['final']['peak_allocated_MiB'] for m,x in bymode.items()},
        final_replay_seconds=assembled['seconds'],
        scope='Wall includes process startup/data loading/train/epoch validation/final restored train-and-validation scoring; excludes code implementation, target preparation and later POR disk replay. No speedup claim.')
    (STUDY/'timing.json').write_text(json.dumps(timing,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# v18 POR私有骨干：最终实测结果','',f"记录时间：{datetime.now().astimezone().isoformat()}。",'',
        '## 已完成与固定保留','',
        '|方案|参数量|POR|三通道TOTAL|最佳轮次|实际完成轮次|','|---|---:|---:|---:|---:|---:|',
        f"|原v5/v8|历史模型|{assembled['reference_scores']['POR']:.9f}|{assembled['reference_scores']['TOTAL']:.9f}|历史|已验证|"]
    for mode in ('tcn104','latent64'):
        r=bymode[mode]['final'];c=cands[mode]
        lines.append(f"|{mode}|{r['trainable_parameters']}|{c['scores']['POR']:.9f}|{c['scores']['TOTAL']:.9f}|{r['best_epoch']}|{r['stopped_epoch']}|")
    lines+=['',f"最终选入新POR={assembled['selected_new_POR']}，当前TOTAL={assembled['selected_scores']['TOTAL']:.12f}。训练至第80轮属于达到预算上限，不能称patience触发的早停。",
        '', '## 匹配范围','',
        f"TCN104与latent64的参数差为{progress['parameter_difference']}，初始输出偏置相同={progress['same_initial_output_bias']}，初始全部预测相同={progress['initial_predictions_exactly_equal']}。两组完整网络权重形状不同，不能声称state_dict相同。",
        '两个候选均从随机权重学习，复用原64训练井拟合的特征与POR归一化/解码元数据，不加载旧模型权重。相同Student-t3八分量头、损失、输入、训练预算和数据窗口，Q=1。原v5模型的骨干容量、预训练与Q流程不同，和v5比较属于方案胜负，不是纯骨干单因素效应。',
        '', '## 原第五折机制比较','',
        f"latent相对TCN：POR{delta['latent_minus_tcn_POR']:+.9f}，TOTAL{delta['latent_minus_tcn_TOTAL']:+.9f}；{delta['well_wins']}/16井提高、{delta['well_losses']}井下降。默认段差{delta['default_POR_delta']:+.9f}，非默认段差{delta['numeric_POR_delta']:+.9f}。",
        f"探索性整井重采样的TOTAL区间{delta['exploratory_selection_unadjusted_TOTAL_95']}。第五折反复用于开发且本轮用于选模，该区间没有选择校正，不是独立泛化证明。",
        '', '|模型|训练POR|验证POR|训练非默认POR|验证非默认POR|','|---|---:|---:|---:|---:|']
    for mode in ('tcn104','latent64'):
        r=bymode[mode]['final']
        lines.append(f"|{mode}|{r['fit']['POR']:.6f}|{r['validation']['POR']:.6f}|{r['fit']['numeric_POR']:.6f}|{r['validation']['numeric_POR']:.6f}|")
    lines+=['','## 验证与边界','',
        '两份新POR检查点从磁盘独立重建，144665点预测逐元素复现。固定PERM/SW使用既有已验证预测数组，按完全相同的行序合并并实际调用原评分。没有重复GPU重载两个未变化的旧模型；其证据仍见v17 assembled_report。',
        '新头Student-t负对数似然与torch.distributions逐点对照通过；两个训练输入窗口的填充隔离和全空有限零输出检查通过。原CDF函数复用未更改。训练发生真实骨干梯度和参数更新，不因恢复旧第0轮而虚报收益。',
        '正式训练失败记录见各输出目录failure.json（若不存在表示未记录训练异常）。准备阶段工具读取曾超过max_bytes，另有内联命令因误含空格无法解析；均在训练之前，仅换为独立检查脚本，没有改数据结果。',
        '', '## 不继续的点与下一步（未启动）','',
        '本次结果只支持在当前容量/损失/预算下比较宽TCN与16潜变量架构，不能把SW上的结构收益直接推广到POR，也不能否定整个潜变量模型家族。',
        '不继续在同一结果附近无界扫描层数、宽度或学习率，不自动启动第三候选或其他折。后续应先核对非默认段的训练—验证差和局部多尺度特征保留，再决定采用多分辨率跳接或训练正则化的匹配对照。仅记录待检验方向，本批未实现/运行。',
        '完整五折目标>82和原第五折严格>80.5仍未完成，旧交付未改变。当前配方见CURRENT_RECIPE.json。',
        '', '## 文件与复核','',
        '模型：experiments/pure_dl/por_architecture_v18.py；数据/评价：por_data_v18.py；训练：train_por_architecture_v18.py。',
        '原始结果：各outputs目录的fold_5_report.json、config.json、validation_history.json、model_best.pt、last_state.pt。',
        '研究汇总：progress.json、assembled_report.json、data_manifest.json、model_metadata.pt、CURRENT_RECIPE.json。',
        f"两项显式并发墙钟{timing['launch_to_both_finish_seconds']:.3f}秒，单项{timing['per_run_seconds']}；另有最终重载{timing['final_replay_seconds']:.3f}秒。各项秒数不能相加当作并发墙钟。详见timing.json。",
        '图谱中文节点由report_por_architecture_v18.py --record可复现维护，build_atlas.py生成静态页面；单通道节点明确标记POR，不冒充TOTAL。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v18 POR结构对照完成'
    entry=marker+'\n'+datetime.now().astimezone().isoformat()+f"。TCN TOTAL={cands['tcn104']['scores']['TOTAL']:.9f}，latent TOTAL={cands['latent64']['scores']['TOTAL']:.9f}，保留TOTAL={assembled['selected_scores']['TOTAL']:.9f}。两个新POR磁盘重载精确，固定PERM/SW数组合并实测；原模型与其他折未改。后续结构/完整五折未运行，当前无新增队列。记录results/deep_learning/por_architecture_v18_20260928/FINAL.md。"
    for p in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):
        append_once(p,marker,entry)
    for row in progress['rows']:
        out=ROOT/'outputs'/row['id'];c=cands[row['mode']]
        text=marker+f"\n完成{row['final']['stopped_epoch']}轮，最佳{row['final']['best_epoch']}轮，POR={c['scores']['POR']:.9f}，实际组合TOTAL={c['scores']['TOTAL']:.9f}。全部新预测磁盘复现；仅本批两个候选，没有后续训练或交付覆盖。详细增益/未完成项见研究FINAL.md。"
        for name in ('README.md','PROJECTS.md'):append_once(out/name,marker,text)
    graph_path=ROOT/'docs/deep_learning/atlas/curation.json';graph=read(graph_path)
    if not assembled['selected_new_POR']:
        deadid='dead:por_architecture_v18_cold'
        dead=dict(id=deadid,title='v18冷启动POR：本设置未替代原模型',stage='architecture',parents=[x['id'] for x in progress['rows']],
            summary=f"近16万参数的TCN/POR潜变量模型均未超过旧POR{assembled['reference_scores']['POR']:.6f}。",
            reason='关闭本批配置的直接替换方案；不否定其他多尺度骨干、容量或目标专用全局表示。')
        graph['deadends']=[x for x in graph.get('deadends',[]) if x.get('id')!=deadid]+[dead]
        graph_path.write_text(json.dumps(graph,ensure_ascii=False,indent=2),encoding='utf-8')
    (STUDY/'tool_attempts.json').write_text(json.dumps(dict(training_failures=[x['failure'] for x in progress['rows'] if x['failure']],
        read_limit_error='AGENTS read retried with sufficient max_bytes',inline_command_error='Whitespace in encoded inspect command; replaced by standalone script before training',
        trained_sources_changed=False),ensure_ascii=False,indent=2),encoding='utf-8')
    print('FINAL_REPORT',STUDY/'FINAL.md')
    print('SELECTED',assembled['selected_scores'])
    print('ACTIVE_TRAINERS',active_jobs())

if __name__=='__main__':main()
