"""Finalize only completed v19 evidence and update project/graph summaries."""
from pathlib import Path
from datetime import datetime
import json,subprocess,sys
from report_por_pyramid_v19 import ROOT,STUDY
from launch_latent_gate_v17 import active_jobs,append_once


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if active_jobs():raise RuntimeError('Do not finalize while trainers run')
    subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).with_name('report_por_pyramid_v19.py')),'--record'],check=True,stdout=subprocess.DEVNULL)
    p=read(STUDY/'progress.json');a=read(STUDY/'assembled_report.json')
    assert p['all_complete'] and a['all_new_POR_reloaded']
    rows={r['mode']:r for r in p['rows']};candidates={c['mode']:c for c in a['candidates']};comparison=a['comparison']
    launches=[read(ROOT/'outputs'/r['id']/'launch.json') for r in p['rows']]
    start=min(datetime.fromisoformat(r['started_at']) for r in launches)
    end=max(datetime.fromisoformat(r['final']['finished_at']) for r in p['rows'])
    timing=dict(explicit_parallel_jobs=2,launch_to_both_finish_seconds=(end-start).total_seconds(),
        per_run_seconds={m:r['final']['seconds'] for m,r in rows.items()},
        peak_allocated_MiB={m:r['final']['peak_allocated_MiB'] for m,r in rows.items()},
        final_replay_setup_seconds=a['seconds'],
        scope='Training wall includes process imports/setup/train/validation/best restoration. Excludes implementation and subsequent POR disk replay; not a concurrency speedup comparison.')
    (STUDY/'timing.json').write_text(json.dumps(timing,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# v19 POR多尺度骨干：最终结果','',f"完成记录：{datetime.now().astimezone().isoformat()}",'',
        '## 探索重点与匹配范围','',
        '使用256/128/64三分辨率编码、双高分辨率跳接、96维局部卷积和粗尺度门控记忆辅助。regional/global两组仅改变记忆读/潜空间处理/写回的连通掩码，所有参数形状与完整初始状态一致。',
        f"实际参数量：{p['rows'][0]['parameter_count']}。相同初始化={p['identical_initial_state']}；相同初始全部预测={p['initial_predictions_identical']}；配置差异={p['config_differences']}。",
        '两者都是从随机网络训练的POR私有模型；只复用同64训练井的输入预处理和目标变换/解码元数据。保持原Student-t3八分量头、BCE/NLL、Q=1、80/18/bs64×64steps和单覆盖原评分。与v18不同的不只有记忆机制，还有局部结构和容量；与旧v5还存在视图和训练历史差异。','',
        '## 实测原第五折总分','',
        '|方案|POR|PERM|SW|TOTAL|相对v5|最佳轮次|完成轮次|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    ref=a['reference_scores']
    lines.append(f"|保留参考v5/v8|{ref['POR']:.9f}|{ref['PERM']:.9f}|{ref['SW']:.9f}|{ref['TOTAL']:.9f}|0|历史|历史|")
    for m in ('regional','global'):
        c=candidates[m];s=c['scores']
        lines.append(f"|{m}|{s['POR']:.9f}|{s['PERM']:.9f}|{s['SW']:.9f}|{s['TOTAL']:.9f}|{c['delta_v5_TOTAL']:+.9f}|{c['best_epoch']}|{c['stopped_epoch']}|")
    lines += ['',f"当前选择新POR={a['selected_new_POR']}；保留TOTAL={a['selected_scores']['TOTAL']:.12f}。第80轮结束表示预算耗尽，不能写成patience早停。原第五折参与选模，不是独立盲测。",'',
        '## 控制实验与错误分解','',
        f"global减regional：POR={comparison['global_minus_regional_POR']:+.9f}，TOTAL={comparison['global_minus_regional_TOTAL']:+.9f}。{comparison['well_wins']}/16井上升，{comparison['well_losses']}井下降。",
        f"真实默认段POR差={comparison['default_POR_delta']:+.9f}，非默认段差={comparison['numeric_POR_delta']:+.9f}；对TOTAL分别贡献{comparison['default_TOTAL_contribution']:+.9f}与{comparison['numeric_TOTAL_contribution']:+.9f}。",
        f"整井重采样的探索性TOTAL区间={comparison['exploratory_selection_unadjusted_TOTAL_95']}，未校正当前及历史选模，不作为独立确认。",'',
        '|模型|训练POR|验证POR|训练非默认POR|验证非默认POR|','|---|---:|---:|---:|---:|']
    for m in ('regional','global'):
        r=rows[m]['final']
        lines.append(f"|{m}|{r['fit']['POR']:.6f}|{r['validation']['POR']:.6f}|{r['fit']['numeric_POR']:.6f}|{r['validation']['numeric_POR']:.6f}|")
    lines+=['','## 已验证事项','',
        '两个新POR从各自model_best.pt在磁盘上独立重建，原144665点预测逐元素一致。PERM/SW复用已有经验证的预测数组，行序和数值不变；本轮未重复GPU重载两个旧模型。总分实际调用原metric计算。',
        '训练后填充隔离、全空输入、Student-t NLL与Torch独立定义对照通过；原CDF函数复用。两训练窗口检查记忆门值、连通边数和绕过辅助分支的输出差异，仅描述已训练模型的依赖性，不作为额外评分候选或重训效果。',
        '网络门sigmoid控制消息比例但不限制消息绝对幅度。regional多层卷积和区域潜变量传播仍有广上下文，因此两组比较是连通范围差异，不是有无任何全局信息。',
        '', '## 不继续的边界及下一步未执行事项','',
        '本批只运行两组，无第三组、旧队列、原其余折或交付覆盖。不会把未超旧模型的结果称SOTA更新，也不会将某一次负差推广成金字塔/注意力整体无效。',
        '下一步应先区分本批相对v18的容量和拓扑影响，再决定是否加入匹配的无记忆重训对照或跨井训练正则；目前均未启动。避免同时改损失、采样与网络而失去可解释性。',
        '完整原五折OOF>82仍未完成；严格第五折80.5及暂缓补折规则不变。当前配方见CURRENT_RECIPE.json。','',
        '## 时间、来源与维护','',
        f"两项并发从显式启动至均结束墙钟{timing['launch_to_both_finish_seconds']:.3f}秒；单项{timing['per_run_seconds']}；另有最终POR磁盘重载与数据准备{a['seconds']:.3f}秒。各项秒数不能相加解释并发墙钟。",
        '配置、曲线、最佳/当前检查点在两个outputs目录；progress.json与assembled_report.json保存统一结果。trainer_adaptation.json记录复用的冻结v18训练循环与精确替换。历史训练源文件未改。',
        '图谱人工节点由report_por_pyramid_v19.py --record维护。对curate.py的编辑被工具内容保护拒绝，原文件未改，未绕过保护；当前curation.json与静态图谱独立更新。',
        '工具读取curate.py时首次max_bytes不足，扩大到真实文件大小后读出；另一次多行精确编辑未匹配，单行编辑命中后被内容保护拒绝。这些不是模型训练异常。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v19多尺度POR对照已完成'
    entry=marker+'\n'+datetime.now().astimezone().isoformat()+f"。regional TOTAL={candidates['regional']['scores']['TOTAL']:.9f}，global TOTAL={candidates['global']['scores']['TOTAL']:.9f}，当前保留{a['selected_scores']['TOTAL']:.9f}。新POR磁盘复现及固定其余两通道数组合并完成。尚未运行新对照/其他目标/原其他折，完整OOF>82未完成。记录por_pyramid_v19_20260928/FINAL.md。"
    for pth in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):
        append_once(pth,marker,entry)
    for row in p['rows']:
        out=ROOT/'outputs'/row['id'];c=candidates[row['mode']]
        finish=marker+f"\n最佳{c['best_epoch']}轮、实际完成{c['stopped_epoch']}轮；POR={c['scores']['POR']:.9f}，实际TOTAL={c['scores']['TOTAL']:.9f}。最终磁盘恢复通过，无其他候选自动启动。相对基线和未完成项详见研究FINAL.md。"
        for name in ('README.md','PROJECTS.md'):append_once(out/name,marker,finish)
    failures=[r['failure'] for r in p['rows'] if r['failure']]
    attempts=dict(training_failures=failures,curator_read_limit_retry=True,curator_edit_blocked_by_content_protection=True,
        original_curator_unchanged=True,training_sources_modified_after_start=False)
    (STUDY/'tool_attempts.json').write_text(json.dumps(attempts,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FINAL_REPORT',STUDY/'FINAL.md');print('SELECTED',a['selected_scores']);print('ACTIVE_TRAINERS',active_jobs())

if __name__=='__main__':main()
