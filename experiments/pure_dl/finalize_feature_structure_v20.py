"""Finalize completed v20 evidence without changing trained code or weights."""
from pathlib import Path
from datetime import datetime
import hashlib,json,subprocess,sys
from launch_latent_gate_v17 import active_jobs,append_once
from report_feature_structure_v20 import ROOT,STUDY,MODES

def load(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if active_jobs():raise RuntimeError('Trainers still active')
    subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).with_name('report_feature_structure_v20.py')),'--record'],check=True,stdout=subprocess.DEVNULL)
    p=load(STUDY/'progress.json');a=load(STUDY/'assembled_report.json');features=load(STUDY/'feature_manifest.json')
    assert p['all_complete'] and a['all_new_POR_reloaded']
    rows={r['mode']:r for r in p['rows']};candidates={c['mode']:c for c in a['candidates']}
    starts=[];ends=[];source_verified=True
    for mode,r in rows.items():
        out=ROOT/'outputs'/r['id'];launch=load(out/'launch.json');cfg=load(out/'config.json')
        starts.append(datetime.fromisoformat(launch['started_at']));ends.append(datetime.fromisoformat(r['final']['finished_at']))
        for name,sha in cfg['source_hashes'].items():
            actual=hashlib.sha256((ROOT/'experiments/pure_dl'/name).read_bytes()).hexdigest()
            if actual!=sha:raise RuntimeError('Trained source changed: '+name)
    timing=dict(max_concurrent_jobs=2,execution='first two concurrent; third explicitly started after slots available',
        first_launch_to_last_finish_seconds=(max(ends)-min(starts)).total_seconds(),
        run_seconds={m:r['final']['seconds'] for m,r in rows.items()},peak_allocated_MiB={m:r['final']['peak_allocated_MiB'] for m,r in rows.items()},
        feature_preparation_seconds=features['seconds'],final_replay_setup_seconds=a['seconds'],
        scope='Wall includes launches/import/data loading/train/validation/restore and gap before third explicit launch; excludes implementation, feature preparation, final replay. No concurrency speedup claim.')
    (STUDY/'timing.json').write_text(json.dumps(timing,ensure_ascii=False,indent=2),encoding='utf-8')
    f=a['comparisons']['feature_shape_minus_amplitude'];s=a['comparisons']['fusion_late_minus_early']
    lines=['# v20：特征表示与双视图融合的实测结果','',f"完成记录：{datetime.now().astimezone().isoformat()}",'',
        '## 目标与已完成','',
        '本批三项均完成。从输入而非扩大网络容量切入，原206维保持，分别检验63维侧输入的幅值/形态表示以及融合位置。所有候选只训练POR，PERM/SW各自私有且保持不变。',
        f"三组参数量相同：{p['rows'][0]['parameter_count']}。完整初始state一致={p['same_initial_state']}，初始预测一致={p['same_initial_predictions']}。配置差异={p['config_differences']}。",'',
        '|方案|POR|实际三通道TOTAL|相对旧v5|最佳轮次|完成轮次|','|---|---:|---:|---:|---:|---:|']
    ref=a['reference_scores'];lines.append(f"|保留参考v5/v8|{ref['POR']:.9f}|{ref['TOTAL']:.9f}|0|历史|历史|")
    for mode in MODES:
        c=candidates[mode]
        lines.append(f"|{mode}|{c['scores']['POR']:.9f}|{c['scores']['TOTAL']:.9f}|{c['delta_v5_TOTAL']:+.9f}|{c['best_epoch']}|{c['stopped_epoch']}|")
    lines+=['',f"采纳新POR={a['selected_new_POR']}；本批最终保留TOTAL={a['selected_scores']['TOTAL']:.12f}。达到80轮属于预算结束，不能写成patience早停。",'',
        '## 预定义的两个对照','',
        f"特征：shape_early减amplitude_early，POR{f['POR_delta']:+.9f}，TOTAL{f['TOTAL_delta']:+.9f}；{f['well_wins']}/16井提高、{f['well_losses']}井下降。默认段POR差{f['default_POR_delta']:+.9f}，非默认段差{f['numeric_POR_delta']:+.9f}。",
        f"结构：shape_late减shape_early，POR{s['POR_delta']:+.9f}，TOTAL{s['TOTAL_delta']:+.9f}；{s['well_wins']}/16井提高、{s['well_losses']}井下降。默认段POR差{s['default_POR_delta']:+.9f}，非默认段差{s['numeric_POR_delta']:+.9f}。",
        f"选择未校正的整井重采样TOTAL区间：特征{f['exploratory_selection_unadjusted_TOTAL_95']}；结构{s['exploratory_selection_unadjusted_TOTAL_95']}。当前只有一个固定种子和一个反复参与开发的第五折，不作独立泛化结论。",
        '特征对照改变整组侧视图，包含残差标准化、趋势差和归一化变化量，尚未单独分离每种派生量的贡献；不能全部归因于除以标准差。结构对照仅改变两视图的混合时机，特征、参数量和初始化相同。', '',
        '## 特征和网络定义','',
        '每条允许曲线得到7个侧特征，总63维。GR/RXO/RT为log1p值；窗口9/51/201各井独立计算。形态残差r=(x-mean_w)/sqrt(std_w^2+(0.05*IQR_train)^2)，加两种均值差和两种跨点变化量。所有全局统计只由原64训练井拟合，其他井只用自身输入。原206幅值和井内特征并未删除。',
        '两路投影到80维。早融合先平均两路投影，分别送入两个独立编码器；晚融合分别处理两路，再使用逐点逐通道sigmoid门混合。最终两层局部卷积和8分量Student-t3输出完全相同。',
        '学习与评价使用原POR训练侧变换、CDF边缘和256格/800动作/单覆盖解码；Q=1，80/18/min10、bs64×64steps、相同优化器/调度器。与旧v5还存在预训练历史、视图和容量差，不能当纯特征消融。', '',
        '## 训练—验证差','',
        '|模式|训练POR|验证POR|训练非默认段|验证非默认段|','|---|---:|---:|---:|---:|']
    for mode in MODES:
        r=rows[mode]['final']
        lines.append(f"|{mode}|{r['fit']['POR']:.6f}|{r['validation']['POR']:.6f}|{r['fit']['numeric_POR']:.6f}|{r['validation']['numeric_POR']:.6f}|")
    lines+=['','## 验证范围','',
        '三份新POR从磁盘重新构建，全部144665点预测逐元素复现；固定PERM/SW直接使用此前已验证预测数组，本轮没有重复声称旧模型GPU重载。最终总分以三个实际列调用原metric，不用日志最大值拼凑。',
        '训练后小批次填充隔离、全空有限输出、Student-t NLL对照通过。侧输入扰动只用于训练输入上的依赖诊断，晚融合的主编码器不受侧输入扰动；不将这种推理干预当额外收益。所有在训练配置中记录的源码指纹已核对保持不变。', '',
        '## 当前选择、停止边界与后续未做','',
        '本批三项到此结束，不追加种子/宽度/窗口或学习率网格，不补原其他四折，不覆盖旧交付。特征与结构的收益分别报告，不把没有超过旧方案的结果写成新的总分冠军。',
        '后续应把有匹配正收益的输入机制在更强的既有私有骨干上单独接入，与原特征通路做函数保持的控制，检验能否缩小从头训练方案与当前冠军之间的差距。该接入与后续训练本批尚未实现；不预告一定有效。',
        '当前暂缓补折不变；原第五折严格>80.5、完整原五折OOF>82未完成。','',
        '## 来源、时间和维护','',
        '代码：prepare_feature_structure_v20.py、feature_structure_v20.py、train_feature_structure_v20.py。feature_manifest.json记录63维名称、所有训练侧常数和输入诊断；METHOD.md记录预定义比较。',
        '原始证据：每个outputs目录的config.json、fold_5_report.json、validation_history.json、model_best.pt、last_state.pt。实际重载/组合：assembled_report.json，保留配方：CURRENT_RECIPE.json。',
        f"从首次显式启动到最后结束墙钟{timing['first_launch_to_last_finish_seconds']:.3f}秒；各项{timing['run_seconds']}。特征准备{features['seconds']:.3f}秒、最终重载{a['seconds']:.3f}秒单独记录，不相加冒称并发墙钟。",
        '本批图谱节点通过report_feature_structure_v20.py --record可复现维护，build_atlas.py更新静态图谱，未改历史curate.py或训练源文件。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v20特征与融合实验完成'
    entry=marker+'\n'+datetime.now().astimezone().isoformat()+f"。三项全部完成并重载，特征TOTAL差{f['TOTAL_delta']:+.9f}，晚融合TOTAL差{s['TOTAL_delta']:+.9f}；保留TOTAL={a['selected_scores']['TOTAL']:.9f}。仅本批新POR，其他目标/折/交付不变；强骨干特征接入与完整OOF尚未做。证据feature_structure_v20_20260928/FINAL.md。"
    for path in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):
        append_once(path,marker,entry)
    for mode,r in rows.items():
        out=ROOT/'outputs'/r['id'];c=candidates[mode]
        note=marker+f"\n实际完成{c['stopped_epoch']}轮，最佳{c['best_epoch']}；POR={c['scores']['POR']:.9f}，实际组合TOTAL={c['scores']['TOTAL']:.9f}。新POR磁盘重载全部点复现；其他目标复用固定预测。收益解释和未做项见研究FINAL.md。"
        for name in ('README.md','PROJECTS.md'):append_once(out/name,marker,note)
    (STUDY/'integrity_summary.json').write_text(json.dumps(dict(training_failures=[r['failure'] for r in p['rows'] if r['failure']],
        trained_sources_unchanged=source_verified,all_completed=True,new_POR_reloads_exact=True,fixed_other_arrays_reused=True),ensure_ascii=False,indent=2),encoding='utf-8')
    print('FINAL_REPORT',STUDY/'FINAL.md');print('SELECTED',a['selected_scores'])
    print('CONTRASTS',json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='per_well'} for k,v in a['comparisons'].items()},ensure_ascii=False))
    print('ACTIVE_TRAINERS',active_jobs())

if __name__=='__main__':main()
