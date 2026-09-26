"""List every completed neural experiment without suppressing negative results."""
from pathlib import Path
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
runs=[('CNN baseline','simple_cnn_cv'),('CNN + outlier treatment','simple_cnn_anomaly_cv'),('CNN + MLP','cnn_mlp_cv'),('State distribution NN','pure_dl_v1'),
      ('Rich-input NN','pure_dl_rich'),('MDN 128','pure_dl_residual'),('MDN analytic','pure_dl_residual/analytic_800'),
      ('Metric-loss NN','pure_dl_metric'),('Well-context NN','pure_dl_context'),('Equal 4 neural models','pure_dl_equal4'),
      ('Equal 5 neural models','pure_dl_equal5'),('POR specialist replacement','pure_dl_por_specialist'),
      ('PERM specialist replacement','pure_dl_perm_specialist'),('SW specialist replacement','pure_dl_sw_specialist'),
      ('Three fixed specialists','pure_dl_three_specialists'),('Equal 6 incl. specialists','pure_dl_equal6_specialists'),
      ('PLE neural model','pure_dl_ple'),('Equal 7 incl. PLE','pure_dl_equal7'),
      ('Equal 11 neural sources','pure_dl_equal11'),
      ('Shared-weight neural ensemble','pure_dl_batchensemble'),('Invariant-input neural model','pure_dl_invariant'),
      ('Inner-well PLE calibration','pure_dl_ple_calibrated'),('Cross-well neural attention','pure_dl_retrieval'),
      ('Density-physics POR replacement','pure_dl_density_physics'),
      ('Invariant shared-weight ensemble','pure_dl_batchensemble_invariant'),
      ('Inner-well decision calibration','pure_dl_decision_calibrated'),('Utility fine-tuning','pure_dl_utility_finetune'),
      ('Equal expanded neural models','pure_dl_equal_expanded'),
      ('Rich invariant shared ensemble','pure_dl_batchensemble_rich_invariant'),
      ('Curve-token Transformer','pure_dl_curve_transformer'),('60-epoch shared ensemble','pure_dl_batchensemble_nll60'),
      ('Learned periodic numerical encoding','pure_dl_batchensemble_periodic'),
      ('Equal extended neural models','pure_dl_equal_extended'),
      ('Student-t shared ensemble','pure_dl_batchensemble_student'),
      ('Equal extended plus Student-t','pure_dl_equal_student'),
      ('Target-private nonlinear heads','pure_dl_batchensemble_targetheads'),
      ('Equal plus target-private heads','pure_dl_equal_targetheads'),
      ('Explicit feature-cross network','pure_dl_batchensemble_cross'),
      ('Equal plus feature-cross network','pure_dl_equal_cross'),
      ('Integrated score training','pure_dl_batchensemble_integrated'),
      ('Equal plus integrated score training','pure_dl_equal_integrated'),
      ('Depth-patch Transformer','pure_dl_depth_patch'),
      ('Equal plus depth-patch Transformer','pure_dl_equal_depth_patch'),
      ('CRPS likelihood neural model','pure_dl_batchensemble_crps'),
      ('Equal plus CRPS neural model','pure_dl_equal_crps'),
      ('TabM-inspired32-member network','pure_dl_tabm32'),
      ('Equal plus TabM-inspired32','pure_dl_equal_tabm32'),
      ('Wider32-member network','pure_dl_tabm32_wide'),
      ('Equal plus wider32-member','pure_dl_equal_tabm32_wide'),
      ('32-member full per-member batch','pure_dl_tabm32_fullbatch'),
      ('Equal plus full-batch32','pure_dl_equal_tabm32_fullbatch'),
      ('Learned cross-well density','pure_dl_neighborhood_density'),
      ('Equal plus learned cross-well density','pure_dl_equal_neighborhood_density'),
      ('PLE cross-well density','pure_dl_neighborhood_ple'),
      ('Equal plus PLE cross-well density','pure_dl_equal_neighborhood_ple'),
      ('Corrected per-feature TabM32','pure_dl_tabm32_aligned'),
      ('Equal plus corrected per-feature TabM32','pure_dl_equal_tabm32_aligned'),
      ('Dense multiscale depth network','pure_dl_multiscale_sequence'),
      ('Dense multiscale plus auxiliary losses','pure_dl_multiscale_sequence_deep'),
      ('Dense sequence direct official utility','pure_dl_action_sequence'),
      ('Gated conditional official utility','pure_dl_hurdle_utility_sequence'),
      ('Multiscale density with bottleneck attention','pure_dl_attention_multiscale_sequence'),
      ('TimeMixer-inspired depth mixing','pure_dl_timemixer_sequence'),
      ('Dense sequence NLL plus CRPS','pure_dl_crps_sequence'),
      ('Target-private dense sequence heads','pure_dl_private_head_sequence'),
      ('Dense sequence dropout0.25','pure_dl_regularized_sequence'),
      ('Target-private sequence with SAM','pure_dl_sam_sequence'),
      ('Adaptive shallow receptive fields','pure_dl_adaptive_scale_sequence')]
out=ROOT/'outputs/pure_dl_summary';out.mkdir(parents=True,exist_ok=True);rows=[]
for name,rel in runs:
    p=ROOT/'outputs'/rel/'report.json'
    if not p.exists():continue
    r=json.loads(p.read_text(encoding='utf-8'));s=r['oof_scores']
    verification=p.parent/'verification.json'
    verified=verification.exists() and json.loads(verification.read_text(encoding='utf-8')).get('passed') is True
    rows.append({'name':name,'directory':rel,**s,'seconds':r.get('seconds',r.get('elapsed_seconds')),
                 'verification':verified})
with (out/'scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fig,ax=plt.subplots(figsize=(11,max(5,.45*len(rows))))
ax.scatter([r['TOTAL'] for r in rows],range(len(rows)),s=40,color=['#2463a6' if r['verification'] else '#999999' for r in rows],zorder=3)
ax.set_yticks(range(len(rows)),[r['name'] for r in rows])
for i,r in enumerate(rows):ax.text(r['TOTAL']+.06,i,f"{r['TOTAL']:.3f}",va='center')
ax.axvline(82,color='#bd3737',ls='--',label='Goal: 82');ax.set_xlim(min(r['TOTAL'] for r in rows)-.5,max(83,max(r['TOTAL'] for r in rows)+.7));ax.invert_yaxis();ax.set_xlabel('Official 5-fold OOF score / 100');ax.grid(axis='x',alpha=.2);ax.legend();ax.set_title('All completed neural experiments | same 80 wells and 723,556 labels')
fig.tight_layout();fig.savefig(out/'scores.png',dpi=180);plt.close(fig)
lines=['# 纯深度学习实验记录','', '所有分数均为同一80井、723556个完整有效标签点的按井五折OOF。测试数据未用于训练或评分。',
       '全部已完成实验均保留，包括负结果。重复使用同一CV进行方法开发可能产生选择偏差；这不是新的盲测成绩。',
       '专用替换模型只替换预先指定的一整列目标，其余两列来自事先固定的同折MDN预测，不按真实标签逐点择优。',
       '', '| 模型 | POR | PERM | SW | 总分 | 耗时/秒 | 评分核验 |','|---|---:|---:|---:|---:|---:|---|']
for r in rows:
    duration=f"{r['seconds']:.1f}" if isinstance(r['seconds'],(int,float)) else '未记录'
    lines.append(f"| {r['name']} | {r['POR']:.3f} | {r['PERM']:.3f} | {r['SW']:.3f} | {r['TOTAL']:.3f} | {duration} | {'通过' if r['verification'] else '待核验'} |")
best=max((r for r in rows if r['verification']),key=lambda r:r['TOTAL']);lines+=['',f"历史最高已核验分数 {best['TOTAL']:.6f}，{'达到' if best['TOTAL']>=82 else '未达到'}82目标。",'',f"历史最佳报告：[{best['directory']}/report.json](../{best['directory']}/report.json)。",'','耗时为各自运行记录；并发会影响耗时，不能将表中时间简单相加视为总墙钟时间。校准、解析解码、续训和微调条目均不包含来源网络此前训练时间。','', '![全实验结果](scores.png)']
current_names = {rel for _, rel in runs[runs.index(('Dense multiscale depth network','pure_dl_multiscale_sequence')):]}
current = [r for r in rows if r['directory'] in current_names]
current_verified = [r for r in current if r['verification']]
current_best = max(current_verified, key=lambda r: r['TOTAL']) if current_verified else None
snapshot = {'scope': 'Current standalone sequence architecture and loss experiments; historical fusion excluded',
            'completed': current, 'best_official_score_verified': current_best,
            'without_complete_report': sorted(current_names-{r['directory'] for r in current}),
            'caution': 'Missing report is not proof of a live process; CV reused for development, not a blind test'}
(out/'current_sequence_results.json').write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
if current_best:
    lines += ['', f"当前序列结构/损失方向最高已核验分数 {current_best['TOTAL']:.6f}（{current_best['directory']}）。历史融合不作为当前方向的结果。"]
(out/'README.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(rows,indent=2));print('BEST',best['name'],best['TOTAL'])
