"""CPU-only live curve snapshot comparing the two fixed-companion B routes."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/deep_learning/abd_b_training_routes'
NAMES=['pure_dl_abd_b_scratch80_fold5','pure_dl_abd_b_fused_scratch80_fold5']
LABELS=['仅验证融合（对照）','训练、验证均融合']
COLORS=['#527991','#bf603d']


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    histories=[json.loads((ROOT/'outputs'/name/'fold_5_validation_history.json').read_text()) for name in NAMES]
    baseline=json.loads((ROOT/'outputs/pure_dl_abd_scratch_control80_fold5/fold_5_report.json').read_text())['scores']
    report={'scope':'fixed fifth, development curves; partial histories are not final results',
        'baseline':baseline,'routes':[]}
    for name,h in zip(NAMES,histories):
        best=max(h,key=lambda r:r['official_total'])
        report['routes'].append({'name':name,'last_epoch':h[-1]['epoch'],'best_epoch':best['epoch'],
            'best_scores':best['official_scores'],'delta_baseline':best['official_total']-baseline['TOTAL'],
            'history':h})
    # Compare same epoch counts, not just each route's current best.
    old={r['epoch']:r for r in histories[0]};new={r['epoch']:r for r in histories[1]}
    common=sorted(old.keys()&new.keys())
    report['matched_epochs']=[{'epoch':e,'old_fused':old[e]['official_total'],'new_fused':new[e]['official_total'],
        'delta':new[e]['official_total']-old[e]['official_total']} for e in common]
    (OUT/'snapshot.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
        'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for ax,channel in zip(axes.flat,['TOTAL','POR','PERM','SW']):
        for h,label,color in zip(histories,LABELS,COLORS):
            xx=[r['epoch'] for r in h]; yy=[r['official_scores'][channel] for r in h]
            ax.plot(xx,yy,label=f'{label}（到第{xx[-1]}轮）',color=color,lw=1.6)
            best=int(np.argmax(yy));ax.scatter([xx[best]],[yy[best]],s=32,color=color,zorder=4)
        ax.axhline(baseline[channel],color='#5b635d',ls='--',lw=1,label='固定基线')
        ax.set_title(channel,loc='left',fontweight='bold');ax.set_xlabel('训练轮次')
        ax.set_ylabel('第五折融合分数');ax.grid(axis='y',alpha=.18)
    axes.flat[0].legend(loc='lower right',fontsize=8)
    fig.suptitle('固定 A、D：B 的训练目标对比｜运行中快照',fontweight='bold',fontsize=15)
    fig.savefig(OUT/'fused_channel_curves.png',dpi=160)
    plt.close(fig)
    text=['# 固定 A/D 的 B 训练路线对比','',
        '这是正在训练的第五折开发曲线快照，尚非最终报告。蓝线只在验证时融合，橙线训练主损失也来自融合。',
        '两路 B 同 seed 随机初始化；都固定 A6/D52，保留原辅助监督和动态 Q 算法。',
        '损失目标变化后参数轨迹及后续 Q 会变化，不能称为逐点权重完全相同的消融。',
        '新路线还未训练到旧路线的轮次，不可仅以当前最高分差异断言最终优劣。','',
        '![融合分通道曲线](fused_channel_curves.png)','',
        '|路线|最新轮次|当前最佳轮次|当前最佳TOTAL|相对80.492309|','|---|---:|---:|---:|---:|']
    for label,r in zip(LABELS,report['routes']):
        text.append(f"|{label}|{r['last_epoch']}|{r['best_epoch']}|{r['best_scores']['TOTAL']:.6f}|{r['delta_baseline']:+.6f}|")
    if common:
        e=common[-1];text+=['',f"相同第{e}轮：原路线{old[e]['official_total']:.6f}，融合训练{new[e]['official_total']:.6f}，差{new[e]['official_total']-old[e]['official_total']:+.6f}。仍属于早期快照。"]
    (OUT/'README.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='routes'},ensure_ascii=False))


if __name__=='__main__':main()
