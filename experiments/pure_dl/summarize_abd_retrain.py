"""Compare completed matched training arms using fused-selected checkpoints."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/deep_learning/abd_retrain_attribution'


def main():
    definitions=[('control','原结构','#1565c0'),('no_global_full','完整视图去全局','#e07826')]
    rows=[];curves=[]
    for mode,title,color in definitions:
        folder=ROOT/f'outputs/pure_dl_abd_scratch_{mode}80_fold5'
        report=json.loads((folder/'fold_5_report.json').read_text(encoding='utf-8'))
        history=json.loads((folder/'fold_5_validation_history.json').read_text(encoding='utf-8'))
        assert report['best_checkpoint_predictions_reproduced'] and report['all_validation_rows_covered_exactly_four_times']
        best=max(history,key=lambda r:r['official_total'])
        assert best['epoch']==report['best_epoch'] and best['official_total']==report['scores']['TOTAL']
        rows.append({'arm':mode,'title':title,'scores':report['scores'],'solo_scores':report['solo_scores'],
                     'best_epoch':report['best_epoch'],'stopped_epoch':report['stopped_epoch'],
                     'delta_adopted_baseline':report['scores']['TOTAL']-80.40915186344968,
                     'seconds':report['seconds'],'peak_gpu_mib':report['peak_gpu_allocated_mib']})
        curves.append((title,color,history,best))
    OUT.mkdir(parents=True,exist_ok=True)
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False})
    fig,axes=plt.subplots(2,1,figsize=(10.5,7.5),sharex=True,layout='constrained')
    for title,color,history,best in curves:
        epochs=[r['epoch'] for r in history]
        for ax,key in [(axes[0],'solo_scores'),(axes[1],'official_scores')]:
            values=[r[key]['TOTAL'] for r in history]
            ax.plot(epochs,values,label=title,color=color,lw=2)
            ax.scatter([best['epoch']],[best[key]['TOTAL']],color=color,s=60,zorder=4)
            ax.grid(alpha=.17)
    axes[0].set_title('A 单模型：随训练继续改善，不等于融合也改善',fontproperties=font)
    axes[1].set_title('A+B+D 融合：圆点是实际保留的最佳 checkpoint',fontproperties=font)
    axes[1].axhline(80.40915186344968,color='#666666',ls='--',lw=1,label='原 ABD 基线 80.40915')
    for ax in axes:
        ax.set_ylabel('第五折 TOTAL',fontproperties=font)
        ax.legend(prop=font,frameon=False)
    axes[1].set_xlabel('训练轮次（最大预算80，patience=18）',fontproperties=font)
    fig.suptitle('相同初始化与训练规则，只改完整视图的全局注意力',fontproperties=font,fontsize=15)
    fig.savefig(OUT/'checkpoint_selection.png',dpi=170)
    plt.close(fig)
    delta=rows[1]['scores']['TOTAL']-rows[0]['scores']['TOTAL']
    summary={'rows':rows,'no_global_minus_control':delta,'scope':'matched fixed-fifth training, fused checkpoint selection; not full OOF'}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    text=['# 双骨干内部全局注意力：匹配训练结果','','![单模型与融合的训练曲线](checkpoint_selection.png)','',
          'B/D完全固定，A从相同随机初始化训练。保留原成员NLL、内部router损失、辅助监督及动态Q；只比较完整视图的全局注意力有无。每轮按ABD融合官方TOTAL选模型。','',
          '|结构|融合最佳|相对ABD基线|该轮A单独|最佳/结束轮|','|---|---:|---:|---:|---:|']
    for r in rows:text.append(f"|{r['title']}|{r['scores']['TOTAL']:.8f}|{r['delta_adopted_baseline']:+.8f}|{r['solo_scores']['TOTAL']:.8f}|{r['best_epoch']}/{r['stopped_epoch']}|")
    text+=['',f'去全局减原结构：**{delta:+.8f}**。这是本次训练规则和种子下的匹配比较，不是所有全局注意力的普遍结论。',
           '先前固定权重的推理删除约+0.02593，与本次从头训练是不同实验，不能混用。若训练结果不支持删除，则应保留原结构，把主要收益归到融合选模上。',
           '两组均直接恢复该折融合最佳checkpoint，没有按最佳轮次重新训练。冻结B/D的来源、第五折井集合及解码保持一致。分数来自反复用于开发的第五折；完整OOF仍待其余四折确认。',
           '', '计时为各折预处理、训练、探测、验证和恢复；两组部分错峰双并发，不把单项时间相加称为总墙钟。显存与秒数见summary.json。']
    (OUT/'README.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
