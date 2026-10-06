"""Render measured attribution results; never substitute an interim score."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/pure_dl_8040_pair_ablation'
font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.facecolor':'#f5f7fb','axes.facecolor':'#ffffff'})
summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
internal=json.loads((ROOT/'outputs/pure_dl_internal_expert_attribution/report.json').read_text(encoding='utf-8'))
arms={v['arm']:v['scores']['TOTAL'] for v in internal['arms']}
fig,axes=plt.subplots(1,3,figsize=(18,6.6),gridspec_kw={'width_ratios':[1.15,1,1]})
fig.subplots_adjust(left=.06,right=.98,top=.79,bottom=.27,wspace=.36)
fig.text(.06,.94,'80.40 基线的收益归因',fontsize=22,fontweight='bold',color='#142840')
fig.text(.06,.875,'固定第五折 · 模型删减和内部专家为冻结推理诊断 · 训练消融另用原双专家作为对照',fontsize=11,color='#506176')
ax=axes[0];colors={2:'#7087a6',3:'#00887e',4:'#db8c22'}
for v in summary['subsets']:
    x=v['parameters']/1e6;y=v['scores']['TOTAL'];n=len(v['sources'])
    ax.scatter(x,y,s=65,color=colors[n],zorder=3)
    ax.annotate(v['label'].replace('+',''),(x,y),xytext=(5,4),textcoords='offset points',fontsize=9)
ax.axhline(summary['baseline']['TOTAL'],color=colors[4],ls='--',lw=1,alpha=.65)
ax.set(xlabel='全部来源参数量（百万）',ylabel='第五折 TOTAL',xlim=(1.85,5.25),ylim=(80.205,80.445))
ax.set_title('① 删减模型：分数与成本',loc='left',pad=15,fontweight='bold')
ax.grid(axis='y',alpha=.15)
for n in [2,3,4]:ax.scatter([],[],color=colors[n],label=f'{n}个来源')
ax.legend(frameon=False,loc='lower right',fontsize=8)
ax=axes[1];labels=['full206','校准68','固定等权','原路由'];keys=['full206','calibrated68','fixed_equal','routed']
for tag,col in [('A','#205c9b'),('B','#00887e')]:
    values=[arms['four_fine__'+tag+'_'+k] for k in keys]
    ax.plot(range(4),values,'o-',color=col,label=tag,lw=1.8)
    for i,y in enumerate(values):ax.annotate(f'{y:.3f}',(i,y),xytext=(0,9 if tag=='A' else -15),textcoords='offset points',ha='center',fontsize=8,color=col)
ax.set_xticks(range(4),labels,rotation=15)
ax.set(ylabel='第五折 TOTAL',ylim=(78.98,80.5))
ax.set_title('② 内部双专家的互补',loc='left',pad=15,fontweight='bold')
ax.grid(axis='y',alpha=.15);ax.legend(frameon=False,loc='upper left')
ax=axes[2];training=[('原双专家对照',summary['training_control']['TOTAL'])]
training += [(v['title'],v['scores']['TOTAL']) for v in summary['training_ablations']]
for i,(label,score) in enumerate(training):
    delta=score-summary['training_control']['TOTAL']
    ax.plot([0,delta],[i,i],color='#c2cbd8',lw=3)
    ax.scatter([delta],[i],color='#205c9b' if i==0 else '#aa5c48',s=65,zorder=3)
    ax.annotate(f'{score:.4f}  ({delta:+.4f})',(delta,i),xytext=(5,9),textcoords='offset points',fontsize=9)
ax.set_yticks(range(len(training)),[v[0] for v in training]);ax.invert_yaxis()
deltas=[score-summary['training_control']['TOTAL'] for _,score in training]
ax.axvline(0,color='#9cabbd',lw=1);ax.set(xlabel='相对原双专家的分数差',xlim=(min(-.26,min(deltas)-.08),max(.15,max(deltas)+.15)))
ax.set_title('③ 独立重训的机制消融',loc='left',pad=15,fontweight='bold')
ax.grid(axis='x',alpha=.15)
fig.text(.06,.16,'A：原双专家 seed06    B：双专家 seed08    C：校准双专家75    D：单骨干 global-cosine',fontsize=10,color='#34475d')
fig.text(.06,.11,'三模型 ABD 比四模型高 0.0059 分，参数少 28.5%；差值很小，不能据此证明泛化精度提升。',fontsize=10,color='#34475d')
fig.text(.06,.06,'内部专家图是训练后干预，不等于去掉分支重训。所有结果来自重复开发的第五折，非完整 OOF 或独立测试成绩。',fontsize=9,color='#6b7788')
fig.savefig(OUT/'attribution_summary.png',dpi=150)
fig.savefig(OUT/'attribution_summary.svg')
print(str(OUT/'attribution_summary.png'))
