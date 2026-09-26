"""Read-only dataset EDA and OOF plots. Outputs standalone figures and a gallery."""
from pathlib import Path
import sys,json,html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'package'))
from src.data import load_split,INPUT_COLUMNS,TARGET_COLUMNS
OUT=ROOT/'outputs/data_analysis';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':110})
COLS=[c for c in INPUT_COLUMNS if c!='CASE'];T=TARGET_COLUMNS
COLOR=['#2463a6','#df7932','#399568'];defs=np.array([.1,.01,99.9])
units={'DEPTH':'m','GR':'API*','PE':'b/e','SP':'mV','CAL':'cm','AC':'us/m','DEN':'g/cm3','CNL':'%/pu','RXO':'ohm m','RT':'ohm m','DEVI':'deg','AZIM':'deg','BIT':'unspecified'}
records={s:load_split(ROOT.parent/'data',s) for s in ['train','test']}
frames={s:pd.concat([f for _,f in rs],ignore_index=True) for s,rs in records.items()}
gallery=[]
def save(fig,name,title,note):
    fig.savefig(OUT/f'{name}.png',dpi=170,bbox_inches='tight');plt.close(fig)
    gallery.append((name,title,note))

stats=[];wellstats=[]
for split,rs in records.items():
    for wid,f in rs:
        d={'split':split,'well':wid,'rows':len(f),'top':float(f.DEPTH.min()),'bottom':float(f.DEPTH.max())}
        for c in T:
            if c in f:
                s=f[c].dropna();default=s<=defs[T.index(c)] if c!='SW' else s>=99.9
                d[c+'_default_fraction']=float(default.mean())
        wellstats.append(d)
    for c in INPUT_COLUMNS+(T if split=='train' else []):
        s=frames[split][c] if c in frames[split] else pd.Series(np.nan,index=frames[split].index)
        q=s.quantile([.01,.05,.25,.5,.75,.95,.99])
        stats.append({'split':split,'column':c,'rows':len(s),'valid':int(s.notna().sum()),'missing_pct':float(s.isna().mean()*100),
                      'min':s.min(),'max':s.max(),'mean':s.mean(),'std':s.std(),**{f'p{int(k*100):02}':v for k,v in q.items()}})
pd.DataFrame(stats).to_csv(OUT/'column_statistics.csv',index=False,encoding='utf-8-sig')
ws=pd.DataFrame(wellstats);ws.to_csv(OUT/'well_statistics.csv',index=False,encoding='utf-8-sig')

fig,ax=plt.subplots(figsize=(12,4))
miss=np.array([[frames[s][c].isna().mean()*100 if c in frames[s] else 100 for c in INPUT_COLUMNS] for s in ['train','test']])
im=ax.imshow(miss,cmap='YlOrRd',vmin=0,vmax=100,aspect='auto')
ax.set_xticks(range(14),INPUT_COLUMNS);ax.set_yticks([0,1],['Train: 80 wells','Test: 10 wells'])
for i in range(2):
 for j in range(14): ax.text(j,i,f'{miss[i,j]:.2f}%',ha='center',va='center',color='white' if miss[i,j]>60 else 'black',fontsize=9)
fig.colorbar(im,ax=ax,label='Missing / out-of-range (%)');ax.set_title('Missing values after repository cleaning rules')
save(fig,'01_missing','缺失与异常值','按原代码范围清洗后统计；缺列按缺失计。CASE 基本无有效信息。')

fig,axes=plt.subplots(4,4,figsize=(16,12));tailnotes=[]
for ax,c in zip(axes.flat,COLS):
    values=[frames[s][c].dropna().to_numpy() for s in ['train','test']]
    log=c in ['GR','RXO','RT']
    if log: values=[np.log1p(v) for v in values]
    both=np.concatenate(values);lo,hi=np.quantile(both,[.005,.995]);hi=max(hi,lo+1e-6)
    for v,col,label in zip(values,COLOR,['Train','Test']):
        ax.hist(v,bins=np.linspace(lo,hi,55),density=True,histtype='step',color=col,label=label,lw=1.5)
    ax.set_title(c);ax.set_xlabel(f'log1p({c})' if log else units.get(c,''));ax.set_ylabel('Density');ax.grid(alpha=.15)
for ax in list(axes.flat)[len(COLS):]:ax.axis('off')
axes.flat[0].legend();fig.suptitle('Input distributions: train vs test | central 99% shown; tails not deleted',fontsize=15)
fig.tight_layout(rect=[0,0,1,.96])
save(fig,'02_input_distributions','13 个输入字段的分布','每组直方图单独归一化；显示合并分布的中央 99%，完整范围见 CSV。GR/RXO/RT 使用 log1p 横轴。训练表头 GR 为 API，测试标为 mg，单位元数据不一致，图中未换算。')

y=frames['train'][T].dropna().to_numpy();dm=np.column_stack([y[:,0]<=.1,y[:,1]<=.01,y[:,2]>=99.9])
fig,axes=plt.subplots(2,3,figsize=(14,8))
for j,t in enumerate(T):
    a=axes[0,j];fr=dm[:,j].mean()*100;a.bar(['Default','Non-default'],[fr,100-fr],color=[COLOR[0],COLOR[1]])
    for k,v in enumerate([fr,100-fr]):a.text(k,v+1,f'{v:.2f}%',ha='center')
    a.set_ylim(0,100);a.set_title(f'{t}: default={defs[j]}');a.set_ylabel('Valid rows (%)')
    v=y[~dm[:,j],j];v=np.log10(v) if t=='PERM' else v
    axes[1,j].hist(v,bins=65,color=COLOR[j],alpha=.85);axes[1,j].set_xlabel('log10(PERM)' if t=='PERM' else t)
    axes[1,j].set_ylabel('Non-default row count');axes[1,j].grid(alpha=.15)
fig.suptitle('Training targets | 723,556 rows with all three labels valid');fig.tight_layout(rect=[0,0,1,.95])
save(fig,'03_target_distributions','三个预测目标的分布','上排展示默认/非默认占比；下排仅看非默认标签，避免默认值尖峰遮住细节。PERM 使用 log10 横轴。')

fig,axes=plt.subplots(1,2,figsize=(16,9),gridspec_kw={'width_ratios':[1.25,1]})
ordered=ws.sort_values(['split','top']).reset_index(drop=True)
for i,r in ordered.iterrows():axes[0].plot([r.top,r.bottom],[i,i],color=COLOR[0] if r['split']=='train' else COLOR[1],lw=2)
axes[0].set_xlabel('Depth (m)');axes[0].set_ylabel('Wells sorted by split and interval top');axes[0].set_title('Logged depth intervals: blue=train, orange=test');axes[0].invert_yaxis()
tr=ws[ws.split=='train'].sort_values('POR_default_fraction').reset_index(drop=True)
for j,t in enumerate(T):axes[1].plot(np.arange(80),tr[t+'_default_fraction']*100,label=t,color=COLOR[j],lw=1.3)
axes[1].set_xlabel('Training wells sorted by POR default fraction');axes[1].set_ylabel('Default labels (%)');axes[1].set_title('Target composition varies across wells');axes[1].legend();axes[1].grid(alpha=.2)
fig.tight_layout();save(fig,'04_well_variation','井间差异','左图每条线是一口井的数据深度区间，不能把不同井相同深度直接当作同一地层。右图三条线使用同一个井排序。')

corr=np.empty((len(COLS),3));validframe=frames['train'].loc[frames['train'][T].notna().all(axis=1)]
for j,t in enumerate(T):
    f=validframe.loc[~dm[:,j],COLS+[t]]
    corr[:,j]=f.corr(method='spearman')[t].loc[COLS]
pd.DataFrame(corr,index=COLS,columns=T).to_csv(OUT/'nondefault_spearman.csv',encoding='utf-8-sig')
fig,ax=plt.subplots(figsize=(7,8));im=ax.imshow(corr,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
ax.set_xticks(range(3),T);ax.set_yticks(range(len(COLS)),COLS)
for i in range(len(COLS)):
 for j in range(3): ax.text(j,i,f'{corr[i,j]:.2f}',ha='center',va='center',color='white' if abs(corr[i,j])>.65 else 'black')
fig.colorbar(im,ax=ax,label='Spearman rho');ax.set_title('Input-target association on non-default labels')
save(fig,'05_correlations','输入与非默认目标的相关性','每一列分别筛除该目标的默认标签；跨井合并的 Spearman 相关不代表因果，也可能受井间差异影响。')

report=json.loads((ROOT/'outputs/simple_cnn_cv/report.json').read_text());oof=np.load(ROOT/'outputs/simple_cnn_cv/oof.npz')
ranking=sorted(report['per_well'],key=lambda w:report['per_well'][w]['TOTAL'])
selected=[('low',ranking[0]),('median',ranking[len(ranking)//2]),('high',ranking[-1])]
selection=[]
for tag,wid in selected:
    wi=next(i for i,(w,_) in enumerate(records['train']) if w==wid);f=records['train'][wi][1]
    v=f[T].notna().all(axis=1).to_numpy();yp=np.full((len(f),3),np.nan);yp[v]=oof['pred'][oof['well']==wi]
    score=report['per_well'][wid]['TOTAL'];selection.append({'role':tag,'well':wid,'score':score})
    fig,axes=plt.subplots(1,9,figsize=(19,10),sharey=True)
    tracks=['GR','AC','DEN','CNL','RT','RXO']+T
    for ax,c in zip(axes,tracks):
        ax.plot(f[c],f.DEPTH,color=COLOR[0],lw=.65,label='Observed / true')
        if c in T:ax.plot(yp[:,T.index(c)],f.DEPTH,color=COLOR[1],lw=.7,alpha=.9,label='CNN OOF')
        if c in ['RT','RXO','PERM']:ax.set_xscale('log');ax.set_xlim(left=.01)
        ax.set_title(c);ax.grid(alpha=.2);ax.tick_params(axis='x',labelsize=8);ax.set_xlabel(units.get(c,c))
    axes[0].invert_yaxis();axes[0].set_ylabel('Depth (m)');axes[-1].legend(loc='upper right',fontsize=8)
    fig.suptitle(f'{tag.upper()} OOF-score well: {wid} | score={score:.2f} | full interval, no subsampling',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.95]);save(fig,f'06_well_{tag}',f'单井深度曲线：{tag}（得分 {score:.2f}）','按本次 CNN 的逐井 OOF 得分选出最低、中位、最高井，非随机样本；蓝线为观测/真实标签，橙线为折外预测，深度向下增加，缺失标签处断线。')
    # Fixed 100 m window with maximum number of non-default POR labels (not selected by prediction error).
    non=(f.POR>.1).astype(int);count=non.rolling(1001,min_periods=1).sum().to_numpy();end=int(np.argmax(count))+1;start=max(0,end-1001)
    fig,axes=plt.subplots(3,1,figsize=(14,8),sharex=True)
    for j,t in enumerate(T):
        axes[j].plot(f.DEPTH.iloc[start:end],f[t].iloc[start:end],color=COLOR[0],lw=1.3,label='True')
        axes[j].plot(f.DEPTH.iloc[start:end],yp[start:end,j],color=COLOR[1],lw=1.1,label='CNN OOF')
        if t=='PERM':axes[j].set_yscale('log')
        axes[j].set_ylabel(t);axes[j].grid(alpha=.2)
    axes[0].legend();axes[-1].set_xlabel('Depth (m)');fig.suptitle(f'{tag.upper()} well: 100 m window with most non-default POR labels | {wid}')
    fig.tight_layout(rect=[0,0,1,.95]);save(fig,f'07_detail_{tag}',f'局部预测对比：{tag}','在该井选择非默认 POR 标签最多的约 100 米窗口，所有点保留；此窗口用于观察曲线形态，不代表总体误差。')

summary={'train_wells':80,'test_wells':10,'train_rows':len(frames['train']),'test_rows':len(frames['test']),
         'complete_target_rows':len(y),'default_pct':dict(zip(T,(dm.mean(0)*100).tolist())),
         'selected_wells':selection,'missing_pct':{s:dict(zip(INPUT_COLUMNS,miss[i].tolist())) for i,s in enumerate(['train','test'])},
         'input_shift':{c:{s:float(frames[s][c].median()) for s in ['train','test']} for c in COLS}}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
cards=''.join(f'<section id="{n}"><h2>{html.escape(title)}</h2><p>{html.escape(note)}</p><a href="{n}.png"><img loading="lazy" src="{n}.png" alt="{html.escape(title)}"></a></section>' for n,title,note in gallery)
nav=''.join(f'<a href="#{n}">{html.escape(title)}</a> ' for n,title,_ in gallery)
(OUT/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>测井数据与 CNN 折外预测分析</title><style>body{font-family:system-ui,sans-serif;max-width:1500px;margin:32px auto;padding:0 24px;color:#203044;background:#f8fafc}h1{font-size:28px}p{line-height:1.8}nav{line-height:2.2}nav a{margin-right:16px}section{margin:32px 0;padding:24px;background:white;border:1px solid #dde3ea}img{width:100%;height:auto}a{color:#2463a6}</style><h1>测井数据特点与简单 CNN 折外预测</h1><p>80 口训练井 / 10 口测试井；统计基于原仓库清洗规则。图中目标值按原始数据标度展示；测试井无目标标签。CNN 整体五折 OOF 得分 77.136。点击图片可查看完整分辨率。</p><nav>'+nav+'</nav>'+cards+'</html>',encoding='utf-8')
print(json.dumps(summary,indent=2));print('figures',len(gallery),'gallery',OUT/'index.html')
