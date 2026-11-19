"""Build the reviewer-response sensitivity and controlled-ablation figure."""
from pathlib import Path
import ast,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_style import apply_style,clean_axis,panel_label,NAVY,TEAL,PURPLE,GREY,RED,DARK
ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'experiments/reviewer_response/results';OUT=ROOT/'paper/current/figures'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
 fusion=read(RES/'fusion_controlled.json');ext=read(RES/'external_sensitivity.json');public=read(RES/'public_gate_sensitivity.json')
 bins={k:read(RES/f'perm_bins{k}_report.json') for k in [30,60,120]}
 apply_style();fig,axs=plt.subplots(2,3,figsize=(12,7.0));fig.subplots_adjust(left=.07,right=.985,bottom=.10,top=.94,wspace=.32,hspace=.48)
 colors={'POR':NAVY,'PERM':TEAL,'SW':PURPLE}
 # (a) Marginal tree-weight sensitivity, maximised over remaining network weights.
 ax=axs[0,0]
 for t in ['POR','PERM','SW']:
  raw=fusion['targets'][t]['continuous_only_by_weight'];d={ast.literal_eval(k):v for k,v in raw.items()}
  selected=tuple(fusion['targets'][t].get('global_choice',{}).get('cont_w',[]))
  if not selected:
   # The concise report retains original fold choices; use the modal tree weight.
   selected=max(d,key=d.get)
  base=d[selected];ws=sorted({w[0] for w in d});vals=[100*(max(v for w,v in d.items() if w[0]==x)-base) for x in ws]
  ax.plot(ws,vals,marker='o',ms=4,lw=1.2,label=t,color=colors[t])
 ax.axhline(0,color=GREY,lw=.8);ax.set_xlabel('Tree weight (network weights re-optimised)');ax.set_ylabel('Δ non-default utility (points)');ax.legend(frameon=False,ncol=3);panel_label(ax,'a','Fusion-weight sensitivity');clean_axis(ax)
 # (b) Internal gate-threshold sensitivity relative to v2.
 ax=axs[0,1]
 for t in ['POR','PERM','SW']:
  d=fusion['targets'][t];th=np.array(d['threshold_sensitivity']['grid']);v=np.array(d['threshold_sensitivity']['micro']);delta=100*(v-d['v2']['micro'])
  ax.plot(th,delta,lw=1.2,color=colors[t],label=t)
  sel=float(np.median(d['fold_selected_thresholds']));ax.axvline(sel,color=colors[t],lw=.65,alpha=.45)
 ax.axhline(0,color=DARK,lw=.8);ax.set_xlabel('Default-state threshold');ax.set_ylabel('Δ target score vs v2 (points)');panel_label(ax,'b','Threshold and ranking stability');clean_axis(ax)
 # (c) External weight.
 ax=axs[0,2];ws=np.array(ext['weights']);means=[ext['results'][str(w)]['mean_fold_TOTAL'] for w in ws]
 ax.plot(ws,means,marker='D',ms=4,lw=1.25,color=NAVY);ax.axvline(.35,color=RED,lw=.8,ls='--');ax.set_xlabel('External-gate weight');ax.set_ylabel('Mean five-fold total score');ax.set_xticks(ws);panel_label(ax,'c','External-gate weight');clean_axis(ax)
 # (d) Matched PERM bin-count retraining, effect over v2 with paired well CI.
 ax=axs[1,0];ks=[30,60,120];vals=[];lo=[];hi=[]
 for k in ks:
  r=bins[k]['full_v2gate']['utility'];e=r['paired_vs_v2'];vals.append(100*e['delta_micro']);lo.append(100*(e['delta_micro']-e['ci95'][0]));hi.append(100*(e['ci95'][1]-e['delta_micro']))
 ax.errorbar(ks,vals,yerr=[lo,hi],fmt='o-',color=TEAL,capsize=3,lw=1.2,ms=5);ax.axhline(0,color=DARK,lw=.8);ax.set_xticks(ks);ax.set_xlabel('PERM quantile-bin count');ax.set_ylabel('Δ target score vs v2 (points)');panel_label(ax,'d','Matched two-seed bin retraining');clean_axis(ax)
 # (e) Public threshold sweep.
 ax=axs[1,1];rows=public['rows'];th=[r['threshold'] for r in rows];rm=[r['rmse_all3'] for r in rows]
 ax.plot(th,rm,color=PURPLE,lw=1.25);ax.axhline(public['ungated_rmse'],color=DARK,lw=.8,ls='--',label='Ungated tree');ax.axvline(.5,color=RED,lw=.8,ls=':',label='Recorded 0.5');ax.set_xlabel('Public SW-gate threshold');ax.set_ylabel('Pooled RMSE');ax.legend(frameon=False);panel_label(ax,'e','Public post-hoc threshold sweep');clean_axis(ax)
 # (f) Controlled module effects: full minus ablated total, with well bootstrap CI.
 ax=axs[1,2];labels=['Tree\ndistribution','Sequence\ndistribution','Sequence\ngates','Joint-state\ngate'];keys=['no_treedist','no_seqdist','no_seqgate','no_joint'];vals=[];lo=[];hi=[]
 for k in keys:
  e=fusion['module_effects_vs_full'][k];vals.append(e['effect']);lo.append(e['effect']-e['ci95'][0]);hi.append(e['ci95'][1]-e['effect'])
 x=np.arange(4);ax.errorbar(x,vals,yerr=[lo,hi],fmt='o',color=NAVY,capsize=3,ms=5,lw=1.1);ax.axhline(0,color=DARK,lw=.8);ax.set_xticks(x,labels);ax.set_ylabel('Full − ablated total (points)');panel_label(ax,'f','Controlled fusion module effects');clean_axis(ax)
 fig.savefig(OUT/'reviewer_sensitivity.pdf',bbox_inches='tight');fig.savefig(OUT/'reviewer_sensitivity.png',dpi=240,bbox_inches='tight');plt.close(fig)
 files=[RES/'fusion_controlled.json',RES/'external_sensitivity.json',RES/'public_gate_sensitivity.json']+[RES/f'perm_bins{k}_report.json' for k in ks]
 meta={'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'reviewer_sensitivity.pdf',OUT/'reviewer_sensitivity.png']}}
 (OUT/'reviewer_sensitivity_data.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
