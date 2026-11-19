"""Controlled fusion sensitivity and leave-one-module-out analysis on frozen OOF learners.

All arms share the same five well folds, cached base learners, simplex grid,
threshold grid and HMM-off protocol. Only the available fusion module set changes.
"""
from pathlib import Path
import os
import argparse
import hashlib
import itertools
import json
import sys
import time
import numpy as np

MAIN = Path(os.environ.get("WELLLOG_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(MAIN))
from harness import DEFAULTS, BOUNDS, V2_CAL, acc_vec, breakdown, default_mask, kernel_matrix, total, paired_well_bootstrap

TS = ('POR','PERM','SW'); TI={t:i for i,t in enumerate(TS)}
TINV={'POR':np.expm1,'PERM':lambda x:10**x,'SW':lambda x:x}
TH=np.round(np.linspace(.02,.9,89),4)
WEIGHTS={'POR':.30,'PERM':.35,'SW':.35}


def simplex(k,step=.25):
 n=int(round(1/step))
 return [tuple(c/n for c in combo) for combo in itertools.product(range(n+1),repeat=k) if sum(combo)==n]


def dense_actions(sup,t):
 return np.linspace(sup[0],sup[-1],800) if t=='SW' else np.exp(np.linspace(np.log(sup[0]*.9),np.log(sup[-1]*1.1),800))


def load_source(name,t,sel):
 if name.startswith('nn_'):
  with np.load(MAIN/'outputs'/name/'oof.npz') as z:
   return z[f'{t}_prob'][sel].astype(np.float32),z[f'{t}_centers'].astype(np.float32)
 with np.load(MAIN/'outputs'/name/'oof.npz') as z:return z['prob'][sel].astype(np.float32),z['centers'].astype(np.float32)


def prepare_continuous(t,y,folds,rows_by_fold,out,sel):
 cache=out/f'{t}_continuous.npy'
 cw=simplex(3)
 if cache.exists():return np.load(cache),cw
 names={'POR':['por_bins80_base_extra','nn_base_c','nn_gate_b'],
        'PERM':['perm_bins60_base','nn_base_c','nn_gate_b'],
        'SW':['sw_bins60_enh','nn_base_c','nn_gate_b']}[t]
 src=[load_source(n,t,sel) for n in names]
 pred=np.zeros((len(cw),len(y)),np.float32)
 for f,rows in enumerate(rows_by_fold):
  supports=[TINV[t](c[f]) for p,c in src];grid=dense_actions(supports[0],t)
  for s in range(0,len(rows),2048):
   rr=rows[s:s+2048];utils=[]
   for (p,c),sup in zip(src,supports):
    pp=p[rr].astype(np.float64);pp/=pp.sum(1,keepdims=True)
    utils.append(pp@kernel_matrix(grid,sup,t).T)
   for j,w in enumerate(cw):
    u=0
    for k,wk in enumerate(w):
     if wk:u=u+wk*utils[k]
    pred[j,rr]=grid[np.argmax(u,1)]
  print('CONT',t,f,len(rows),flush=True)
 pred=np.clip(pred,*BOUNDS[t])
 np.save(cache,pred)
 return pred,cw


def prepare_gates(t,n,V2,states,nn1,nn2):
 g0=V2[{'POR':'por_gate','PERM':'perm_gate','SW':'sw_gate'}[t]]
 post=states['post'];g1={'POR':post[...,0],'PERM':post[...,0]+post[...,3],'SW':post[...,0]+post[...,2]}[t]
 key=f'{t}_gate'
 G=np.stack([g0,g1,nn1[key],nn2[key]]).astype(np.float32)
 gw=simplex(4)
 gates=np.stack([np.asarray(w)@G for w in gw]).astype(np.float32)
 return G,gates,gw


def precompute_inner(t,cont,acc_cont,acc_default,gates,folds,rows_by_fold,out):
 path=out/f'{t}_inner_choices.npz'
 if path.exists():
  z=np.load(path);return z['score'],z['threshold']
 scores=np.full((5,len(gates),len(cont)),-np.inf,np.float32);ths=np.zeros_like(scores,dtype=np.int16)
 for f in range(5):
  inner=folds!=f;nin=int(inner.sum());base=acc_default[inner].sum()
  for gi in range(len(gates)):
   g=gates[gi,inner];order=np.argsort(-g);gs=g[order]
   cnt=np.searchsorted(-gs,-TH,side='right')
   ad=acc_default[inner]
   for ci in range(len(cont)):
    base=acc_cont[ci,inner].sum()
    gain=(ad-acc_cont[ci,inner])[order]
    cs=np.cumsum(gain)
    add=np.where(cnt>0,cs[np.clip(cnt-1,0,len(cs)-1)],0.)
    curve=(base+add)/nin;j=int(np.argmax(curve))
    scores[f,gi,ci]=curve[j];ths[f,gi,ci]=j
  print('INNER',t,f,flush=True)
 np.savez(path,score=scores,threshold=ths)
 return scores,ths


def evaluate_config(name,t,cont,acc_cont,gates,cw,gw,folds,rows_by_fold,scores,ths,y,wells):
 allowed_c=[i for i in range(len(cw))]
 allowed_g=[i for i in range(len(gw))]
 if 'no_seqdist' in name:allowed_c=[cw.index((1.,0.,0.))]
 if 'no_treedist' in name:allowed_c=[i for i,w in enumerate(cw) if w[0]==0]
 if 'no_joint' in name:allowed_g=[i for i,w in enumerate(gw) if w[1]==0]
 if 'no_seqgate' in name:allowed_g=[i for i,w in enumerate(gw) if w[2]==0 and w[3]==0]
 pred=np.zeros(len(y),np.float64);choices=[]
 for f,rows in enumerate(rows_by_fold):
  sub=scores[f][np.ix_(allowed_g,allowed_c)];a,b=np.unravel_index(np.argmax(sub),sub.shape)
  gi,ci=allowed_g[a],allowed_c[b];th=TH[ths[f,gi,ci]]
  pred[rows]=np.where(gates[gi,rows]>=th,DEFAULTS[t],cont[ci,rows])
  choices.append({'fold':f,'cont_w':cw[ci],'gate_w':gw[gi],'threshold':float(th),'inner_score':float(scores[f,gi,ci])})
 rep=breakdown(pred,y,t,wells,folds);rep.pop('per_well',None)
 return {'nested':rep,'choices':choices,'pred':pred,'acc':acc_vec(pred,y,t)}


def bootstrap_effect(acc_new,acc_old,y,wells,seed=20261117,n_boot=1000):
 # Total-score effect, retaining target-specific micro denominators per resample.
 ws=np.unique(wells);rng=np.random.default_rng(seed)
 sums=np.zeros((3,len(ws)));cnt=np.zeros((3,len(ws)))
 for j,w in enumerate(ws):
  m=wells==w
  for k,t in enumerate(TS):
   a=acc_new[k][m]-acc_old[k][m];sums[k,j]=a.sum();cnt[k,j]=m.sum()
 obs=100*sum(WEIGHTS[t]*sums[k].sum()/cnt[k].sum() for k,t in enumerate(TS))
 vals=np.empty(n_boot)
 for b in range(n_boot):
  ix=rng.integers(0,len(ws),len(ws))
  vals[b]=100*sum(WEIGHTS[t]*sums[k,ix].sum()/cnt[k,ix].sum() for k,t in enumerate(TS))
 return {'effect':float(obs),'ci95':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]}


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--update-target-effects',action='store_true');args=ap.parse_args()
 root=MAIN/'paper_workspace/reviewer_response_20261117';out=root/('smoke_fusion_v2' if args.smoke else 'fusion_v2');out.mkdir(parents=True,exist_ok=True)
 if not args.smoke and (out/'metrics.json').exists() and not args.update_target_effects:raise RuntimeError('Refusing to overwrite completed analysis')
 y=np.load(MAIN/'cache/tr_y.npy');folds=np.load(MAIN/'cache/tr_fold.npy');wells=np.load(MAIN/'cache/tr_well.npy')
 raw_n=len(y);sel=np.arange(raw_n)
 if args.smoke:
  sel=np.concatenate([np.flatnonzero(folds==f)[:1500] for f in range(5)]);y,folds,wells=y[sel],folds[sel],wells[sel]
 rows_by_fold=[np.flatnonzero(folds==f) for f in range(5)];n=len(y)
 def load_subset(path):
  with np.load(path) as z:return {k:(z[k][sel] if z[k].shape[0]==raw_n else z[k]) for k in z.files}
 V2=load_subset(MAIN/'outputs/cv_v2/oof.npz');states=load_subset(MAIN/'outputs/states_base_extra/oof.npz')
 nn1=load_subset(MAIN/'outputs/nn_gate_b/oof.npz');nn2=load_subset(MAIN/'outputs/nn_base_c/oof.npz')
 original=json.loads((MAIN/'outputs/assembly_D/report.json').read_text())
 report={'protocol':{'folds':5,'continuous_grid':'simplex .25, three sources','gate_grid':'simplex .25, four sources','hmm':'off in every controlled arm','threshold_grid':[.02,.9,89],'base_learners':'frozen OOF predictions; no base-learner retraining','interpretation':'conditional fusion/decoder LOMO, not end-to-end backbone ablation'},'targets':{}}
 names=['full','no_joint','no_seqgate','no_seqdist','no_treedist','no_seqdist_no_joint']
 all_preds={}
 for t in TS:
  ti=TI[t];yt=y[:,ti];cont,cw=prepare_continuous(t,yt,folds,rows_by_fold,out,sel)
  acc_cont=np.stack([acc_vec(c,yt,t) for c in cont]).astype(np.float32)
  acc_default=acc_vec(np.full(n,DEFAULTS[t]),yt,t)
  G,gates,gw=prepare_gates(t,n,V2,states,nn1,nn2)
  scores,ths=precompute_inner(t,cont,acc_cont,acc_default,gates,folds,rows_by_fold,out)
  reps={}
  for name in names:
   reps[name]=evaluate_config(name,t,cont,acc_cont,gates,cw,gw,folds,rows_by_fold,scores,ths,yt,wells)
   print('ARM',t,name,reps[name]['nested']['micro'],flush=True)
  # v2 reference under its frozen calibration
  if t=='POR':
   v2c=np.zeros(n)
   for f,rows in enumerate(rows_by_fold):v2c[rows]=np.expm1(V2['por_centers'][f])[np.argmax(V2['por_prob'][rows],1)]
   v2g=V2['por_gate']
  elif t=='PERM':v2c=np.clip(10**(V2['perm_log']*1.03),*BOUNDS[t]);v2g=V2['perm_gate']
  else:v2c=np.clip(V2['sw_reg']*1.05-3.,*BOUNDS[t]);v2g=V2['sw_gate']
  v2pred=np.where(v2g>=V2_CAL[t]['threshold'],DEFAULTS[t],v2c)
  # Threshold sensitivity conditional on original fold-specific continuous/gate choices.
  curves=[]
  for th in TH:
   p=np.zeros(n)
   for f,rows in enumerate(rows_by_fold):
    ch=original[t]['nested_choices'][f];ci=cw.index(tuple(ch['cont_w']));gi=gw.index(tuple(ch['gate_w']))
    p[rows]=np.where(gates[gi,rows]>=th,DEFAULTS[t],cont[ci,rows])
   curves.append(float(acc_vec(p,yt,t).mean()))
  clean={k:{q:v for q,v in r.items() if q not in ('pred','acc','choices')} for k,r in reps.items()}
  clean['v2']={'micro':float(acc_vec(v2pred,yt,t).mean())}
  clean['threshold_sensitivity']={'grid':TH.tolist(),'micro':curves,'above_v2_fraction':float(np.mean(np.array(curves)>clean['v2']['micro']))}
  clean['fold_selected_thresholds']=[c['threshold'] for c in original[t]['nested_choices']]
  clean['continuous_only_by_weight']=original[t]['continuous_only_by_weight']
  report['targets'][t]=clean
  all_preds[t]={'reps':reps,'v2acc':acc_vec(v2pred,yt,t)}
 # Total scores and module effects
 totals={name:100*sum(WEIGHTS[t]*all_preds[t]['reps'][name]['nested']['micro'] for t in TS) for name in names}
 effects={}
 for name in names[1:]:
  new=[all_preds[t]['reps']['full']['acc'] for t in TS];old=[all_preds[t]['reps'][name]['acc'] for t in TS]
  effects[name]=bootstrap_effect(new,old,y,wells,n_boot=50 if args.smoke else 1000)
 # Interaction: full - no_joint - no_seqdist + no_seqdist_no_joint
 interaction={}
 for t in TS:
  r=all_preds[t]['reps'];interaction[t]=float(r['full']['nested']['micro']-r['no_joint']['nested']['micro']-r['no_seqdist']['nested']['micro']+r['no_seqdist_no_joint']['nested']['micro'])
 accI={t:all_preds[t]['reps']['full']['acc']-all_preds[t]['reps']['no_joint']['acc']-all_preds[t]['reps']['no_seqdist']['acc']+all_preds[t]['reps']['no_seqdist_no_joint']['acc'] for t in TS}
 # Bootstrap total interaction using a synthetic zero old arm and weighted row differences.
 ws=np.unique(wells);rng=np.random.default_rng(20261117);vals=[]
 sums=np.zeros((3,len(ws)));cnt=np.zeros((3,len(ws)))
 for j,w in enumerate(ws):
  m=wells==w
  for k,t in enumerate(TS):sums[k,j]=accI[t][m].sum();cnt[k,j]=m.sum()
 obs=100*sum(WEIGHTS[t]*sums[k].sum()/cnt[k].sum() for k,t in enumerate(TS))
 for b in range(50 if args.smoke else 1000):
  ix=rng.integers(0,len(ws),len(ws));vals.append(100*sum(WEIGHTS[t]*sums[k,ix].sum()/cnt[k,ix].sum() for k,t in enumerate(TS)))
 interaction['TOTAL']={'effect':float(obs),'ci95':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]}
 report['total_scores']=totals;report['module_effects_vs_full']=effects;report['interaction_seqdist_x_jointgate']=interaction
 target_effects={}
 for t in TS:
  target_effects[t]={}
  for name in names[1:]:
   r=all_preds[t]['reps']
   target_effects[t][name]=paired_well_bootstrap(r['full']['acc'],r[name]['acc'],wells,n_boot=50 if args.smoke else 1000,seed=20261117)
 report['target_effects_vs_full']=target_effects
 report['validation_against_original']={'controlled_full_total':totals['full'],'original_total':original['TOTAL_nested'],'absolute_difference':abs(totals['full']-original['TOTAL_nested'])}
 report['source_sha256']={str(p.relative_to(MAIN)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [MAIN/'assemble.py',MAIN/'harness.py',MAIN/'final_cfg_v3.json']}
 (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps({'totals':totals,'effects':effects,'interaction':interaction},indent=2),flush=True)

if __name__=='__main__':main()
