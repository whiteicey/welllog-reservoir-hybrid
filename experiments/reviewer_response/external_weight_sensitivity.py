"""External-gate weight sensitivity using frozen conditional densities and OOF gates."""
from pathlib import Path
import os
import hashlib,importlib.util,json,sys
import numpy as np
import torch
import torch.nn.functional as F
MAIN=Path(os.environ.get("WELLLOG_ROOT", Path(__file__).resolve().parents[2]));RT=MAIN/'v97_verify/runtime/v97'
PACK=MAIN/'outputs/mega_pack/submission_code_mega/runtime'
sys.path.insert(0,str(PACK));sys.path.insert(0,str(RT/'experiments/pure_dl_s11'));sys.path.insert(0,str(RT/'package'));sys.path.insert(0,str(RT/'experiments/pure_dl'))
import train_invariant as base
from por_t3_patch_loss import probability
from src.data import load_split
TS=['POR','PERM','SW'];WEIGHTS=[0.,.1,.2,.35,.5,.7,1.];HI=40.;DEV='cuda'

def logit(p):
 p=np.clip(p,1e-6,1-1e-6);return np.log(p)-np.log(1-p)

def main():
 out=MAIN/'paper_workspace/reviewer_response_20261117/external';out.mkdir(parents=True,exist_ok=True)
 if (out/'metrics.json').exists():raise RuntimeError('Refusing to overwrite completed analysis')
 pairs=load_split(MAIN/'data','train');records=[f for _,f in pairs];n_rows=[len(f) for f in records]
 starts=np.r_[0,np.cumsum(n_rows)];ok=[np.isfinite(np.stack([f[t].to_numpy(np.float64) for t in TS],1)).all(1) for f in records]
 extz=np.load(MAIN/'outputs/gate_coord/coord_gate_probs.npz');ext={t:extz[t] for t in TS}
 consts=np.load(MAIN/'outputs/param_export/decode_consts.npz')
 rep={str(w):{'folds':{}} for w in WEIGHTS}
 accum={w:{t:[0.,0] for t in TS} for w in WEIGHTS}
 for f in range(1,6):
  P=np.load(MAIN/f'outputs/param_export/params_f{f}.npz')
  held=sorted({int(k.split('_')[0]) for k in P.keys() if k.endswith('_raw')})
  cache=[]
  for c,t in enumerate(TS):
   actions=consts[f'f{f}_actions_{c}'];centers=consts[f'f{f}_centers_{c}'];edges=torch.as_tensor(consts[f'f{f}_edges_{c}'],dtype=torch.float32,device=DEV)
   util=torch.as_tensor(base.kernel_matrix(actions,np.r_[base.DEFAULTS[t],centers],t).T,dtype=torch.float32,device=DEV)
   A=util[0,:].detach().cpu().numpy()
   for wi in held:
    gdl=P[f'{wi}_c{c}_g'].astype(np.float32);raw=P[f'{wi}_c{c}_raw'].astype(np.float32)
    wt=torch.as_tensor(raw[:,0,:],device=DEV);mu=torch.as_tensor(raw[:,1,:],device=DEV);sd=F.softplus(torch.as_tensor(raw[:,2,:],device=DEV))+.025
    cp=probability(c,wt.softmax(-1),mu,sd,edges);B=(cp@util[1:,:]).detach().cpu().numpy()
    sl=slice(starts[wi],starts[wi+1]);m=ok[wi];y=records[wi][t].to_numpy(np.float64)[m]
    cache.append({'c':c,'t':t,'actions':actions,'A':A,'B':B[m],'gdl':gdl[m],'ext':ext[t][sl][m],'y':y})
  preds={w:{t:[] for t in TS} for w in WEIGHTS};truth={t:[] for t in TS}
  for w in WEIGHTS:
   for z in cache:
    ld=logit(z['gdl']);le=-logit(z['ext']);g=1/(1+np.exp(-((1-w)*ld+w*le)))
    U=g[:,None]*z['A'][None,:]+(1-g[:,None])*z['B'];p=z['actions'][U.argmax(-1)]
    if z['t']=='PERM':p=np.clip(p,.01,HI)
    preds[w][z['t']].append(p)
   for t in TS:
    # Truth is appended once below; each target has identical complete-label rows per well.
    pass
  for t in TS:truth[t]=[z['y'] for z in cache if z['t']==t]
  for w in WEIGHTS:
   ys={t:np.concatenate(truth[t]) for t in TS};ps={t:np.concatenate(preds[w][t]) for t in TS}
   ycat=np.column_stack([ys[t] for t in TS]);pcat=np.column_stack([ps[t] for t in TS]);sc=base.metric(ycat,pcat)
   rep[str(w)]['folds'][str(f)]={k:float(v) for k,v in sc.items()}
   for t in TS:
    accum[w][t][0]+=float(sc[t])*len(ys[t]);accum[w][t][1]+=len(ys[t])
  print('EXTERNAL_FOLD',f,flush=True)
 for w in WEIGHTS:
  ch={t:accum[w][t][0]/accum[w][t][1] for t in TS};ch['TOTAL']=.3*ch['POR']+.35*ch['PERM']+.35*ch['SW']
  rep[str(w)]['pooled']=ch;rep[str(w)]['mean_fold_TOTAL']=float(np.mean([rep[str(w)]['folds'][str(f)]['TOTAL'] for f in range(1,6)]))
  rep[str(w)]['std_fold_TOTAL']=float(np.std([rep[str(w)]['folds'][str(f)]['TOTAL'] for f in range(1,6)]))
  rep[str(w)]['fold_wins_vs_w0']=sum(rep[str(w)]['folds'][str(f)]['TOTAL']>rep['0.0']['folds'][str(f)]['TOTAL'] for f in range(1,6))
 best=max(WEIGHTS,key=lambda w:rep[str(w)]['mean_fold_TOTAL']);ordered=sorted(WEIGHTS,key=lambda w:rep[str(w)]['mean_fold_TOTAL'],reverse=True)
 result={'protocol':'Frozen v4 conditional-density OOF parameters; coordinate-consistent external gates; PERM cap 40; dense two-expert logit mixture','weights':WEIGHTS,'results':rep,'best_weight':best,'rank_of_recorded_0.35':ordered.index(.35)+1,'best_weight_per_fold':{str(f):max(WEIGHTS,key=lambda w:rep[str(w)]['folds'][str(f)]['TOTAL']) for f in range(1,6)},'source_sha256':hashlib.sha256((MAIN/'_gate_coord.py').read_bytes()).hexdigest()}
 (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'best':best,'rank035':result['rank_of_recorded_0.35'],'scores':{str(w):rep[str(w)]['mean_fold_TOTAL'] for w in WEIGHTS}},indent=2),flush=True)
if __name__=='__main__':main()
