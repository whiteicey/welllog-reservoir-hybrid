"""Public SW-gate threshold sensitivity; only the gate is refit, trees stay frozen."""
from pathlib import Path
import os
import hashlib,json,sys
import numpy as np
MAIN=Path(os.environ.get("WELLLOG_ROOT", Path(__file__).resolve().parents[2]));sys.path.insert(0,str(MAIN))
from run_transfer_pdda2021 import TARGETS,adapt,build_features,cb_gate,load,rmse3,rmse_per

def main():
 out=Path(__file__).resolve().parent/'output/public_gate';out.mkdir(parents=True,exist_ok=True)
 if (out/'metrics.json').exists():raise RuntimeError('Refusing to overwrite completed analysis')
 tr,te=load();_,(Xtr,_,Ytr,Wtr,_),(Xte,_,Yte,Wte,Dte)=build_features(tr,te);lab=np.isfinite(Ytr).all(1);dtr=Ytr[:,1]>=.999
 saved=np.load(MAIN/'reproducibility/public_volve/output/predictions.npz');preg=saved['tree_regression']
 assert np.allclose(saved['truth'],Yte,equal_nan=True) and np.array_equal(saved['well'],Wte)
 gate=cb_gate(Xtr[lab],dtr[lab],Xte,seed=42)
 thresholds=np.round(np.linspace(.05,.95,19),2);base=rmse3(preg,Yte);rows=[]
 for th in thresholds:
  p=preg.copy();p[:,1]=np.where(gate>=th,1.,p[:,1]);per={}
  for w in np.unique(Wte):
   m=Wte==w;per[str(int(w))]=rmse3(p[m],Yte[m])
  rows.append({'threshold':float(th),'rmse_all3':rmse3(p,Yte),**rmse_per(p,Yte),'gate_accuracy':float(((gate>=th)==(Yte[:,1]>=.999)).mean()),'default_fraction':float((gate>=th).mean()),'improvement_vs_ungated':base-rmse3(p,Yte),'per_well_rmse':per})
 best=min(rows,key=lambda r:r['rmse_all3']);record=next(r for r in rows if abs(r['threshold']-.5)<1e-8)
 result={'protocol':'Released-test post-hoc sensitivity; public wells were used in prior development; descriptive, not a fresh blind selection','thresholds':thresholds.tolist(),'rows':rows,'ungated_rmse':base,'recorded_threshold_0.5':record,'posthoc_best':best,'fraction_thresholds_better_than_ungated':float(np.mean([r['rmse_all3']<base for r in rows])),'source_sha256':hashlib.sha256((MAIN/'run_transfer_pdda2021.py').read_bytes()).hexdigest()}
 (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8');np.savez_compressed(out/'gate_probabilities.npz',gate=gate,truth=Yte,well=Wte)
 print(json.dumps({'ungated':base,'recorded':record,'best':best,'improving_fraction':result['fraction_thresholds_better_than_ungated']},indent=2),flush=True)
if __name__=='__main__':main()
