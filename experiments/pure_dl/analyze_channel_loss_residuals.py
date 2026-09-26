"""Loss suitability evidence from only training-well labels and residuals."""
import json
import numpy as np
import torch
from scipy.stats import skew,kurtosis
import train_invariant as base
out=base.ROOT/'outputs/channel_loss_analysis';source=base.ROOT/'outputs/pure_dl_blur_no_dropout_sequence'
a=np.load(source/'oof.npz');truth=a['truth'];fold=a['fold'];well=a['well'];d=base.dmask(truth)
b=np.load(out/'sample_predictions.npz');m=fold[b['row_indices']]!=0;ids=b['row_indices'][m];pred=b['pred'][m];mean=b['conditional_z_mean'][m]
assert np.all(fold[ids]!=0) and len(np.unique(well[ids]))==64
ck=torch.load(source/'fold_1.pt',map_location='cpu',weights_only=False)
transformed=mean*np.asarray(ck['regression_scale'])+np.asarray(ck['regression_median'])
point=np.column_stack([np.expm1(transformed[:,0]),10**transformed[:,1],np.exp(transformed[:,2])])
def summarize(residual):
 r=np.asarray(residual);e=np.abs(r);n=max(1,int(np.ceil(len(e)*.01)));order=np.argsort(e)[-n:]
 h=np.where(e<1,.5*e*e,e-.5)
 return {'n':len(e),'signed_median':float(np.median(r)),'absolute_q50_q90_q99':np.quantile(e,[.5,.9,.99]).tolist(),'skewness':float(skew(r)),'excess_kurtosis':float(kurtosis(r)), 'mse_top1percent_loss_share':float((e[order]**2).sum()/(e**2).sum()),'mae_top1percent_loss_share':float(e[order].sum()/e.sum()),'huber_delta1_top1percent_loss_share':float(h[order].sum()/h.sum()),'outside_score_support_fraction':float((e>=1).mean())}
result={}
for j,name in enumerate(base.TS):
 full=truth[(fold!=0)&~d[:,j],j];active=~d[ids,j];y=truth[ids[active],j]
 item={'train_nondefault_rows':len(full),'raw_label_q01_q25_q50_q75_q99':np.quantile(full,[.01,.25,.5,.75,.99]).tolist(),'raw_label_skewness':float(skew(full)),'train_sample_nondefault_rows':int(active.sum()),'residuals':{}}
 for desc,pp in [('actual_decoded_prediction',pred[active,j]),('inverse_conditional_transformed_mean',point[active,j])]:
  r=np.log10(np.maximum(pp,1e-8)/y) if j==1 else (pp-y)/((.08 if j==0 else .05)*(np.abs(y)+1e-5))
  item['residuals'][desc]=summarize(r)
 # Robust residual in log metric units, no clipping of original targets.
 logerr=np.log(np.maximum(point[active,j],1e-8)/y)/(np.log(10) if j==1 else np.log(1.08 if j==0 else 1.05))
 item['log_residuals']=summarize(logerr)
 result[name]=item
report={'scope':'Only 64 first-fold training wells; full labels and fixed sampled in-sample residuals; no optimizer update or validation-fit loss tuning','score_normalization':'POR 8% relative, PERM log10 ratio, SW5% relative; absolute residual>=1 corresponds to zero-score region, before official prediction clipping','channels':result}
(out/'loss_analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2),flush=True)
