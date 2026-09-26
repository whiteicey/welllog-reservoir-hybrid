"""Label-free train-well audit: same raw points under shifted contexts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import train_invariant as base
from patch_hierarchy_sequence import Network
from por_t3_patch_loss import probability

torch.set_num_threads(3)
source=base.ROOT/'outputs/pure_dl_por_t3_lr0006_sequence'
ck=torch.load(source/'fold_5.pt',map_location='cpu',weights_only=False)
model=Network().cuda().eval();model.load_state_dict(ck['state_dict'])
held=set(ck['config']['outer_wells']['4'])
paths=[p for p in sorted((base.ROOT.parent/'data/train').glob('*.txt')) if p.stem not in held]
assert len(paths)==64
paths=[paths[i] for i in np.linspace(0,63,16,dtype=int)]
builder=base.FeatureBuilder();builder.medians_=ck['medians']
selected=ck['selected_feature_indices'];fm=np.asarray(ck['feature_median']);fs=np.asarray(ck['feature_scale'])
actions=[];kernels=[]
for j,name in enumerate(base.TS):
    ct=np.asarray(ck['centers'][j])
    a=np.linspace(ct[0],ct[-1],800) if j==2 else np.geomspace(max(ct[0]*.9,.001),ct[-1]*1.1,800)
    a=np.unique(np.r_[base.DEFAULTS[name],np.clip(a,*base.BOUNDS[name])]);actions.append(a)
    kernels.append(torch.tensor(base.kernel_matrix(a,np.r_[base.DEFAULTS[name],ct],name).T,device='cuda',dtype=torch.float32))
def decode(output):
    g,r=output;g=g.reshape(-1,3).float().sigmoid();r=r.reshape(-1,3,3,8).float()
    w=r[:,:,0].softmax(-1);mu=r[:,:,1];sd=torch.nn.functional.softplus(r[:,:,2])+.025
    pred=[];cdf=[]
    for j in range(3):
        edges=torch.as_tensor(ck['density_edges'][j],device='cuda',dtype=torch.float32)
        cp=probability(j,w[:,j],mu[:,j],sd[:,j],edges)
        joint=torch.cat([g[:,j:j+1],(1-g[:,j:j+1])*cp],1)
        pred.append(actions[j][(joint@kernels[j]).argmax(-1).cpu().numpy()]);cdf.append(cp.cumsum(-1).cpu().numpy())
    return np.stack(pred,1),g.cpu().numpy(),np.stack(cdf,1)
all_pred=[];all_gate=[];all_cdf=[];wells=[]
with torch.no_grad():
    for path in paths:
        frame=pd.read_csv(path,skiprows=[1],usecols=lambda name:name in base.INPUT_COLUMNS)
        assert not set(base.TS)&set(frame.columns) and path.stem not in held
        frame=frame.replace(-99999.,np.nan)
        for name in frame.columns:
            frame[name]=pd.to_numeric(frame[name],errors='coerce');lo,hi=base.VALID_RANGES[name]
            frame.loc[(frame[name]<lo)|(frame[name]>hi),name]=np.nan
        _,enhanced=builder._transform_frames(frame)
        assert [enhanced.columns[i] for i in selected]==ck['selected_feature_names']
        value=np.nan_to_num(enhanced.to_numpy(dtype=np.float32,copy=True))[:,selected]
        x=torch.tensor(np.clip((value-fm)/fs,-8,8).astype(np.float16),device='cuda')
        depth=torch.tensor(frame['DEPTH'].to_numpy(np.float32),device='cuda');assert torch.isfinite(depth).all()
        centers=torch.tensor(np.linspace(160,len(frame)-161,4,dtype=int),device='cuda')
        views=[]
        for shift,start in [(0,64),(32,32)]:
            anchors=centers+shift;ix=anchors[:,None]+torch.arange(-128,128,device='cuda')
            assert int(ix.min())>=0 and int(ix.max())<len(frame)
            relative=torch.tanh((depth[ix]-depth[anchors,None])/10)
            absolute=((depth[ix]-ck['absolute_depth_median'])/ck['absolute_depth_scale']).clamp(-8,8)
            with torch.autocast('cuda',dtype=torch.bfloat16):g,r=model(x[ix],relative,torch.ones_like(ix,dtype=torch.bool),absolute_depth=absolute)
            raw_ids=ix[:,start:start+128]
            if not shift:reference_ids=raw_ids.clone()
            else:assert torch.equal(reference_ids,raw_ids)
            views.append(decode((g[:,start:start+128],r[:,start:start+128])))
        all_pred.append(np.stack([v[0] for v in views]));all_gate.append(np.stack([v[1] for v in views]));all_cdf.append(np.stack([v[2] for v in views]));wells.append(path.stem)
        print('AUDITED',path.stem,flush=True)
pred=np.concatenate(all_pred,axis=1);gate=np.concatenate(all_gate,axis=1);cdf=np.concatenate(all_cdf,axis=1)
results={}
for j,name in enumerate(base.TS):
    distance=abs(base.TRANSFORM[j](pred[0,:,j])-base.TRANSFORM[j](pred[1,:,j]))
    default=base.DEFAULTS[name]
    changed=(pred[0,:,j]==default)!=(pred[1,:,j]==default)
    results[name]={'mean_abs_gate_change':float(abs(gate[0,:,j]-gate[1,:,j]).mean()),'mean_conditional_CDF_L1':float(abs(cdf[0,:,j]-cdf[1,:,j]).mean()),'decoded_default_change_pct':float(changed.mean()*100),'decoded_value_change_pct':float((pred[0,:,j]!=pred[1,:,j]).mean()*100),'transformed_prediction_distance_median_p90_p99':np.quantile(distance,[.5,.9,.99]).tolist()}
result={'scope':'No labels read;16 training wells,4 windows each; same core128 raw points under center+32 shift, original decode, no prediction averaging or model updates. Not an accuracy score.','source':str(source),'wells':wells,'point_pairs':pred.shape[1],'targets':results}
out=base.ROOT/'outputs/window_shift_diagnostic';out.mkdir(exist_ok=True);(out/'diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
