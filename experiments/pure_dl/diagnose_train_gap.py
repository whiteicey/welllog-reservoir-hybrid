"""Read-only matched-window train/validation gap diagnostic; zero optimizer steps."""
import json,time,hashlib
import numpy as np
import torch
from torch.nn import functional as F
import train_invariant as base
from depth_norm_joint_sequence import Network
from multiscale_sequence import window_indices
from gpu_job_guard import check_slot
torch.set_num_threads(3);print('GPU_ADMISSION',check_slot(),flush=True);start=time.perf_counter()
folder=base.ROOT/'outputs/pure_dl_lr55_cosine_1e3';path=folder/'fold_1.pt';before=hashlib.sha256(path.read_bytes()).hexdigest()
ck=torch.load(path,map_location='cpu',weights_only=False)
rec=base.load_split(base.ROOT.parent/'data','train')
a=np.load(folder/'oof.npz');truth=a['truth'];well=a['well'];fold=a['fold'];saved=a['pred']
valid=[np.isfinite(frame[base.TS].to_numpy()).all(1) for _,frame in rec]
assert np.array_equal(truth,np.concatenate([frame[base.TS].to_numpy()[m] for (_,frame),m in zip(rec,valid)]))
train_wells=[i for i in range(80) if fold[np.flatnonzero(well==i)[0]]!=0]
builder=base.FeatureBuilder().fit([rec[i][1] for i in train_wells]);assert builder.medians_==ck['medians']
model=Network().cuda().eval();model.load_state_dict(ck['state_dict']);assert not model.training
fm=np.asarray(ck['feature_median']);fs=np.asarray(ck['feature_scale']);indices=ck['selected_feature_indices']
actions=[];utility=[]
for j,name in enumerate(base.TS):
 ct=np.asarray(ck['centers'][j]);action=np.linspace(ct[0],ct[-1],800) if j==2 else np.geomspace(max(ct[0]*.9,.001),ct[-1]*1.1,800)
 action=np.r_[base.DEFAULTS[name],action];actions.append(action)
 utility.append(torch.tensor(base.kernel_matrix(action,np.r_[base.DEFAULTS[name],ct],name).T,device='cuda',dtype=torch.float32))
rng=np.random.default_rng(60217);preds=[];ids=[];offset=0
with torch.inference_mode():
 for wi,(_,frame) in enumerate(rec):
  n=len(frame);grid=np.arange(0,n,128)+64;chosen=np.sort(rng.choice(grid,size=min(8,len(grid)),replace=False))
  x=np.clip((builder.transform(frame).enhanced[:,indices]-fm)/fs,-8,8).astype(np.float16);x=torch.tensor(x,device='cuda')
  depth=torch.tensor(frame['DEPTH'].to_numpy(np.float32),device='cuda');absolute=((depth-ck['absolute_depth_median'])/ck['absolute_depth_scale']).clamp(-8,8)
  cc=torch.tensor(chosen,device='cuda');anchor=cc.clamp_max(n-1)
  ix,exists=window_indices(cc,torch.zeros(n,device='cuda',dtype=torch.long),torch.full((n,),n,device='cuda',dtype=torch.long),anchors=anchor)
  relative=torch.tanh((depth[ix]-depth[anchor,None])/10)
  with torch.autocast('cuda',dtype=torch.bfloat16):go,raw=model(x[ix],relative,exists,absolute_depth=absolute[ix])
  mapping=np.full(n,-1,dtype=np.int64);mapping[valid[wi]]=np.arange(valid[wi].sum())+offset;offset+=valid[wi].sum()
  row=mapping[ix[:,64:192].cpu().numpy()];use=exists[:,64:192].cpu().numpy()&(row>=0);ids.append(row[use]);select=torch.tensor(use,device='cuda')
  gate=go[:,64:192][select].float().sigmoid();r=raw[:,64:192][select].float();weight=r[:,:,0].softmax(-1);mu=r[:,:,1];sd=F.softplus(r[:,:,2])+.025
  pred=np.empty((len(r),3))
  for j in range(3):
   cp=base.conditional_probability(weight[:,j],mu[:,j],sd[:,j],torch.tensor(ck['density_edges'][j],device='cuda',dtype=torch.float32))
   joint=torch.cat([gate[:,j:j+1],(1-gate[:,j:j+1])*cp],-1)
   pred[:,j]=actions[j][(joint@utility[j]).argmax(-1).cpu().numpy()]
  preds.append(pred)
  if (wi+1)%20==0:print('WELLS',wi+1,flush=True)
ids=np.concatenate(ids);pred=np.concatenate(preds);assert len(np.unique(ids))==len(ids)
assert all(torch.equal(v.cpu(),ck['state_dict'][k]) for k,v in model.state_dict().items());assert hashlib.sha256(path.read_bytes()).hexdigest()==before
summary={}
for name,m in [('train',fold[ids]!=0),('validation',fold[ids]==0)]:
 y=truth[ids[m]];pr=pred[m];d=base.dmask(y);groups={}
 for j,t in enumerate(base.TS):
  groups[t]={}
  for label,active in [('default',d[:,j]),('nondefault',~d[:,j])]:
   groups[t][label]={'rows':int(active.sum()),'score':base.metric(y[active],pr[active])[t]}
 summary[name]={'rows':int(m.sum()),'wells':len(np.unique(well[ids[m]])),'scores':base.metric(y,pr),'target_groups':groups}
m=fold[ids]==0;old=saved[ids[m]]
summary['validation_replay']={'exact_prediction_fraction':float((old==pred[m]).all(1).mean()),'saved_prediction_score_on_same_rows':base.metric(truth[ids[m]],old),'replayed_score_on_same_rows':summary['validation']['scores']}
summary.update({'scope':'8 fixed-grid windows uniformly sampled per well; descriptive balanced-well sample, not full CV; training score is in-sample','optimizer_updates':0,'state_and_checkpoint_unchanged':True,'checkpoint_sha256':before,'seconds':time.perf_counter()-start,'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/1024**2})
out=base.ROOT/'outputs/train_validation_gap';out.mkdir(exist_ok=True);(out/'report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');np.savez_compressed(out/'sample_predictions.npz',row_indices=ids,pred=pred)
print(json.dumps(summary,indent=2),flush=True)
