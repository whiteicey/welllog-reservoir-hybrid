"""Four-group, well-excluded auxiliary predictions. No old neural weights.
Each target has a private 63-64-64-25 MLP (8 mixture components).
40 wells fit, 8 choose auxiliary NLL checkpoint, 16 receive OOF features.
The original fifth is NEVER used by auxiliary fitting or checkpoint selection.
"""
from pathlib import Path
from datetime import datetime
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,math,random,signal,sys,time,traceback
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import GroupKFold
import train_invariant as base
from prepare_feature_structure_v20 import CURVES,transform
from train_dual8222_heads80 import atomic_json,atomic_torch,atomic_array,cpu_copy
from gpu_job_guard_v2 import check_slot

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/cross_target_v34_20260929'
TARGETS=('POR','PERM','SW')
STOP=False

def request_stop(*_):
    global STOP
    STOP=True

def prepare():
    start=time.perf_counter();torch.set_num_threads(2)
    if (STUDY/'auxiliary_manifest.json').exists():raise RuntimeError('Auxiliary preparation is immutable')
    cache=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
    records=base.load_split(ROOT.parent/'data','train')
    starts=np.asarray(cache['starts']);wells=np.asarray(cache['wells']);fold=np.asarray(cache['fold'])
    vtor=np.flatnonzero(cache['raw_to_valid']>=0);outer_train=np.flatnonzero(fold!=4)
    assert len(records)==80 and len(outer_train)==578891 and np.array_equal(vtor[outer_train],cache['train_raw'].numpy())
    raw_well=np.concatenate([np.full(len(fr),w,np.int16) for w,(_,fr) in enumerate(records)])
    # Only original64 target labels are stored in this auxiliary training source.
    truth=np.full((len(raw_well),3),np.nan,np.float64)
    truth[vtor[outer_train]]=cache['truth'][outer_train]
    groups=[];heldout_group=np.full(len(raw_well),-1,np.int8)
    for g,(remaining,held) in enumerate(GroupKFold(4).split(outer_train,groups=wells[outer_train])):
        candidates=outer_train[remaining];oof_valid=outer_train[held]
        inner_fit,inner_stop=next(GroupKFold(6).split(candidates,groups=wells[candidates]))
        fit_valid=candidates[inner_fit];stop_valid=candidates[inner_stop]
        fw=np.unique(wells[fit_valid]);sw=np.unique(wells[stop_valid]);hw=np.unique(wells[oof_valid])
        outer=np.unique(wells[fold==4])
        assert (len(fw),len(sw),len(hw),len(outer))==(40,8,16,16)
        assert set(fw).isdisjoint(sw) and set(hw).isdisjoint(set(fw)|set(sw))
        assert set(outer).isdisjoint(set(fw)|set(sw)|set(hw))
        fit_raw=vtor[fit_valid];stop_raw=vtor[stop_valid]
        med={}
        for c in CURVES:
            values=np.concatenate([records[int(w)][1][c].replace([np.inf,-np.inf],np.nan).to_numpy(np.float64) for w in fw])
            med[c]=float(np.nanmedian(values));assert np.isfinite(med[c])
        features=[]
        for _,fr in records:
            columns=[]
            for c in CURVES:
                v=transform(fr[c].replace([np.inf,-np.inf],np.nan).interpolate(limit_direction='both').fillna(med[c]).to_numpy(np.float64),c)
                s=pd.Series(v);columns.append(v)
                columns.extend(s.rolling(k,center=True,min_periods=1).mean().to_numpy() for k in (9,51,201))
                columns.extend(s.rolling(k,center=True,min_periods=1).std(ddof=0).fillna(0).to_numpy() for k in (9,51,201))
            features.append(np.column_stack(columns).astype(np.float32))
        x=np.concatenate(features);q=np.quantile(x[fit_raw],[.25,.5,.75],axis=0)
        center=q[1].astype(np.float32);scale=np.maximum(q[2]-q[0],1e-3).astype(np.float32)
        x=np.clip((x-center)/scale,-8,8).astype(np.float16)
        default=np.zeros((len(x),3),bool);labels=np.zeros((len(x),3),np.float32)
        usable=np.r_[fit_raw,stop_raw];default[usable]=base.dmask(truth[usable])
        target_meta=[]
        for c in range(3):
            d=base.dmask(truth[fit_raw])[:,c];values=base.TRANSFORM[c](truth[fit_raw[~d],c])
            tq=np.quantile(values,[.25,.5,.75]);yc=float(tq[1]);ys=float(max(tq[2]-tq[0],1e-3))
            yv=(values-yc)/ys
            labels[usable,c]=((base.TRANSFORM[c](truth[usable,c])-yc)/ys).astype(np.float32)
            labels[usable[default[usable,c]],c]=0
            target_meta.append(dict(center=yc,scale=ys,initial_means=np.quantile(yv,np.linspace(.05,.95,8)).tolist(),default_fraction=float(d.mean())))
        assert np.isfinite(x).all() and np.isfinite(labels).all()
        heldout_group[np.isin(raw_well,hw)]=g
        file=STUDY/f'aux_data_g{g}.npz'
        np.savez(file,x=x,labels=labels,default=default,fit_raw=fit_raw,stop_raw=stop_raw,
                 lower=cache['lower'].numpy(),upper=cache['upper'].numpy())
        entry=dict(group=g,fit_wells=fw.tolist(),stop_wells=sw.tolist(),oof_wells=hw.tolist(),outer_wells=outer.tolist(),
            fit_points=len(fit_raw),stop_points=len(stop_raw),oof_points=len(oof_valid),
            imputation_medians=med,input_median=center.tolist(),input_scale=scale.tolist(),target_meta=target_meta,
            data_file=str(file.relative_to(ROOT)),no_outer_labels=True,no_oof_labels_in_training_source=True,
            preprocessing_fitted_on_fit40_only=True,auxiliary_quality_weights='ones; no inherited Q50',
            targets_separate_models=True,input_features=63,architecture='private MLP63-64-64-25, GELU',
            checkpoint_selection='own-target complete predictive NLL on independent8 stop wells',
            summary='default_probability,mean,log_sd,q10,q50,q90 in target transform units')
        atomic_json(STUDY/f'aux_config_g{g}.json',entry);groups.append(entry)
        print('PREPARED_GROUP',g,len(fit_raw),len(stop_raw),len(oof_valid),flush=True)
    assert (heldout_group[vtor[outer_train]]>=0).all()
    np.savez(STUDY/'row_mapping.npz',heldout_group=heldout_group,raw_well=raw_well,valid_to_raw=vtor,
             train_raw=vtor[outer_train],validation_raw=vtor[cache['validation']])
    manifest=dict(created_at=datetime.now().astimezone().isoformat(),groups=groups,raw_rows=len(raw_well),
        original_train_wells=64,original_validation_wells=16,validation_labels_used=False,
        existing_models_used_for_auxiliary=False,seconds=time.perf_counter()-start,
        outer_prediction='arithmetic mean of four excluded-group auxiliary summaries; single-group OOF during training',
        caveat='Small fresh auxiliary models, not cross-fitted copies of the retained 1.29M champion; outer ensemble summary differs from single-group OOF.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(STUDY/'auxiliary_manifest.json',manifest)
    print('PREPARATION_DONE',manifest['seconds'],flush=True)

class Auxiliary(nn.Module):
    def __init__(self,c,meta,g):
        super().__init__();self.c=c
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261120+10*g+c)
            self.net=nn.Sequential(nn.Linear(63,64),nn.GELU(),nn.Linear(64,64),nn.GELU(),nn.Linear(64,25))
            nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
            with torch.no_grad():
                p=np.clip(meta['default_fraction'],1e-5,1-1e-5)
                self.net[-1].bias[0]=float(np.log(p/(1-p)))
                self.net[-1].bias[9:17]=torch.tensor(meta['initial_means'])
                self.net[-1].bias[17:25]=float(np.log(np.expm1(.975)))
    def forward(self,x):
        z=self.net(x);return z[:,0],z[:,1:].reshape(-1,3,8)

def point_loss(g,r,y,d,c):
    mu=r[:,1];sd=F.softplus(r[:,2])+.025
    if c==0:
        scale=sd/math.sqrt(3)
        logp=math.lgamma(2)-math.lgamma(1.5)-.5*math.log(3*math.pi)-scale.log()-2*torch.log1p(((y[:,None]-mu)/scale).square()/3)
    else:logp=-.5*((y[:,None]-mu)/sd).square()-sd.log()-.5*math.log(2*math.pi)
    nll=-(r[:,0].log_softmax(-1)+logp).logsumexp(-1)
    return F.binary_cross_entropy_with_logits(g,d.float(),reduction='none')+nll.masked_fill(d,0)

@torch.no_grad()
def monitor(models,x,y,d,ix):
    result=[]
    for c,m in enumerate(models):
        value=0.;m.eval()
        for rows in ix.split(16384):
            g,r=m(x[rows].float());value+=float(point_loss(g,r,y[rows,c],d[rows,c],c).sum())
        result.append(value/len(ix))
    return result

@torch.no_grad()
def descriptions(model,x,meta):
    g,r=model(x.float());w=r[:,0].softmax(-1);mu=r[:,1];sd=F.softplus(r[:,2])+.025
    mean=(w*mu).sum(-1);var=(w*(sd.square()+mu.square())).sum(-1)-mean.square()
    # Analytic mixture CDF and monotone bisection. No label-derived output grid.
    low=(mu-50*sd).amin(-1)[:,None].expand(-1,3).clone()
    high=(mu+50*sd).amax(-1)[:,None].expand(-1,3).clone()
    q=torch.tensor([.1,.5,.9],device=x.device)[None]
    for _ in range(26):
        mid=(low+high)*.5;z=(mid[:,:,None]-mu[:,None])/sd[:,None]
        if model.c==0:
            phi=torch.atan(z);cdf=.5+(phi+phi.sin()*phi.cos())/math.pi
        else:cdf=torch.special.ndtr(z)
        mass=(w[:,None]*cdf).sum(-1);low=torch.where(mass<q,mid,low);high=torch.where(mass>=q,mid,high)
    qs=(low+high)*.5*meta['scale']+meta['center']
    mean=mean*meta['scale']+meta['center'];logsd=(var.clamp_min(1e-8).sqrt()*meta['scale']).log()
    result=torch.cat((g.sigmoid()[:,None],mean[:,None],logsd[:,None],qs),-1)
    assert bool(torch.isfinite(result).all());return result

def train(g,resume=False):
    out=ROOT/'outputs'/f'pure_dl_cross_target_v34_aux{g}'
    out.mkdir(exist_ok=True);start=time.perf_counter();torch.set_num_threads(3)
    if (out/'auxiliary_report.json').exists():raise RuntimeError('Completed auxiliary models immutable')
    if (out/'last_state.pt').exists() and not resume:raise RuntimeError('Explicit resume required')
    print('ADMISSION',check_slot(min_free_gib=2.),flush=True)
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    torch.manual_seed(20261120+g);np.random.seed(20261120+g);random.seed(20261120+g)
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
    cfg=json.loads((STUDY/f'aux_config_g{g}.json').read_text(encoding='utf-8'))
    cfg.update(epochs=80,patience=18,min_epochs=10,steps=64,batch_anchors=64,supervised_points_per_anchor=128,
        lr=.0006,eta_min=.00001,weight_decay=.01,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(out/'config.json',cfg)
    with np.load(ROOT/cfg['data_file']) as z:
        data={k:torch.as_tensor(z[k],device='cuda') for k in z.files}
    x,y,d=data['x'],data['labels'],data['default'];fit=data['fit_raw'].long();stop=data['stop_raw'].long()
    fitmask=torch.zeros(len(x),device='cuda',dtype=torch.bool);fitmask[fit]=True
    assert not bool(fitmask[stop].any())
    models=nn.ModuleList([Auxiliary(c,cfg['target_meta'][c],g) for c in range(3)]).cuda()
    opts=[torch.optim.AdamW(m.parameters(),lr=.0006,weight_decay=.01) for m in models]
    sched=[torch.optim.lr_scheduler.CosineAnnealingLR(o,80*64,eta_min=.00001) for o in opts]
    rng=torch.Generator(device='cuda').manual_seed(20261120+g);epoch=0;timing=[]
    values=monitor(models,x,y,d,stop)
    states=[dict(best=v,best_epoch=0,state=cpu_copy(m.state_dict()),bad=0,done=False,history=[]) for v,m in zip(values,models)]
    if resume:
        saved=torch.load(out/'last_state.pt',map_location='cpu',weights_only=False)
        assert saved['config']==cfg;models.load_state_dict(saved['models']);states=saved['states'];epoch=saved['epoch'];timing=saved['timing']
        for o,v in zip(opts,saved['optimizers']):o.load_state_dict(v)
        for q,v in zip(sched,saved['schedulers']):q.load_state_dict(v)
        rng.set_state(saved['rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    def persist(status):
        atomic_torch(out/'last_state.pt',dict(config=cfg,models=cpu_copy(models.state_dict()),states=states,epoch=epoch,timing=timing,
            optimizers=[cpu_copy(o.state_dict()) for o in opts],schedulers=[q.state_dict() for q in sched],rng=rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
        atomic_json(out/'runtime_status.json',dict(status=status,pid=os.getpid(),group=g,epoch=epoch,
            targets=[dict(target=TARGETS[c],best_NLL=s['best'],best_epoch=s['best_epoch'],bad=s['bad'],done=s['done']) for c,s in enumerate(states)]))
    persist('running');torch.cuda.reset_peak_memory_stats();offset=torch.arange(-64,64,device='cuda')
    for ep in range(epoch+1,81):
        if STOP or all(s['done'] for s in states):break
        tick=time.perf_counter();acc=[0.,0.,0.]
        for _ in range(64):
            anchors=fit[torch.randint(len(fit),(64,),device='cuda',generator=rng)]
            ix=anchors[:,None]+offset;valid=(ix>=data['lower'][anchors,None])&(ix<data['upper'][anchors,None]);ix=ix[valid]
            # Entire windows stay in fit wells; remove unlabeled context positions.
            ix=ix[fitmask[ix]]
            xb=x[ix].float()
            for c,m in enumerate(models):
                if states[c]['done']:continue
                opts[c].zero_grad(set_to_none=True);m.train();go,raw=m(xb)
                loss=point_loss(go,raw,y[ix,c],d[ix,c],c).mean()
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Auxiliary loss is nonfinite')
                loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2.,error_if_nonfinite=True)
                opts[c].step();sched[c].step();acc[c]+=float(loss.detach())
        epoch=ep;values=monitor(models,x,y,d,stop)
        for c,(s,v,m) in enumerate(zip(states,values,models)):
            if s['done']:continue
            if v<s['best']-1e-8:s.update(best=v,best_epoch=ep,state=cpu_copy(m.state_dict()),bad=0)
            else:s['bad']+=1
            s['history'].append(dict(epoch=ep,stop_NLL=v,fit_batch_NLL=acc[c]/64))
            if (ep>=10 and s['bad']>=18) or ep==80:s['done']=True
        timing.append(dict(epoch=ep,seconds=time.perf_counter()-tick));persist('running')
        if ep%5==0:print('AUX_EPOCH',g,ep,[(s['best_epoch'],round(s['best'],6),s['done']) for s in states],flush=True)
    if STOP:persist('stopped_by_user');return
    assert all(s['done'] for s in states)
    for m,s in zip(models,states):m.load_state_dict(s['state'])
    restored=monitor(models,x,y,d,stop)
    assert max(abs(v-s['best']) for v,s in zip(restored,states))<1e-7
    all_desc=np.empty((len(x),3,6),np.float32)
    for c,m in enumerate(models):
        for lo in range(0,len(x),8192):all_desc[lo:lo+8192,c]=descriptions(m,x[lo:lo+8192],cfg['target_meta'][c]).cpu().numpy()
    atomic_array(out/'all_raw_summaries.npy',all_desc)
    atomic_torch(out/'model_best.pt',dict(config=cfg,states=[s['state'] for s in states],best_epochs=[s['best_epoch'] for s in states]))
    report=dict(group=g,config=cfg,models=[dict(target=TARGETS[c],parameters=sum(p.numel() for p in m.parameters()),best_epoch=s['best_epoch'],stopped_epoch=s['history'][-1]['epoch'],stop_NLL=s['best'],history=s['history']) for c,(m,s) in enumerate(zip(models,states))],
        seconds=time.perf_counter()-start,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        finished_at=datetime.now().astimezone().isoformat(),oof_labels_used=False,outer_labels_used=False,restored_NLL_exact=True,summary_rows=len(x))
    atomic_json(out/'auxiliary_report.json',report);persist('complete')
    print('AUX_COMPLETE',g,[(q['target'],q['best_epoch'],q['stop_NLL']) for q in report['models']],report['seconds'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--group',type=int,choices=range(4));ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    try:
        if args.prepare:prepare()
        elif args.group is not None:train(args.group,args.resume)
        else:ap.error('Explicit --prepare or --group is required')
    except BaseException as exc:
        dest=STUDY if args.prepare else ROOT/'outputs'/f'pure_dl_cross_target_v34_aux{args.group}'
        dest.mkdir(exist_ok=True,parents=True);atomic_json(dest/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=datetime.now().astimezone().isoformat()))
        raise
