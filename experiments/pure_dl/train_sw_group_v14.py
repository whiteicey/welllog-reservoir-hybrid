"""SW group-generalization experiment; one inner split, no fake fold5 scores.

Same fresh TCN and minibatches; point versus inverse-well-size learning weights.
Q is1. Initial priors are identical. Fit and monitor metrics are evaluated in
inference mode every epoch so their trajectories are directly comparable.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from datetime import datetime
import argparse,json,hashlib,random,signal,time,traceback
import numpy as np
import torch
from torch.nn import functional as F
from independent_sw_scratch_v10 import Network
from multiscale_sequence import window_indices
from train_dual8222_heads80 import atomic_json,atomic_torch,atomic_array,cpu_copy
from gpu_job_guard_v2 import check_slot
import train_invariant as base

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/sw_group_generalization_v14_20260928'
STOP=False;OWNED=None

def stop(*_):
    global STOP
    STOP=True
    print('STOP_REQUESTED: finish epoch and save; no automatic resume',flush=True)

def load_data(phase):
    with np.load(STUDY/('data_'+phase+'.npz')) as z:d={k:z[k] for k in z.files}
    for k in ['x','depth','lower','upper','labels','default','fitmask','group_weight','fit_raw']:
        d[k]=torch.from_numpy(d[k]).cuda()
    return d

class Evaluator:
    def __init__(self,data,ids):
        self.d=data;self.ids=np.asarray(ids,dtype=np.int64)
        self.local=np.full(len(data['truth']),-1,np.int64);self.local[self.ids]=np.arange(len(self.ids))
        wells=np.unique(data['wells'][self.ids]);self.centers=[];self.anchors=[]
        for j,w in enumerate(data['included_wells']):
            if w not in wells:continue
            lo,hi=data['starts'][j:j+2]
            cc=np.arange(lo,hi,128)+64
            self.centers.extend(cc);self.anchors.extend(np.minimum(cc,hi-1))
        ct=data['centers']
        self.actions=np.unique(np.r_[99.9,np.clip(np.linspace(ct[0],ct[-1],800),*base.BOUNDS['SW'])])
        self.utilities=torch.tensor(base.kernel_matrix(self.actions,np.r_[99.9,ct],'SW').T,device='cuda',dtype=torch.float32)
    @torch.no_grad()
    def __call__(self,model,retain=False):
        d=self.d;model.eval();pred=np.full(len(self.ids),np.nan,np.float64);seen=np.zeros(len(pred),np.int8)
        bces=ces=0.;numeric_n=0
        for start in range(0,len(self.centers),32):
            centers=torch.tensor(self.centers[start:start+32],device='cuda')
            anchors=torch.tensor(self.anchors[start:start+32],device='cuda')
            ix,valid=window_indices(centers,d['lower'],d['upper'],anchors=anchors)
            rel=torch.tanh((d['depth'][ix]-d['depth'][anchors,None])/10)
            with torch.autocast('cuda',dtype=torch.bfloat16):g,l=model(d['x'][ix],rel,valid)
            rawids=ix[:,64:192].cpu().numpy();locs=self.local[rawids]
            use=valid[:,64:192].cpu().numpy()&(locs>=0)
            dest=locs[use];assert np.all(seen[dest]==0) and len(np.unique(dest))==len(dest)
            seen[dest]+=1;mask=torch.from_numpy(use).cuda()
            gl=g[:,64:192][mask][:,0].float();lp=l[:,64:192][mask].float().log_softmax(-1)
            mass=torch.cat([gl.sigmoid()[:,None],(1-gl.sigmoid()[:,None])*lp.exp()],-1)
            pred[dest]=self.actions[(mass@self.utilities).argmax(-1).cpu().numpy()]
            ids=rawids[use]
            dd=torch.tensor(d['default_truth'][ids],device='cuda')
            zz=torch.tensor((np.log(d['truth'][ids])-model.target_median)/model.target_scale,device='cuda',dtype=torch.float32)
            low,high,frac=model.soft_targets(zz)
            ce=-((1-frac)*lp.gather(1,low[:,None])[:,0]+frac*lp.gather(1,high[:,None])[:,0])
            bces+=float(F.binary_cross_entropy_with_logits(gl,dd.float(),reduction='sum'))
            ces+=float(ce[~dd].sum());numeric_n+=int((~dd).sum())
        assert (seen==1).all() and np.isfinite(pred).all()
        yy=d['truth'][self.ids];dd=d['default_truth'][self.ids]
        point=100*np.maximum(0,1-np.abs(pred-yy)/(.05*(np.abs(yy)+1e-5)))
        perwell=[dict(well=int(w),n=int((d['wells'][self.ids]==w).sum()),SW=float(point[d['wells'][self.ids]==w].mean())) for w in np.unique(d['wells'][self.ids])]
        stat=dict(SW=float(point.mean()),well_macro_SW=float(np.mean([w['SW'] for w in perwell])),
            default_SW=float(point[dd].mean()),numeric_SW=float(point[~dd].mean()),default_points=int(dd.sum()),numeric_points=int((~dd).sum()),
            BCE=bces/len(self.ids),conditional_CE=ces/max(numeric_n,1),per_well=perwell)
        return pred,stat

def loss_terms(model,output,d,ix,mask,mode):
    gate,logits=output;g=gate[mask][:,0].float();lp=logits[mask].float().log_softmax(-1)
    dd=d['default'][ix][mask][:,2];zz=d['labels'][ix][mask][:,2]
    low,high,frac=model.soft_targets(zz.masked_fill(dd,0))
    ce=-((1-frac)*lp.gather(1,low[:,None])[:,0]+frac*lp.gather(1,high[:,None])[:,0])
    bce=F.binary_cross_entropy_with_logits(g,dd.float(),reduction='none')
    w=d['group_weight'][ix][mask] if mode=='well' else torch.ones_like(g)
    return (w*(bce+(~dd)*ce)).mean()/3

def main():
    global OWNED
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['inner','outer'],required=True)
    ap.add_argument('--mode',choices=['point','well'],required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--resume',action='store_true');a=ap.parse_args();out=a.out.resolve()
    if (out/'final_report.json').exists():raise RuntimeError('Completed run immutable')
    if (out/'config.json').exists() and not a.resume:raise RuntimeError('Explicit --resume required')
    OWNED=out;out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(3)
    print('ADMISSION',check_slot(min_free_gib=3.),flush=True)
    atomic_json(out/'runtime_status.json',dict(status='running',pid=os.getpid(),phase=a.phase,mode=a.mode,atlas_current_trial=True))
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    start=time.perf_counter();random.seed(20261010);np.random.seed(20261010);torch.manual_seed(20261010)
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
    d=load_data(a.phase);manifest=json.loads((STUDY/('data_'+a.phase+'_manifest.json')).read_text(encoding='utf-8'))
    model=Network('tcn',d['centers'],float(d['target_median']),float(d['target_scale'])).cuda()
    prior=model.initialize_training_priors(dict(train_raw=d['fit_raw'],default=d['default'],labels=d['labels'],quality=torch.ones(len(d['truth']),device='cuda')))
    prior['fit_population']='current '+str(len(manifest['fit_wells']))+' fit wells only'
    initial_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    cfg=dict(phase=a.phase,mode=a.mode,model_module='independent_sw_scratch_v10',architecture='tcn',target='SW',
        trainable_parameters=sum(p.numel() for p in model.parameters()),epochs=80,patience=18,minimum_epochs=10,steps_per_epoch=64,batch_size=64,
        seed=20261010,lr=6e-4,eta_min=1e-5,weight_decay=.01,clip_norm=2.,initial_state_sha256=initial_hash,initialization=prior,
        loss='mean(w*(BCE+nondefault soft2bin CE))/3',weight='1' if a.mode=='point' else 'N_fit/(number_fit_wells*n_w)',
        Q=1,old_model_or_statistics_used=False,fit_wells=manifest['fit_wells'],monitor_wells=manifest['monitor_wells'],
        selection='monitor SW point average;not training score or wellmacro',
        score_scope='inner grouped48to16 development only' if a.phase=='inner' else 'original fixed fifth',
        fold5_evaluated=a.phase=='outer',remaining_original_folds_started=False,
        deterministic_mode='warn_only',cache= 'data_'+a.phase+'.npz',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if a.resume:assert json.loads((out/'config.json').read_text())==cfg
    else:atomic_json(out/'config.json',cfg)
    train_eval=Evaluator(d,d['fit_raw'].cpu().numpy());val_eval=Evaluator(d,d['monitor_raw'])
    opt=torch.optim.AdamW(model.parameters(),lr=6e-4,weight_decay=.01);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,80*64,eta_min=1e-5)
    rng=torch.Generator(device='cuda').manual_seed(20261010)
    epoch=0;seconds=0.;proof=None
    if a.resume:
        state=torch.load(out/'last_state.pt',map_location='cpu',weights_only=False);assert state['config']==cfg
        model.load_state_dict(state['state']);opt.load_state_dict(state['optimizer']);sch.load_state_dict(state['scheduler'])
        epoch=state['epoch'];history=state['history'];best=state['best'];best_epoch=state['best_epoch'];best_state=state['best_state'];best_pred=state['best_pred'];bad=state['bad'];seconds=state['seconds'];proof=state['proof']
        torch.set_rng_state(state['rng']['torch']);torch.cuda.set_rng_state_all(state['rng']['cuda']);rng.set_state(state['rng']['windows']);np.random.set_state(state['rng']['numpy']);random.setstate(state['rng']['python'])
    else:
        pred,initial=val_eval(model);_,fitinitial=train_eval(model)
        atomic_json(out/'initial.json',dict(monitor=initial,fit=fitinitial,initial_state_sha256=initial_hash))
        best=initial['SW'];best_epoch=0;best_state=cpu_copy(model.state_dict());best_pred=pred.copy();bad=0;history=[]
    def persist():
        atomic_torch(out/'model_best.pt',dict(state_dict=best_state,config=cfg,best_epoch=best_epoch,target_score=best,
            centers=d['centers'],target_median=model.target_median,target_scale=model.target_scale))
        atomic_array(out/'monitor_pred.npy',best_pred);atomic_json(out/'validation_history.json',history)
        atomic_torch(out/'last_state.pt',dict(config=cfg,state=cpu_copy(model.state_dict()),optimizer=cpu_copy(opt.state_dict()),scheduler=sch.state_dict(),epoch=epoch,
            history=history,best=best,best_epoch=best_epoch,best_state=best_state,best_pred=best_pred,bad=bad,proof=proof,seconds=seconds+time.perf_counter()-start,
            rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),windows=rng.get_state(),numpy=np.random.get_state(),python=random.getstate())))
    persist();torch.cuda.reset_peak_memory_stats();core=torch.zeros((1,256),device='cuda',dtype=torch.bool);core[:,64:192]=True
    for ep in range(epoch+1,81):
        if STOP or (epoch>=10 and bad>=18):break
        model.train();tick=time.perf_counter();trainloss=0.
        for step in range(64):
            anchors=d['fit_raw'][torch.randint(len(d['fit_raw']),(64,),device='cuda',generator=rng)]
            ix,valid=window_indices(anchors,d['lower'],d['upper']);rel=torch.tanh((d['depth'][ix]-d['depth'][anchors,None])/10)
            mask=valid&core&d['fitmask'][ix];opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):output=model(d['x'][ix],rel,valid)
            loss=loss_terms(model,output,d,ix,mask,a.mode)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            if proof is None and step==1:
                gradients=[p.grad.detach().float().norm() for n,p in model.named_parameters() if not n.startswith('output.') and p.grad is not None]
                gn=float(torch.stack(gradients).square().sum().sqrt());assert gn>0 and np.isfinite(gn)
                proof=dict(backbone_gradient_norm=gn,fit_well_count=len(manifest['fit_wells']),only_fit_labels_supervised=True)
                atomic_json(out/'gradient_scope.json',proof)
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True);opt.step();sch.step();trainloss+=float(loss.detach())
        epoch=ep;pred,vs=val_eval(model);_,ts=train_eval(model)
        if vs['SW']>best:best=vs['SW'];best_epoch=epoch;best_state=cpu_copy(model.state_dict());best_pred=pred.copy();bad=0
        else:bad+=1
        summary=dict(epoch=epoch,fit=ts,monitor=vs,best_target_score=best,best_epoch=best_epoch,bad_epochs=bad,training_loss=trainloss/64,
                     gap_SW=ts['SW']-vs['SW'],seconds=time.perf_counter()-tick)
        history.append(summary);persist()
        atomic_json(out/'runtime_status.json',dict(status='running',pid=os.getpid(),phase=a.phase,mode=a.mode,last_completed_epoch=epoch,best_target_score=best,best_epoch=best_epoch,atlas_current_trial=True))
        print('EPOCH',json.dumps({k:v for k,v in summary.items() if k not in ['fit','monitor']}|dict(fit_SW=ts['SW'],monitor_SW=vs['SW'])),flush=True)
    if STOP:
        atomic_json(out/'runtime_status.json',dict(status='stopped_by_user',pid=os.getpid(),resume_available=True,auto_resume=False));return
    model.load_state_dict(best_state);pred,vs=val_eval(model);fitpred,ts=train_eval(model)
    assert np.array_equal(pred,best_pred);atomic_array(out/'fit_pred.npy',fitpred)
    report=dict(phase=a.phase,mode=a.mode,target='SW',score_scope=cfg['score_scope'],fit=ts,monitor=vs,target_score=vs['SW'],gap_SW=ts['SW']-vs['SW'],
        best_epoch=best_epoch,stopped_epoch=epoch,trainable_parameters=cfg['trainable_parameters'],gradient_scope=proof,monitor_replayed_exactly=True,
        new_preprocessing_fit_only=True,old_Q_used=False,original_fifth_evaluated=a.phase=='outer',complete_nestedCV=False,complete_five_folds=False,
        seconds=seconds+time.perf_counter()-start,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,finished_at=datetime.now().astimezone().isoformat())
    atomic_json(out/'final_report.json',report)
    atomic_json(out/'runtime_status.json',dict(status='complete',pid=os.getpid(),phase=a.phase,mode=a.mode,atlas_current_trial=False))
    print('FINAL',json.dumps(report),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OWNED:
            failure=dict(status='failed',pid=os.getpid(),error=repr(e),traceback=traceback.format_exc(),auto_resume=False)
            atomic_json(OWNED/'failure.json',failure);atomic_json(OWNED/'runtime_status.json',failure)
        raise
