"""One preselected stage on one original fold, own target loss/best state."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import hashlib
import json
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from fullcv_context_v97 import ROOT, STUDY, RECIPE, require_training_context, sha
from fullcv_models_v97 import STAGES, fold_dir, stage_dir, root_path, make_model, export_selected
from fullcv_resume_v97 import atomic_save, cpu_state, rng_state, restore_rng
from multiscale_sequence import window_indices
from por_t3_patch_loss import probability
from train_scaling_v21 import channel_objective
from gpu_job_guard_v2 import check_slot
import train_invariant as base


class Evaluator:
    """Public RMSE evaluator using posterior means, not the internal tolerance action."""
    def __init__(self,data,master,c,length):
        self.data=data;self.c=c;self.length=length;self.batch=16 if length>256 else 32
        self.ct=torch.as_tensor(master['centers'][c],device='cuda',dtype=torch.float32)
        self.edges=torch.tensor(master['density_edges'][c],device='cuda',dtype=torch.float32)
        ct=np.asarray(master['centers'][c]);name=base.TS[c]
        if c==1:
            self.actions=np.unique(np.r_[1.0,np.linspace(ct[0],ct[-1],100)])
            support=np.r_[1.0,ct]
        else:
            self.actions=np.unique(np.linspace(ct[0],ct[-1],100))
            support=ct
        self.utilities=torch.tensor(base.kernel_matrix(self.actions,support,name).T,device='cuda',dtype=torch.float32)
        self.local=np.full(len(data['truth']),-1,np.int64);self.local[data['validation']]=np.arange(len(data['validation']))
        self.centers=[];self.anchors=[]
        for wi in data['population_rows']['heldout_wells']:
            cc=np.arange(data['starts'][wi],data['starts'][wi+1],128)+64
            self.centers.extend(cc);self.anchors.extend(np.minimum(cc,data['starts'][wi+1]-1))

    @torch.no_grad()
    def __call__(self,model):
        d=self.data;model.eval();lo=self.length//2-64;hi=lo+128
        result=np.full(len(d['validation']),np.nan,np.float64);coverage=np.zeros(len(result),np.int8)
        for begin in range(0,len(self.centers),self.batch):
            cc=torch.tensor(self.centers[begin:begin+self.batch],device='cuda');anchor=torch.tensor(self.anchors[begin:begin+self.batch],device='cuda')
            ix,valid=window_indices(cc,d['lower'],d['upper'],length=self.length,anchors=anchor)
            rel=torch.tanh((d['depth'][ix]-d['depth'][anchor,None])/10)
            context=d['well_bank'][d['row_well'][anchor]]
            with torch.autocast('cuda',dtype=torch.bfloat16):go,raw=model(d['x'][ix],rel,valid,absolute_depth=d['absolute'][ix],well_inputs=context)
            rows=d['raw_to_valid'][ix[:,lo:hi].cpu().numpy()];use=valid[:,lo:hi].cpu().numpy()&(rows>=0);dest=self.local[rows[use]]
            assert (dest>=0).all() and (coverage[dest]==0).all() and len(np.unique(dest))==len(dest)
            coverage[dest]+=1;take=torch.as_tensor(use,device='cuda')
            g=go[:,lo:hi][take][:,0].float().sigmoid();r=raw[:,lo:hi][take][:,0].float()
            cp=probability(self.c,r[:,0].softmax(-1),r[:,1],F.softplus(r[:,2])+.025,self.edges)
            if self.c==1:
                value=g+(1-g)*(cp@self.ct)
            else:
                value=cp@self.ct
            result[dest]=value.cpu().numpy()
        assert (coverage==1).all() and np.isfinite(result).all()
        y=d['truth'][d['validation']].copy();p=y.copy();p[:,self.c]=result
        score=base.metric(y,p)[base.TS[self.c]]
        return result,float(score)


def frozen_signature(model):
    trainable={n for n,p in model.named_parameters() if p.requires_grad}
    digest=hashlib.sha256()
    for n,v in sorted(model.state_dict().items()):
        if n in trainable:continue
        digest.update(n.encode());digest.update(v.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,choices=[1,2,3,4],required=True);ap.add_argument('--stage',choices=[s[0] for s in STAGES],required=True);ap.add_argument('--resume',action='store_true');ap.add_argument('--test',action='store_true');ap.add_argument('--eval-only',action='store_true');a=ap.parse_args()
    if a.eval_only:a.resume=True
    f=a.fold;stage=a.stage;out=stage_dir(f,stage);out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists() and not a.eval_only:raise RuntimeError('Completed stage is immutable')
    auth=json.loads((STUDY/'authorization.json').read_text(encoding='utf-8'));assert auth['remaining_folds_authorized'] and auth['recipe_sha256']==sha(RECIPE)
    torch.set_num_threads(3);print('GPU_ADMISSION',check_slot(min_free_gib=3.),flush=True)
    seed=20261006+f-1;torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
    cache=torch.load(fold_dir(f)/'data.pt',map_location='cpu',weights_only=False);assert cache['root_sha256']==sha(root_path(f))
    data={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in cache['data'].items()}
    metadata={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in cache['metadata'].items()};del cache
    require_training_context(data);master=torch.load(root_path(f),map_location='cpu',weights_only=False)
    spec=next(s for s in STAGES if s[0]==stage);_,target,mode,lr,length,sam,alpha=spec;c={'POR':0,'PERM':2,'SW':1}[target]
    if target in ('POR','PERM'):
        with np.load(fold_dir(f)/'amplitude.npz') as z:side=torch.as_tensor(z['amplitude'],device='cuda')
        if stage not in ('por_density','perm_gate','por_mean','por_point'):data['x']=torch.cat((data['x'],side),-1)
        del side
    model,spec,trace=make_model(f,stage,master,metadata);model.cuda()
    initialization=model.model.initialize_priors(data) if hasattr(model.model,'initialize_priors') else None
    evaluator=Evaluator(data,master,c,length)
    if target=='SW' and stage in ('sw_utility','sw_sam','sw_density'):
        from sw_public_objective import DecisionObjective
        criterion=DecisionObjective(evaluator,metadata,'a100').cuda();criterion.alpha=alpha
        loss_fn=lambda output,ix,mask:criterion(output,data['labels'][ix],data['default'][ix],mask,data['quality'][ix])[0]
    else:loss_fn=lambda output,ix,mask:channel_objective(output,data['labels'][ix],data['default'][ix],mask,data['quality'][ix],c)[0]
    if stage in ('por_coadapt','por_anchored'):groups=model.model.trainable_groups(mode)
    else:groups=[dict(params=[p for p in model.parameters() if p.requires_grad],lr=lr)]
    params=[p for group in groups for p in group['params']]
    fixed=frozen_signature(model)
    cfg=dict(original_fold=f,target=target,channel=c,stage=stage,mode=mode,seed=seed,epochs=24,patience=4,minimum_epochs=6,batch_size=32,steps=32,lr=lr,
        eta_min=1e-5,weight_decay=.01,clip_norm=2.,sam_rho=.1 if sam else 0.,utility_weight=alpha,window_length=length,
        training_rows=len(data['train_raw']),validation_rows=len(data['validation']),quality=data['qinfo'],initialization=initialization,
        dependency_trace=trace,source_recipe_sha256=sha(RECIPE),fixed_signature=fixed,trainable_parameters=sum(p.numel() for p in params),
        input_features=int(data['x'].shape[-1]),source_hashes={n:sha(Path(__file__).with_name(n)) for n in ['train_stage_fullcv_v97.py','fullcv_models_v97.py','fullcv_resume_v97.py']})
    if (out/'config.json').exists():
        assert a.resume and json.loads((out/'config.json').read_text(encoding='utf-8'))==cfg
    else:(out/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    optimizer=torch.optim.AdamW(groups,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,24*32,eta_min=1e-5)
    windows=torch.Generator(device='cuda').manual_seed(seed);started=time.perf_counter();elapsed=0.;peak=0.;epoch=0;bad=0;history=[]
    last=out/'last_state.pt'
    if a.resume:
        saved=torch.load(last,map_location='cpu',weights_only=False);assert saved['config']==cfg
        model.load_state_dict(saved['current']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        best=saved['best'];best_epoch=saved['best_epoch'];best_state=saved['best_state'];best_pred=saved['best_pred'];bad=saved['bad'];epoch=saved['epoch'];history=saved['history'];elapsed=saved['seconds'];peak=saved['peak']
        restore_rng(saved['rng']);windows.set_state(saved['windows']);del saved
    else:
        best_pred,best=evaluator(model);best_epoch=0;best_state=cpu_state(model)
    def persist():
        nonlocal peak
        peak=max(peak,torch.cuda.max_memory_allocated()/1024**2)
        atomic_save(dict(config=cfg,current=cpu_state(model),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            epoch=epoch,bad=bad,best=best,best_epoch=best_epoch,best_state=best_state,best_pred=best_pred,history=history,
            rng=rng_state(),windows=windows.get_state(),seconds=elapsed+time.perf_counter()-started,peak=peak),last)
    persist();lo=length//2-64;hi=lo+128;core=torch.zeros((1,length),device='cuda',dtype=torch.bool);core[:,lo:hi]=True
    print('INITIAL',json.dumps(dict(fold=f,stage=stage,target_score=best,best_epoch=best_epoch)),flush=True)
    while not a.eval_only and epoch<24 and not (epoch>=6 and bad>=4):
        tick=time.perf_counter();model.train();total=0.
        for step in range(32):
            anchors=data['train_raw'][torch.randint(len(data['train_raw']),(32,),device='cuda',generator=windows)]
            ix,valid=window_indices(anchors,data['lower'],data['upper'],length=length);rel=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
            mask=valid&core&data['labeled'][ix];assert bool(data['labeled'][anchors].all())
            kwargs=dict(absolute_depth=data['absolute'][ix],well_inputs=data['well_bank'][data['row_well'][anchors]])
            optimizer.zero_grad(set_to_none=True);before=(torch.get_rng_state(),torch.cuda.get_rng_state_all())
            with torch.autocast('cuda',dtype=torch.bfloat16):output=model(data['x'][ix],rel,valid,**kwargs)
            loss=loss_fn(output,ix,mask);assert bool(torch.isfinite(loss));loss.backward()
            if sam:
                after=(torch.get_rng_state(),torch.cuda.get_rng_state_all());active=[p for p in params if p.grad is not None]
                norm=torch.stack([p.grad.detach().float().square().sum() for p in active]).sum().sqrt();assert bool(torch.isfinite(norm))
                backups=[(p,p.detach().clone()) for p in active]
                try:
                    with torch.no_grad():
                        for p,_ in backups:p.add_(p.grad*(.1/(norm+1e-12)))
                    optimizer.zero_grad(set_to_none=True);torch.set_rng_state(before[0]);torch.cuda.set_rng_state_all(before[1])
                    with torch.autocast('cuda',dtype=torch.bfloat16):output=model(data['x'][ix],rel,valid,**kwargs)
                    loss=loss_fn(output,ix,mask);assert bool(torch.isfinite(loss));loss.backward()
                    assert torch.equal(torch.get_rng_state(),after[0]) and all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),after[1]))
                finally:
                    with torch.no_grad():
                        for p,old in backups:p.copy_(old)
                    torch.set_rng_state(after[0]);torch.cuda.set_rng_state_all(after[1])
            for group in groups:torch.nn.utils.clip_grad_norm_(group['params'],2.,error_if_nonfinite=True)
            optimizer.step();scheduler.step();total+=float(loss.detach())
        epoch+=1;pred,score=evaluator(model)
        if score>best:best=score;best_epoch=epoch;best_state=cpu_state(model);best_pred=pred.copy();bad=0
        else:bad+=1
        history.append(dict(epoch=epoch,target_score=score,best=best,best_epoch=best_epoch,bad=bad,loss=total/32,seconds=time.perf_counter()-tick))
        assert frozen_signature(model)==fixed;persist()
        (out/'validation_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print('EPOCH',json.dumps(history[-1]|dict(fold=f,stage=stage,peak_MiB=peak)),flush=True)
    model.load_state_dict(best_state);pred,score=evaluator(model);assert np.array_equal(pred,best_pred) and abs(score-best)<1e-9
    assert frozen_signature(model)==fixed
    test_score=None
    if a.test:
        test_cache=torch.load(fold_dir(f).with_name(f'fold_{f}_test')/'data.pt',map_location='cpu',weights_only=False)
        test_data={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in test_cache['data'].items()}
        test_metadata={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in test_cache['metadata'].items()}
        if target in ('POR','PERM'):
            with np.load(fold_dir(f).with_name(f'fold_{f}_test')/'amplitude.npz') as z:
                side=torch.as_tensor(z['amplitude'],device='cuda')
            if stage not in ('por_density','perm_gate','por_mean','por_point'):
                test_data['x']=torch.cat((test_data['x'],side),-1)
        test_eval=Evaluator(test_data,master,c,length)
        test_pred,test_score=test_eval(model)
        np.save(out/'test_pred.npy',test_pred)
    atomic_save(export_selected(model,f,stage,cfg,best_epoch,best),out/'model_best.pt');np.save(out/'channel_pred.npy',pred)
    report=dict(fold=f,stage=stage,target=target,target_score=score,best_epoch=best_epoch,stopped_epoch=epoch,
        best_predictions_reproduced_exactly=True,validation_rows=len(pred),training_rows=len(data['train_raw']),source_sha256=sha(root_path(f)),
        seconds=elapsed+time.perf_counter()-started,peak_allocated_MiB=peak,complete_five_folds=False,checkpoint_sha256=sha(out/'model_best.pt'))
    if a.test:
        report['test_target_rmse']=-test_score
        report['test_evaluated_after_best_state']=True
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('STAGE_FINAL',json.dumps(report),flush=True)


from pathlib import Path
if __name__=='__main__':main()
