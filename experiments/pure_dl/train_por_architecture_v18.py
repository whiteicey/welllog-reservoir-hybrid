"""Bounded original-fold5 POR architecture training, private networks only."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from datetime import datetime
import argparse,hashlib,json,random,signal,time,traceback
import numpy as np
import torch
from por_architecture_v18 import Network,MODES
from por_data_v18 import ROOT,STUDY,REFERENCE,load_data,Evaluator
from train_dual8222_heads80 import atomic_json,atomic_array,atomic_torch,cpu_copy
from multiscale_sequence import window_indices
from gpu_job_guard_v2 import check_slot

STOP=False
OWNED=None

def request_stop(*_):
    global STOP
    STOP=True
    print('STOP_REQUESTED: finish epoch, save; explicit resume only',flush=True)


def main():
    global OWNED
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=MODES,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();out=args.out.resolve()
    if (out/'fold_5_report.json').exists():raise RuntimeError('Completed run immutable')
    if (out/'config.json').exists() and not args.resume:raise RuntimeError('Explicit resume required')
    out.mkdir(parents=True,exist_ok=True);OWNED=out
    torch.set_num_threads(3)
    print('ADMISSION',check_slot(min_free_gib=3.),flush=True)
    atomic_json(out/'runtime_status.json',dict(status='running',mode=args.mode,pid=os.getpid(),atlas_current_trial=True))
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    started=time.perf_counter()
    random.seed(20261010);np.random.seed(20261010);torch.manual_seed(20261010)
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
    d=load_data();meta=torch.load(STUDY/'model_metadata.pt',map_location='cpu',weights_only=False)
    manifest=json.loads((STUDY/'data_manifest.json').read_text(encoding='utf-8'))
    reference=json.loads((REFERENCE/'fold_5_report.json').read_text(encoding='utf-8'))['scores']
    model=Network(args.mode,meta).cuda()
    initial_body={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n.startswith('blocks.')}
    count=sum(p.numel() for p in model.parameters())
    bias_hash=hashlib.sha256(model.output.bias.detach().cpu().numpy().tobytes()).hexdigest()
    deps=[Path(__file__),Path(__file__).with_name('por_architecture_v18.py'),Path(__file__).with_name('por_data_v18.py'),
          Path(__file__).with_name('architecture_zoo_v15.py'),Path(__file__).with_name('por_student3_sequence.py')]
    cfg=dict(mode=args.mode,target='POR',screening_fold=5,network_module='por_architecture_v18',trainable_parameters=count,
        width=model.width,latent_tokens=16 if args.mode=='latent64' else 0,attention_heads=4 if args.mode=='latent64' else 0,
        input_features=206,relative_depth=1,window_length=256,central_supervised_points=128,
        mixture_components=8,density_family='StudentT3,std=softplus(raw)+0.025,scale=std/sqrt3',
        source_network_weights_used=False,target_decoder_metadata_reused=True,old_Q_used=False,Q=1,
        seed=20261010,epochs=80,patience=18,minimum_epochs=10,steps_per_epoch=64,batch_size=64,
        lr=.0006,eta_min=1e-5,weight_decay=.01,clip_norm=2.,loss='mean(BCE(default)+nondefault StudentT mixture NLL)/3',
        decoder='original POR supports256,geomspace800+default,single coverage,batch32',
        fit_points=manifest['fit_points'],monitor_points=manifest['monitor_points'],fit_wells=manifest['fit_wells'],monitor_wells=manifest['monitor_wells'],
        initial_output_bias_sha256=bias_hash,reference_scores=reference,parameter_count_approximately_matched=True,
        statistics_manifest_sha256=hashlib.sha256((STUDY/'data_manifest.json').read_bytes()).hexdigest(),
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in deps},
        selection='original fifth POR, best checkpoint restored; development not blind test',
        deterministic_mode='warn_only; no bitwise training repeatability claim',torch_version=str(torch.__version__),remaining_folds_started=False)
    if args.resume:assert cfg==json.loads((out/'config.json').read_text(encoding='utf-8'))
    else:atomic_json(out/'config.json',cfg)
    val=Evaluator(d,d['monitor_raw'],meta)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0006,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,80*64,eta_min=1e-5)
    windows=torch.Generator(device='cuda').manual_seed(20261010)
    epoch=0;elapsed=0.;proof=None
    if args.resume:
        s=torch.load(out/'last_state.pt',map_location='cpu',weights_only=False);assert s['config']==cfg
        model.load_state_dict(s['current']);optimizer.load_state_dict(s['optimizer']);scheduler.load_state_dict(s['scheduler'])
        epoch=s['epoch'];best=s['best'];best_epoch=s['best_epoch'];best_state=s['best_state'];best_pred=s['best_pred']
        history=s['history'];bad=s['bad'];elapsed=s['seconds'];proof=s['proof']
        torch.set_rng_state(s['rng']['torch']);torch.cuda.set_rng_state_all(s['rng']['cuda']);windows.set_state(s['rng']['windows'])
        np.random.set_state(s['rng']['numpy']);random.setstate(s['rng']['python']);del s
    else:
        pred,vs=val(model)
        atomic_json(out/'initial.json',dict(validation=vs,initial_bias_sha256=bias_hash))
        atomic_array(out/'initial_pred.npy',pred)
        best=vs['POR'];best_epoch=0;best_state=cpu_copy(model.state_dict());best_pred=pred.copy();history=[];bad=0
    def persist():
        atomic_torch(out/'model_best.pt',dict(state_dict=best_state,metadata=meta,config=cfg,best_epoch=best_epoch,target_score=best))
        atomic_array(out/'channel_pred.npy',best_pred);atomic_json(out/'validation_history.json',history)
        atomic_torch(out/'last_state.pt',dict(config=cfg,current=cpu_copy(model.state_dict()),optimizer=cpu_copy(optimizer.state_dict()),
            scheduler=scheduler.state_dict(),epoch=epoch,best=best,best_epoch=best_epoch,best_state=best_state,best_pred=best_pred,
            history=history,bad=bad,proof=proof,seconds=elapsed+time.perf_counter()-started,
            rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),windows=windows.get_state(),numpy=np.random.get_state(),python=random.getstate())))
    persist();torch.cuda.reset_peak_memory_stats()
    core=torch.zeros((1,256),device='cuda',dtype=torch.bool);core[:,64:192]=True
    for ep in range(epoch+1,81):
        if STOP or (epoch>=10 and bad>=18):break
        model.train();tick=time.perf_counter();total_loss=0.
        for step in range(64):
            anchors=d['fit_raw'][torch.randint(len(d['fit_raw']),(64,),device='cuda',generator=windows)]
            ix,valid=window_indices(anchors,d['lower'],d['upper'])
            rel=torch.tanh((d['depth'][ix]-d['depth'][anchors,None])/10)
            mask=valid&core&d['fitmask'][ix];optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):o=model(d['x'][ix],rel,valid)
            loss=model.loss(o,d['labels'][ix],d['default'][ix],mask)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            if proof is None and step==1:
                grads=[p.grad.float().square().sum() for n,p in model.named_parameters() if n.startswith('blocks.') and p.grad is not None]
                norm=float(torch.stack(grads).sum().sqrt());assert norm>0 and np.isfinite(norm)
                proof=dict(backbone_gradient_norm=norm,POR_only_supervision=True,other_target_losses=False,all_parameters_trainable=True)
                atomic_json(out/'gradient_scope.json',proof)
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True)
            optimizer.step();scheduler.step();total_loss+=float(loss.detach())
        epoch=ep;pred,vs=val(model)
        if vs['POR']>best:best=vs['POR'];best_epoch=epoch;best_state=cpu_copy(model.state_dict());best_pred=pred.copy();bad=0
        else:bad+=1
        replacement=reference['TOTAL']+.30*(vs['POR']-reference['POR'])
        row=dict(epoch=epoch,validation=vs,best_target_score=best,best_epoch=best_epoch,bad_epochs=bad,
            training_loss=total_loss/64,seconds=time.perf_counter()-tick,reference_replacement_total=replacement)
        history.append(row);persist()
        atomic_json(out/'runtime_status.json',dict(status='running',mode=args.mode,pid=os.getpid(),last_completed_epoch=epoch,
            best_target_score=best,best_epoch=best_epoch,reference_replacement_total=reference['TOTAL']+.30*(best-reference['POR']),atlas_current_trial=True))
        print('EPOCH',json.dumps(dict(mode=args.mode,epoch=epoch,POR=vs['POR'],best=best,best_epoch=best_epoch,
            default_POR=vs['default_POR'],numeric_POR=vs['numeric_POR'],seconds=row['seconds'])),flush=True)
    if STOP:
        atomic_json(out/'runtime_status.json',dict(status='stopped_by_user',mode=args.mode,pid=os.getpid(),resume_available=True,auto_resume=False));return
    model.load_state_dict(best_state,strict=True);pred,vs=val(model);assert np.array_equal(pred,best_pred)
    fit=Evaluator(d,d['fit_raw'].cpu().numpy(),meta);fitpred,ts=fit(model);atomic_array(out/'fit_pred.npy',fitpred)
    change=float(sum((p.detach().cpu().double()-initial_body[n].double()).square().sum() for n,p in model.named_parameters() if n in initial_body).sqrt())
    report=dict(fold=5,screening_fold=5,target='POR',mode=args.mode,scores={'POR':vs['POR']},target_score=vs['POR'],
        validation=vs,fit=ts,best_epoch=best_epoch,stopped_epoch=epoch,trainable_parameters=count,
        reference_replacement_total=reference['TOTAL']+.30*(vs['POR']-reference['POR']),delta_reference_POR=vs['POR']-reference['POR'],
        TOTAL_scope='fixed verified PERM/SW, pending fresh POR replay and actual array assembly',
        gradient_scope=proof,selected_backbone_change_L2=change,all144665rows_restored_exactly=True,
        mechanism=model.mechanism_statistics(),seconds=elapsed+time.perf_counter()-started,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        complete_five_folds=False,remaining_folds_started=False,finished_at=datetime.now().astimezone().isoformat())
    atomic_json(out/'fold_5_report.json',report)
    atomic_json(out/'runtime_status.json',dict(status='complete',mode=args.mode,pid=os.getpid(),atlas_current_trial=False))
    print('FINAL',json.dumps({k:v for k,v in report.items() if k not in ('fit','validation')}),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OWNED is not None:
            error=dict(status='failed',pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc(),auto_resume=False)
            atomic_json(OWNED/'failure.json',error);atomic_json(OWNED/'runtime_status.json',error)
        raise
