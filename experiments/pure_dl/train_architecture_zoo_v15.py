"""Bounded architecture screen on the frozen v14 inner48/16 split.

Reuse data, objective, score, inference batch and training budget; replace the
entire backbone. No pretrained network or fixed predictor is loaded.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from datetime import datetime
import argparse,hashlib,json,random,signal,time,traceback
import numpy as np
import torch
from architecture_zoo_v15 import Network,ARCHITECTURES,DESCRIPTIONS
from train_sw_group_v14 import load_data,Evaluator,loss_terms,STUDY as DATA_STUDY
from multiscale_sequence import window_indices
from train_dual8222_heads80 import atomic_json,atomic_torch,atomic_array,cpu_copy
from gpu_job_guard_v2 import check_slot

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/architecture_zoo_v15_20260928'
CONTROL=ROOT/'outputs/pure_dl_sw_group_v14_inner_point80'
STOP=False;OWNED=None

def request_stop(*_):
    global STOP
    STOP=True
    print('STOP_REQUESTED: finish current epoch, save, no automatic restart',flush=True)


def main():
    global OWNED
    ap=argparse.ArgumentParser();ap.add_argument('--architecture',choices=ARCHITECTURES,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();out=args.out.resolve()
    if (out/'final_report.json').exists():raise RuntimeError('Completed experiment is immutable')
    if (out/'config.json').exists() and not args.resume:raise RuntimeError('Use explicit --resume for existing run')
    out.mkdir(parents=True,exist_ok=True);OWNED=out
    torch.set_num_threads(3);print('ADMISSION',check_slot(min_free_gib=3.),flush=True)
    atomic_json(out/'runtime_status.json',dict(status='running',pid=os.getpid(),architecture=args.architecture,atlas_current_trial=True))
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    started=time.perf_counter();random.seed(20261010);np.random.seed(20261010);torch.manual_seed(20261010)
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
    d=load_data('inner');manifest=json.loads((DATA_STUDY/'data_inner_manifest.json').read_text(encoding='utf-8'))
    assert len(manifest['fit_wells'])==48 and len(manifest['monitor_wells'])==16
    assert not set(manifest['outer_fifth_wells'])&(set(manifest['fit_wells'])|set(manifest['monitor_wells']))
    model=Network(args.architecture,d['centers'],float(d['target_median']),float(d['target_scale'])).cuda()
    prior=model.initialize_training_priors(dict(train_raw=d['fit_raw'],default=d['default'],labels=d['labels'],quality=torch.ones(len(d['truth']),device='cuda')))
    prior['fit_population']='inner48 fit wells only'
    initial_body={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not n.startswith('output.')}
    shared_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for k,v in model.state_dict().items() if not k.startswith('blocks.'))).hexdigest()
    cfg=dict(architecture=args.architecture,model_module='architecture_zoo_v15',target='SW',phase='inner',description=DESCRIPTIONS[args.architecture],
        trainable_parameters=sum(p.numel() for p in model.parameters()),hidden_width=64,input_dimensions=207,
        output='scalar gate +256 categorical density',pretrained_network_used=False,old_predictions_used=False,
        common_stem_readout_initial_hash=shared_hash,initialization=prior,seed=20261010,architecture_seed=20261027,
        epochs=80,patience=18,minimum_epochs=10,steps_per_epoch=64,batch_size=64,lr=6e-4,eta_min=1e-5,weight_decay=.01,clip_norm=2.,
        Q=1,learning_weights='point1',loss='original BCE+conditional two-bin CE /3',decoder='v14 unchanged801actions,max utility,single coverage,batch32',
        fit_wells=manifest['fit_wells'],monitor_wells=manifest['monitor_wells'],fit_points=manifest['fit_valid_points'],monitor_points=manifest['monitor_valid_points'],
        original_fifth_evaluated=False,complete_nestedCV=False,remaining_original_folds_started=False,
        control_reused='outputs/pure_dl_sw_group_v14_inner_point80/final_report.json',control_not_rerun=True,
        parameter_matched_to_control=False,interpretation='whole architecture comparison at common width/budget;not parameter-matched mechanism ablation',
        data_manifest_sha256=hashlib.sha256((DATA_STUDY/'data_inner_manifest.json').read_bytes()).hexdigest(),
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('architecture_zoo_v15.py'),Path(__file__).with_name('train_sw_group_v14.py'),Path(__file__).with_name('independent_sw_scratch_v10.py')]},
        deterministic_mode='warn_only;CUDA reductions may not be bitwise reproducible',torch_version=str(torch.__version__))
    if args.resume:assert json.loads((out/'config.json').read_text(encoding='utf-8'))==cfg
    else:atomic_json(out/'config.json',cfg)
    train_eval=Evaluator(d,d['fit_raw'].cpu().numpy());monitor_eval=Evaluator(d,d['monitor_raw'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=6e-4,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,80*64,eta_min=1e-5)
    windows=torch.Generator(device='cuda').manual_seed(20261010)
    epoch=0;elapsed=0.;proof=None
    if args.resume:
        s=torch.load(out/'last_state.pt',map_location='cpu',weights_only=False);assert s['config']==cfg
        model.load_state_dict(s['current']);optimizer.load_state_dict(s['optimizer']);scheduler.load_state_dict(s['scheduler'])
        epoch=s['epoch'];best=s['best'];best_epoch=s['best_epoch'];best_state=s['best_state'];best_pred=s['best_pred'];history=s['history'];bad=s['bad'];elapsed=s['seconds'];proof=s['proof']
        torch.set_rng_state(s['rng']['torch']);torch.cuda.set_rng_state_all(s['rng']['cuda']);windows.set_state(s['rng']['windows']);np.random.set_state(s['rng']['numpy']);random.setstate(s['rng']['python'])
        del s
    else:
        pred,initial=monitor_eval(model);_,initial_fit=train_eval(model)
        control_initial=json.loads((CONTROL/'initial.json').read_text(encoding='utf-8'))
        assert abs(initial['SW']-control_initial['monitor']['SW'])<1e-10
        atomic_json(out/'initial.json',dict(monitor=initial,fit=initial_fit,same_constant_initial_score_as_control=True,mechanism=model.mechanism_statistics()))
        best=initial['SW'];best_epoch=0;best_state=cpu_copy(model.state_dict());best_pred=pred.copy();history=[];bad=0
    def persist():
        atomic_torch(out/'model_best.pt',dict(state_dict=best_state,config=cfg,best_epoch=best_epoch,target_score=best,centers=d['centers'],target_median=model.target_median,target_scale=model.target_scale))
        atomic_array(out/'monitor_pred.npy',best_pred);atomic_json(out/'validation_history.json',history)
        atomic_torch(out/'last_state.pt',dict(config=cfg,current=cpu_copy(model.state_dict()),optimizer=cpu_copy(optimizer.state_dict()),scheduler=scheduler.state_dict(),
            epoch=epoch,best=best,best_epoch=best_epoch,best_state=best_state,best_pred=best_pred,history=history,bad=bad,proof=proof,seconds=elapsed+time.perf_counter()-started,
            rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),windows=windows.get_state(),numpy=np.random.get_state(),python=random.getstate())))
    persist();torch.cuda.reset_peak_memory_stats();core=torch.zeros((1,256),device='cuda',dtype=torch.bool);core[:,64:192]=True
    for ep in range(epoch+1,81):
        if STOP or (epoch>=10 and bad>=18):break
        tick=time.perf_counter();model.train();summed=0.
        for step in range(64):
            anchors=d['fit_raw'][torch.randint(len(d['fit_raw']),(64,),device='cuda',generator=windows)]
            ix,valid=window_indices(anchors,d['lower'],d['upper']);relative=torch.tanh((d['depth'][ix]-d['depth'][anchors,None])/10)
            mask=valid&core&d['fitmask'][ix];optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):output=model(d['x'][ix],relative,valid)
            loss=loss_terms(model,output,d,ix,mask,'point')
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward()
            if proof is None and step==1:
                norms={n:float(p.grad.detach().float().norm()) for n,p in model.named_parameters() if p.grad is not None}
                gn=sum(v*v for k,v in norms.items() if k.startswith('blocks.'))**.5
                assert np.isfinite(gn) and gn>0
                proof=dict(backbone_gradient_norm=gn,nonzero_backbone_tensors=sum(v>0 for k,v in norms.items() if k.startswith('blocks.')),
                    learned_offset_gradients={k:v for k,v in norms.items() if 'coordinates' in k},state_gradients={k:v for k,v in norms.items() if 'raw_tau' in k},
                    only_fit48_supervised=True,all_parameters_trainable=True)
                atomic_json(out/'gradient_scope.json',proof)
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True);optimizer.step();scheduler.step();summed+=float(loss.detach())
        epoch=ep;pred,vs=monitor_eval(model);_,ts=train_eval(model)
        if vs['SW']>best:best=vs['SW'];best_epoch=epoch;best_state=cpu_copy(model.state_dict());best_pred=pred.copy();bad=0
        else:bad+=1
        row=dict(epoch=epoch,fit=ts,monitor=vs,best_target_score=best,best_epoch=best_epoch,bad_epochs=bad,training_loss=summed/64,seconds=time.perf_counter()-tick)
        history.append(row);persist()
        atomic_json(out/'runtime_status.json',dict(status='running',pid=os.getpid(),architecture=args.architecture,last_completed_epoch=epoch,best_target_score=best,best_epoch=best_epoch,atlas_current_trial=True))
        print('EPOCH',json.dumps(dict(epoch=epoch,fit_SW=ts['SW'],monitor_SW=vs['SW'],best=best,best_epoch=best_epoch,seconds=row['seconds'])),flush=True)
    if STOP:
        atomic_json(out/'runtime_status.json',dict(status='stopped_by_user',pid=os.getpid(),resume_available=True,auto_resume=False));return
    model.load_state_dict(best_state,strict=True);pred,vs=monitor_eval(model);fitpred,ts=train_eval(model)
    assert np.array_equal(pred,best_pred);atomic_array(out/'fit_pred.npy',fitpred)
    change=float(sum((p.detach().cpu().double()-initial_body[n].double()).square().sum() for n,p in model.named_parameters() if n in initial_body).sqrt())
    report=dict(architecture=args.architecture,target='SW',score_scope='inner48fit16monitor SW,not originalfold5 orTOTAL',fit=ts,monitor=vs,target_score=vs['SW'],
        best_epoch=best_epoch,stopped_epoch=epoch,trainable_parameters=cfg['trainable_parameters'],gradient_scope=proof,selected_backbone_change_L2=change,
        mechanism=model.mechanism_statistics(),monitor_replayed_exactly=True,original_fifth_evaluated=False,complete_five_folds=False,
        seconds=elapsed+time.perf_counter()-started,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,finished_at=datetime.now().astimezone().isoformat())
    atomic_json(out/'final_report.json',report);atomic_json(out/'runtime_status.json',dict(status='complete',pid=os.getpid(),architecture=args.architecture,atlas_current_trial=False))
    print('FINAL',json.dumps(dict(architecture=args.architecture,SW=vs['SW'],best_epoch=best_epoch,stopped_epoch=epoch)),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OWNED is not None:
            err=dict(status='failed',pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc(),auto_resume=False)
            atomic_json(OWNED/'failure.json',err);atomic_json(OWNED/'runtime_status.json',err)
        raise
