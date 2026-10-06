"""Two bounded original-fold5 architecture candidates, trained from scratch.

V15 latent architecture is unchanged. A fresh TCN is the same-protocol control.
Three-target scores during training use fixed, previously verified POR/PERM;
final disk replay and actual assembly are a separate explicit verification.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
from datetime import datetime
import argparse, hashlib, json, random, signal, time, traceback
import numpy as np
import torch
from independent_sw_scratch_v10 import Network as TCN
from architecture_zoo_v15 import Network as ArchitectureNetwork
from train_sw_group_v14 import Evaluator, loss_terms
from multiscale_sequence import window_indices
from train_dual8222_heads80 import atomic_json, atomic_torch, atomic_array, cpu_copy
from gpu_job_guard_v2 import check_slot

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT/'results/deep_learning/latent_fold5_v16_20260928'
REFERENCE = ROOT/'outputs/pure_dl_independent_v8_selected_fold5'
MODES = ('tcn', 'latent')
STOP = False
OWNED = None


def load_outer_data():
    with np.load(STUDY/'data_outer.npz') as data:
        d = {k: data[k] for k in data.files}
    for key in ('x','depth','lower','upper','labels','default','fitmask','group_weight','fit_raw'):
        d[key] = torch.from_numpy(d[key]).cuda()
    assert len(d['fit_raw']) == 578891 and len(d['monitor_raw']) == 144665
    assert not bool(d['fitmask'][torch.as_tensor(d['monitor_raw'],device='cuda')].any())
    return d


def make_model(mode, centers, median, scale):
    if mode == 'tcn':
        return TCN('tcn', centers, median, scale)
    if mode == 'latent':
        return ArchitectureNetwork('latent', centers, median, scale)
    raise ValueError(mode)


def stop_request(*_):
    global STOP
    STOP = True
    print('STOP_REQUESTED: finish epoch and save, no automatic restart', flush=True)


def main():
    global OWNED
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=MODES, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args(); out = args.out.resolve()
    if (out/'fold_5_report.json').exists(): raise RuntimeError('Completed run immutable')
    if (out/'config.json').exists() and not args.resume: raise RuntimeError('Explicit --resume required')
    out.mkdir(parents=True, exist_ok=True); OWNED = out
    torch.set_num_threads(3)
    print('ADMISSION', check_slot(min_free_gib=3.0), flush=True)
    atomic_json(out/'runtime_status.json', dict(status='running',pid=os.getpid(),mode=args.mode,atlas_current_trial=True))
    signal.signal(signal.SIGINT,stop_request); signal.signal(signal.SIGTERM,stop_request)
    started = time.perf_counter()
    random.seed(20261010); np.random.seed(20261010); torch.manual_seed(20261010)
    torch.use_deterministic_algorithms(True,warn_only=True); torch.backends.cudnn.benchmark=False
    data = load_outer_data()
    manifest = json.loads((STUDY/'data_outer_manifest.json').read_text(encoding='utf-8'))
    assert manifest['original_validation_row_order_verified'] and not manifest['outer_fifth_used_for_fitting']
    model = make_model(args.mode,data['centers'],float(data['target_median']),float(data['target_scale'])).cuda()
    initialization = model.initialize_training_priors(dict(train_raw=data['fit_raw'],labels=data['labels'],
        default=data['default'],quality=torch.ones(len(data['truth']),device='cuda')))
    initial_body = {n:p.detach().cpu().clone() for n,p in model.named_parameters() if not n.startswith('output.')}
    common_hash = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for n,v in model.state_dict().items() if not n.startswith('blocks.'))).hexdigest()
    reference = json.loads((REFERENCE/'fold_5_report.json').read_text(encoding='utf-8'))['scores']
    cfg = dict(mode=args.mode,target='SW',screening_fold=5,phase='original64to16',model_module='train_latent_fold5_v16',
        network_module='architecture_zoo_v15' if args.mode=='latent' else 'independent_sw_scratch_v10',
        trainable_parameters=sum(p.numel() for p in model.parameters()),hidden_width=64,
        input_features=206,relative_depth=1,window_length=256,central_supervised_points=128,
        latent_count=16 if args.mode=='latent' else None,attention_heads=4 if args.mode=='latent' else None,
        pretrained_network_used=False,inner_checkpoint_used=False,only_SW_labels_used=True,
        common_stem_output_initial_hash=common_hash,initialization=initialization,
        seed=20261010,epochs=80,patience=18,minimum_epochs=10,steps_per_epoch=64,batch_size=64,
        lr=6e-4,eta_min=1e-5,weight_decay=.01,clip_norm=2.,Q=1,learning_weight='point1',
        loss='BCE(default)+nondefault soft2bin CE,mean/3',decoder='same physical256support,800linear actions+default,singlecoverage,batch32',
        fit_wells=manifest['fit_wells'],monitor_wells=manifest['monitor_wells'],fit_points=578891,monitor_points=144665,
        reference_scores=reference,selection='original fifth SW; best restored, no fixed-epoch refit',
        data_manifest_sha256=hashlib.sha256((STUDY/'data_outer_manifest.json').read_bytes()).hexdigest(),
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),
            Path(__file__).with_name('architecture_zoo_v15.py'),Path(__file__).with_name('independent_sw_scratch_v10.py'),
            Path(__file__).with_name('train_sw_group_v14.py')]},
        remaining_original_folds_started=False,parameter_matched_architecture_comparison=False,
        deterministic_mode='warn_only; bitwise training reproducibility not claimed',torch_version=str(torch.__version__))
    if args.resume: assert json.loads((out/'config.json').read_text(encoding='utf-8')) == cfg
    else: atomic_json(out/'config.json',cfg)
    train_eval = Evaluator(data,data['fit_raw'].cpu().numpy())
    val_eval = Evaluator(data,data['monitor_raw'])
    optimizer = torch.optim.AdamW(model.parameters(),lr=6e-4,weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,80*64,eta_min=1e-5)
    windows = torch.Generator(device='cuda').manual_seed(20261010)
    epoch=0; elapsed=0.; proof=None
    if args.resume:
        s = torch.load(out/'last_state.pt',map_location='cpu',weights_only=False); assert s['config']==cfg
        model.load_state_dict(s['current']); optimizer.load_state_dict(s['optimizer']); scheduler.load_state_dict(s['scheduler'])
        epoch=s['epoch']; best=s['best']; best_epoch=s['best_epoch']; best_state=s['best_state']; best_pred=s['best_pred']
        history=s['history']; bad=s['bad']; elapsed=s['seconds']; proof=s['proof']
        torch.set_rng_state(s['rng']['torch']); torch.cuda.set_rng_state_all(s['rng']['cuda']); windows.set_state(s['rng']['windows'])
        np.random.set_state(s['rng']['numpy']); random.setstate(s['rng']['python']); del s
    else:
        pred,initial = val_eval(model)
        atomic_json(out/'initial.json',dict(validation=initial,common_initial_hash=common_hash))
        best=initial['SW']; best_epoch=0; best_state=cpu_copy(model.state_dict()); best_pred=pred.copy(); history=[]; bad=0
    def persist():
        atomic_torch(out/'model_best.pt',dict(state_dict=best_state,config=cfg,best_epoch=best_epoch,target_score=best,
            centers=data['centers'],target_median=model.target_median,target_scale=model.target_scale))
        atomic_array(out/'channel_pred.npy',best_pred); atomic_json(out/'validation_history.json',history)
        atomic_torch(out/'last_state.pt',dict(config=cfg,current=cpu_copy(model.state_dict()),optimizer=cpu_copy(optimizer.state_dict()),
            scheduler=scheduler.state_dict(),epoch=epoch,best=best,best_epoch=best_epoch,best_state=best_state,best_pred=best_pred,
            history=history,bad=bad,proof=proof,seconds=elapsed+time.perf_counter()-started,
            rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),windows=windows.get_state(),
                     numpy=np.random.get_state(),python=random.getstate())))
    persist(); torch.cuda.reset_peak_memory_stats()
    core=torch.zeros((1,256),device='cuda',dtype=torch.bool); core[:,64:192]=True
    for ep in range(epoch+1,81):
        if STOP or (epoch>=10 and bad>=18): break
        model.train(); tick=time.perf_counter(); total_loss=0.
        for step in range(64):
            anchors=data['fit_raw'][torch.randint(len(data['fit_raw']),(64,),device='cuda',generator=windows)]
            ix,valid=window_indices(anchors,data['lower'],data['upper'])
            relative=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
            mask=valid&core&data['fitmask'][ix]; optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16): output=model(data['x'][ix],relative,valid)
            loss=loss_terms(model,output,data,ix,mask,'point')
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite loss')
            loss.backward()
            if proof is None and step==1:
                gradients=[p.grad.detach().float().square().sum() for n,p in model.named_parameters() if n.startswith('blocks.') and p.grad is not None]
                norm=float(torch.stack(gradients).sum().sqrt()); assert norm>0 and np.isfinite(norm)
                proof=dict(backbone_gradient_norm=norm,only_original64_fit_labels=True,private_SW=True,other_target_losses=False)
                atomic_json(out/'gradient_scope.json',proof)
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); total_loss+=float(loss.detach())
        epoch=ep; pred,vs=val_eval(model); _,ts=train_eval(model)
        if vs['SW']>best:
            best=vs['SW'];best_epoch=epoch;best_state=cpu_copy(model.state_dict());best_pred=pred.copy();bad=0
        else: bad+=1
        replacement=reference['TOTAL']+.35*(vs['SW']-reference['SW'])
        row=dict(epoch=epoch,fit=ts,validation=vs,best_target_score=best,best_epoch=best_epoch,bad_epochs=bad,
            reference_replacement_total=replacement,training_loss=total_loss/64,seconds=time.perf_counter()-tick)
        history.append(row); persist()
        atomic_json(out/'runtime_status.json',dict(status='running',pid=os.getpid(),mode=args.mode,last_completed_epoch=epoch,
            best_target_score=best,best_epoch=best_epoch,reference_replacement_total=reference['TOTAL']+.35*(best-reference['SW']),atlas_current_trial=True))
        print('EPOCH',json.dumps(dict(mode=args.mode,epoch=epoch,fit_SW=ts['SW'],validation_SW=vs['SW'],best=best,
            replacement_TOTAL=replacement,best_epoch=best_epoch,seconds=row['seconds'])),flush=True)
    if STOP:
        atomic_json(out/'runtime_status.json',dict(status='stopped_by_user',pid=os.getpid(),mode=args.mode,resume_available=True,auto_resume=False)); return
    model.load_state_dict(best_state,strict=True); pred,vs=val_eval(model); fitpred,ts=train_eval(model)
    assert np.array_equal(pred,best_pred); atomic_array(out/'fit_pred.npy',fitpred)
    change=float(sum((p.detach().cpu().double()-initial_body[n].double()).square().sum() for n,p in model.named_parameters() if n in initial_body).sqrt())
    report=dict(fold=5,screening_fold=5,target='SW',architecture=args.mode,scores={'SW':vs['SW']},target_score=vs['SW'],
        fit=ts,validation=vs,best_epoch=best_epoch,stopped_epoch=epoch,trainable_parameters=cfg['trainable_parameters'],
        reference_replacement_total=reference['TOTAL']+.35*(vs['SW']-reference['SW']),
        TOTAL_scope='fixed previously verified POR/PERM, new SW prediction; actual fresh assembly performed separately',
        delta_old_SW=vs['SW']-reference['SW'],source='original64 training,random initialization',
        gradient_scope=proof,selected_backbone_change_L2=change,all144665rows_restored_exactly=True,
        complete_five_folds=False,remaining_folds_started=False,seconds=elapsed+time.perf_counter()-started,
        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,finished_at=datetime.now().astimezone().isoformat())
    atomic_json(out/'fold_5_report.json',report)
    atomic_json(out/'runtime_status.json',dict(status='complete',pid=os.getpid(),mode=args.mode,atlas_current_trial=False))
    print('FINAL',json.dumps({k:v for k,v in report.items() if k not in ('fit','validation')}),flush=True)


if __name__=='__main__':
    try: main()
    except BaseException as exc:
        if OWNED is not None:
            error=dict(status='failed',pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc(),auto_resume=False)
            atomic_json(OWNED/'failure.json',error);atomic_json(OWNED/'runtime_status.json',error)
        raise
