"""POR location-readout comparison from the verified 80.268735 recipe.

Three predeclared modes; launch explicitly, at most two simultaneously. No queue,
no extra folds, no changes to prior checkpoints. Optimizer/scheduler restart is
part of this NEW matched study, not a claim of resuming the previous trajectory.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import signal
import time
import traceback
import numpy as np
import torch
from train_dual8222_heads80 import (SourceEvaluator,prepare_data,atomic_json,atomic_torch,
                                   atomic_array,cpu_copy,SOURCE,SOURCE_DIR)
from paired_local_context_router_sequence import Network as Original,retained_indices,calibration_indices
from dual8222_location_readout_v5 import Network,MODES
from frozen_companion_fused_loss import channel_loss
from multiscale_sequence import window_indices
from gpu_job_guard_v2 import check_slot
from fold5_screening_805 import THRESHOLD

ROOT=Path(__file__).resolve().parents[2]
SELECTION=ROOT/'outputs/pure_dl_dual8222_v4_por_mean_only80_fold5'
STOP=False
OWNED=None


def request_stop(_sig,_frame):
    global STOP
    STOP=True
    print('STOP_REQUESTED: save current complete epoch, no automatic resume',flush=True)


def main():
    global OWNED
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--mode',choices=list(MODES),required=True)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();out=args.out.resolve()
    if (out/'fold_5_report.json').exists():raise RuntimeError('Completed run is immutable')
    if (out/'config.json').exists() and not args.resume:raise RuntimeError('Existing run: explicit --resume required')
    if args.resume and not (out/'last_state.pt').exists():raise RuntimeError('No last-state checkpoint')
    out.mkdir(parents=True,exist_ok=True);OWNED=out
    torch.set_num_threads(3)
    admission=check_slot(min_free_gib=3.)
    print('GPU_ADMISSION',json.dumps(admission),flush=True)
    atomic_json(out/'runtime_status.json',{'status':'running','pid':os.getpid(),'mode':args.mode,'atlas_current_trial':True})
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    start=time.perf_counter()
    torch.manual_seed(20261010);np.random.seed(20261010);random.seed(20261010)
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    master=torch.load(SOURCE,map_location='cpu',weights_only=False)
    selected_path=SELECTION/'selected_heads.pt'
    selected=torch.load(selected_path,map_location='cpu',weights_only=False)
    baseline_report=json.loads((SELECTION/'fold_5_report.json').read_text(encoding='utf-8'))
    assert baseline_report['assembled_heads_exactly_reproduce_selected_predictions']
    assert abs(baseline_report['scores']['TOTAL']-80.26873450885158)<1e-10
    assert selected['source_sha256']==hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    data=prepare_data(master)
    numeric,conditioning=calibration_indices(master['selected_feature_names'])
    source=Original(retain_indices=retained_indices(master['selected_feature_names']),
                    numeric_indices=numeric,conditioning_indices=conditioning)
    source.load_state_dict(master['state_dict'],strict=True)
    source.cuda().eval().requires_grad_(False)
    model=Network(source,selected,args.mode).cuda()
    params=[p for p in model.parameters() if p.requires_grad]
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    fixed={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k not in set(names)}
    evaluator=SourceEvaluator(data,master)
    reference=np.load(SELECTION/'fold_5_pred.npy')
    dependencies=[Path(__file__),Path(__file__).with_name('dual8222_location_readout_v5.py'), Path(__file__).with_name('dual8222_por_subspace.py'),
                  Path(__file__).with_name('train_dual8222_heads80.py'),
                  Path(__file__).with_name('frozen_companion_fused_loss.py'),
                  Path(__file__).with_name('paired_local_context_router_sequence.py'),
                  Path(__file__).with_name('paired_detached_router_sequence.py'),
                  Path(__file__).with_name('por_t3_patch_loss.py'),
                  Path(__file__).with_name('fold5_screening_805.py')]
    cfg={'screening_fold':5,'screening_threshold':THRESHOLD,'mode':args.mode,
         'baseline_total':baseline_report['scores']['TOTAL'],'historical_dual_total':master['best_validation_score'],
         'source_backbone':str(SOURCE.relative_to(ROOT)),'source_backbone_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
         'warm_start_heads':str(selected_path.relative_to(ROOT)),'warm_start_sha256':hashlib.sha256(selected_path.read_bytes()).hexdigest(),
         'warm_start_choices':selected['choices'],'initialization':'new optimization from fixed selected heads, NOT exact continuation of earlier run',
         'epochs':80,'steps':64,'batch_size':64,'fold_seed':20261010,'learning_rate':3e-5,'eta_min':1e-5,'weight_decay':.01,
         'early_stopping':{'maximum_epochs':80,'minimum_epochs':10,'patience':18,'criterion':'POR official score; PERM/SW frozen'},
         'trainable_parameters':sum(p.numel() for p in params),'trainable_names':names,
         'variable_factors':list(MODES[args.mode]),'fixed_targets':['PERM','SW'],'fixed_gate':True,
         'loss':'POR full dual predictive BCE+nondefault NLL, fixedQ50, divided by3 as historical active-target objective',
         'quality':data['qinfo'],'decoder':'original single coverage/256density bins/800actions+default/batch32',
         'test_access':False,'remaining_folds_authorized':False,
         'location_readout':'linear base trainable in all arms; optional bounded32wide residual',
         'residual_bound_fraction':0.25 if args.mode!='linear' else None,
         'local_kernel':5 if args.mode=='local32' else 1,
         'comparison':'same initial80.268735; same linear parameters trainable; point32 versus local32 differs only added192per-expert context parameters',
         'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}}
    if args.resume:
        assert json.loads((out/'config.json').read_text(encoding='utf-8'))==cfg
    else:atomic_json(out/'config.json',cfg)
    optimizer=torch.optim.AdamW(params,lr=3e-5,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,80*64,eta_min=1e-5)
    windows=torch.Generator(device='cuda').manual_seed(20261010)
    elapsed_previous=0.;epoch=0
    torch.cuda.reset_peak_memory_stats()
    if args.resume:
        saved=torch.load(out/'last_state.pt',map_location='cpu',weights_only=False)
        assert saved['config']==cfg
        model.heads.load_state_dict(saved['current_heads'],strict=True)
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        epoch=saved['epoch'];bad=saved['bad_epochs'];best_epoch=saved['best_epoch']
        best_state=saved['best_heads'];best_pred=saved['best_predictions'];best_scores=saved['best_scores']
        history=saved['history'];initial_scores=saved['initial_scores'];elapsed_previous=saved['seconds']
        torch.set_rng_state(saved['rng']['torch']);torch.cuda.set_rng_state_all(saved['rng']['cuda'])
        np.random.set_state(saved['rng']['numpy']);random.setstate(saved['rng']['python'])
        windows.set_state(saved['rng']['windows']);del saved
        print('EXPLICIT_RESUME',epoch,flush=True)
    else:
        initial,initial_scores=evaluator(model)
        check={'different_elements':int(np.count_nonzero(initial!=reference)),
               'score_delta':initial_scores['TOTAL']-cfg['baseline_total'],'scores':initial_scores}
        atomic_json(out/'initial_verification.json',check)
        assert np.array_equal(initial,reference),check
        best_pred=initial.copy();best_scores=initial_scores.copy();best_state=cpu_copy(model.heads.state_dict())
        bad=0;best_epoch=0;history=[]
        print('INITIAL_RECIPE_EXACT',json.dumps(check),flush=True)

    def persist(completed_epoch):
        atomic_torch(out/'heads_best.pt',{'state_dict':best_state,'config':cfg,'best_epoch':best_epoch,'scores':best_scores})
        atomic_array(out/'fold_5_pred.npy',best_pred)
        state={'config':cfg,'epoch':completed_epoch,'current_heads':cpu_copy(model.heads.state_dict()),
               'optimizer':cpu_copy(optimizer.state_dict()),'scheduler':scheduler.state_dict(),
               'bad_epochs':bad,'best_epoch':best_epoch,'best_heads':best_state,'best_predictions':best_pred,
               'best_scores':best_scores,'history':history,'initial_scores':initial_scores,
               'seconds':elapsed_previous+time.perf_counter()-start,
               'rng':{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),
                      'numpy':np.random.get_state(),'python':random.getstate(),'windows':windows.get_state()}}
        atomic_torch(out/'last_state.pt',state)
        atomic_json(out/'fold_5_validation_history.json',history)
    persist(epoch)
    core=torch.zeros(1,256,dtype=torch.bool,device='cuda');core[:,64:192]=True
    for current in range(epoch+1,81):
        if STOP or (epoch>=10 and bad>=18):break
        t=time.perf_counter();model.train();sums=np.zeros(3)
        for step in range(64):
            anchors=data['train_raw'][torch.randint(len(data['train_raw']),(64,),device='cuda',generator=windows)]
            ix,exists=window_indices(anchors,data['lower'],data['upper'])
            relative=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
            mask=exists&core&data['labeled'][ix]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(data['x'][ix],relative,exists,absolute_depth=data['absolute'][ix])
            channels,bce,nll=channel_loss(output,data['labels'][ix],data['default'][ix],mask,data['quality'][ix])
            loss=channels[0]/3
            assert torch.isfinite(loss);loss.backward()
            if current==1 and step==0:
                assert all(p.grad is None for p in model.backbone.parameters())
                assert all(n.startswith('heads.0.') for n,p in model.named_parameters() if p.requires_grad)
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
                assert any(bool(p.grad.abs().sum()>0) for p in params)
                print('GRADIENT_SCOPE_VERIFIED POR '+args.mode+' only; fixed outputs stored as buffers',flush=True)
            torch.nn.utils.clip_grad_norm_(params,2)
            optimizer.step();scheduler.step()
            sums+=np.array([float(channels[0].detach()),float(bce[0].detach()),float(nll[0].detach())])
        epoch=current
        pred,scores=evaluator(model)
        assert np.array_equal(pred[:,1:],reference[:,1:]),'PERM/SW changed'
        if scores['POR']>best_scores['POR']:
            best_scores=scores.copy();best_pred=pred.copy();best_state=cpu_copy(model.heads.state_dict());best_epoch=epoch;bad=0
        else:bad+=1
        item={'epoch':epoch,'official_scores':scores,'official_total':scores['TOTAL'],
              'best_score':best_scores['TOTAL'],'best_epoch':best_epoch,'bad_epochs':bad,
              'por_train_loss':float(sums[0]/64),'por_gate_bce':float(sums[1]/64),'por_continuous_nll':float(sums[2]/64),
              'next_lr':optimizer.param_groups[0]['lr'],'seconds':time.perf_counter()-t}
        history.append(item);persist(epoch)
        atomic_json(out/'runtime_status.json',{'status':'running','pid':os.getpid(),'atlas_current_trial':True,
                    'mode':args.mode,'last_completed_epoch':epoch,'best_assembled_candidate':best_scores['TOTAL']})
        print('EPOCH_VALIDATION',json.dumps(item),flush=True)
    if STOP:
        atomic_json(out/'runtime_status.json',{'status':'stopped_by_user','pid':os.getpid(),'atlas_current_trial':False,
                    'last_completed_epoch':epoch,'resume_available':True,'auto_resume':False})
        return
    model.heads.load_state_dict(best_state,strict=True)
    final,scores=evaluator(model)
    assert np.array_equal(final,best_pred),'Best head replay mismatch'
    assert np.array_equal(final[:,1:],reference[:,1:])
    current_state=model.state_dict()
    assert all(torch.equal(current_state[k].detach().cpu(),v) for k,v in fixed.items()),'Fixed factor or backbone changed'
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==cfg['source_backbone_sha256']
    assert hashlib.sha256(selected_path.read_bytes()).hexdigest()==cfg['warm_start_sha256']
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==cfg['source_hashes'][p.name] for p in dependencies)
    exported={'source_checkpoint':str(SOURCE.relative_to(ROOT)),'source_sha256':cfg['source_backbone_sha256'],
              'readout':model.export_readout(), 'initialization_manifest':selected,
              'choices':[{'target':'POR','arm':args.mode,'warm_start':'v4_mean_only@12','epoch':best_epoch},selected['choices'][1],selected['choices'][2]],
              'selection_scope':'fixed whole-channel developer choice, all rows/wells same recipe','config':cfg}
    atomic_torch(out/'readout_checkpoint.pt',exported)
    report={'fold':5,'screening_fold':5,'scores':scores,'mode':args.mode,'best_epoch':best_epoch,'stopped_epoch':epoch,
            'best_epochs':[best_epoch,34,52],'initial_scores':initial_scores,
            'delta_initial':scores['TOTAL']-initial_scores['TOTAL'],'delta_historical_dual':scores['TOTAL']-master['best_validation_score'],
            'screening_threshold':THRESHOLD,'screening_passed':scores['TOTAL']>THRESHOLD,
            'trainable_parameters':cfg['trainable_parameters'],'fixed_all_gates_unchanged':True,
            'fixed_density_factors_unchanged':True,'PERM_SW_predictions_unchanged':True,'frozen_source_unchanged':True,
            'assembled_heads_exactly_reproduce_selected_predictions':True,'all_validation_rows_covered_once':True,
            'complete_five_folds':False,'remaining_folds_started':False,
            'seconds':elapsed_previous+time.perf_counter()-start,'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
            'timing_scope':'per process setup/training/evaluation/final replay; concurrent pairs not summed as wall time',
            'finished_at':datetime.now().astimezone().isoformat()}
    atomic_json(out/'fold_5_report.json',report)
    atomic_json(out/'runtime_status.json',{'status':'complete','pid':os.getpid(),'atlas_current_trial':False})
    print('FOLD_RESULT',json.dumps(report),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OWNED is not None:
            failure={'status':'failed','pid':os.getpid(),'atlas_current_trial':False,'error':repr(exc),
                     'traceback':traceback.format_exc(),'resume_available':(OWNED/'last_state.pt').exists(),'auto_resume':False}
            atomic_json(OWNED/'failure.json',failure);atomic_json(OWNED/'runtime_status.json',failure)
        raise
