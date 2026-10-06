"""Post-training checks for the newly introduced interpolation/memory mechanisms.

No fitting, no new candidate selection from interventions. Only completed runs.
Full monitor replay is done for inner-screen positives; others get small-window
padding/finite checks. Disabling the best candidate's mechanism is a diagnostic
on the trained weights, not a matched retraining ablation.
"""
from pathlib import Path
from datetime import datetime
import json,time
import numpy as np
import torch
from architecture_zoo_v15 import Network,AdaptiveSampling,BidirectionalStateSpace,LatentMemory
from train_sw_group_v14 import load_data,Evaluator
from train_dual8222_heads80 import atomic_json
from gpu_job_guard_v2 import check_slot

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/architecture_zoo_v15_20260928'
MODES=('adaptive','latent','state_space')


def read(p):return json.loads(p.read_text(encoding='utf-8'))

@torch.no_grad()
def padding_check(model):
    rng=torch.Generator(device='cpu').manual_seed(762401)
    x=torch.randn(4,256,206,generator=rng).cuda();relative=torch.randn(4,256,generator=rng).cuda()
    mask=torch.zeros(4,256,device='cuda',dtype=torch.bool)
    mask[0,:]=True;mask[1,23:]=True;mask[2,:171]=True
    with torch.autocast('cuda',dtype=torch.bfloat16):first=model(x,relative,mask)
    altered=x.clone();altered[~mask]=1000.;rr=relative.clone();rr[~mask]=-900.
    with torch.autocast('cuda',dtype=torch.bfloat16):second=model(altered,rr,mask)
    if not all(torch.isfinite(t).all() for t in first+second):raise RuntimeError('Nonfinite padding forward')
    err=max(float((a-b).abs().max()) for a,b in zip(first,second))
    zero=max(float(a[~mask].abs().max()) for a in first)
    if err!=0 or zero!=0:raise RuntimeError('Invalid padding isolation')
    return dict(padding_change_max_error=err,invalid_positions_max_abs=zero,all_empty_finite=True)

@torch.no_grad()
def recurrence_check(model):
    maximum=0.
    rng=torch.Generator(device='cpu').manual_seed(974)
    u=torch.randn(2,64,17,generator=rng).cuda()
    for block in model.blocks:
        if not isinstance(block,BidirectionalStateSpace):continue
        for direction in [0,1]:
            got=block.convolve(u,direction)
            tau=torch.nn.functional.softplus(block.raw_tau[direction].float())+1
            a=torch.exp(-1/tau)[None];carry=torch.zeros(2,64,device='cuda');states=[]
            for t in range(17):
                carry=a*carry+(1-a)*u[:,:,t];states.append(carry.clone())
            expected=torch.stack(states,-1)
            maximum=max(maximum,float((got-expected).abs().max()))
    if maximum>2e-5:raise RuntimeError('FFT differs from zero-state recurrence')
    return dict(fft_vs_explicit_recurrence_max_error=maximum,no_circular_wrap_test_length=17)

@torch.no_grad()
def interpolation_check():
    values=torch.arange(8,device='cuda',dtype=torch.float32).reshape(1,8,1)
    valid=torch.ones(1,8,device='cuda',dtype=torch.bool)
    positions=torch.tensor([[[-1.,.5,2.75,7.,7.5,8.]]],device='cuda')
    got=AdaptiveSampling.sample(values,valid,positions).flatten()
    expected=torch.tensor([0.,.5,2.75,7.,3.5,0.],device='cuda')
    error=float((got-expected).abs().max());assert error==0
    return dict(fractional_interpolation_known_values_max_error=error)


def main():
    progress=read(STUDY/'progress.json')
    if not progress['all_complete']:raise RuntimeError('Wait for all completed runs; no extra concurrent workload')
    torch.set_num_threads(3);check_slot(min_free_gib=3.)
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cudnn.benchmark=False
    d=load_data('inner');evaluator=Evaluator(d,d['monitor_raw'])
    best=progress['best_architecture'];records=[]
    for row in progress['rows']:
        mode=row['architecture'];out=ROOT/'outputs'/row['id'];start=time.perf_counter()
        cp=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False)
        model=Network(mode,cp['centers'],cp['target_median'],cp['target_scale']).cuda()
        model.load_state_dict(cp['state_dict'],strict=True);model.eval()
        result=dict(architecture=mode,checked_at=datetime.now().astimezone().isoformat(),loaded_from_disk=True,weights_changed=False,
            **padding_check(model),full_monitor_replay=False)
        if mode=='adaptive':result.update(interpolation_check())
        if mode=='state_space':result.update(recurrence_check(model))
        if row['comparison']['predeclared_signal']:
            pred,stats=evaluator(model);saved=np.load(out/'monitor_pred.npy')
            exact=bool(np.array_equal(pred,saved))
            if not exact:raise RuntimeError('Independent reload differs for '+mode)
            result.update(full_monitor_replay=True,monitor_points=len(pred),all_monitor_predictions_exact=exact,SW=stats['SW'])
        if mode==best and row['comparison']['predeclared_signal']:
            for block in model.blocks:
                if isinstance(block,AdaptiveSampling):block.freeze_offsets_for_diagnostic=True
                elif isinstance(block,LatentMemory):block.disable_global_for_diagnostic=True
                elif isinstance(block,BidirectionalStateSpace):block.disable_memory_for_diagnostic=True
            pred,stats=evaluator(model)
            result['trained_model_mechanism_disabled']=dict(SW=stats['SW'],delta_vs_full=stats['SW']-row['report']['target_score'],
                action='zero learned offsets' if mode=='adaptive' else ('bypass latent read/process/write' if mode=='latent' else 'zero recurrent memory contribution'),
                interpretation='post-training intervention only;not a matched retraining effect or a new selectable candidate')
        result['seconds']=time.perf_counter()-start
        atomic_json(out/'implementation_verification.json',result);records.append(result)
        print('VERIFIED',json.dumps(result),flush=True)
        del cp,model;torch.cuda.empty_cache()
    atomic_json(STUDY/'implementation_checks.json',dict(records=records,original_fifth_evaluated=False,no_training_updates=True))

if __name__=='__main__':main()
