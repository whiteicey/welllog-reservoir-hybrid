"""Train one target's independent factor candidates, sharing frozen inference only."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from datetime import datetime
import argparse,hashlib,json,random,signal,sys,time,traceback
import numpy as np
import torch
from champion_all_heads_v26 import Network,Evaluator,FACTORS,ROOT,STUDY,PREVIOUS,load_data,reference_prediction,TARGETS
from train_scaling_v21 import channel_objective
from train_dual8222_heads80 import atomic_json,atomic_array,atomic_torch,cpu_copy
from multiscale_sequence import window_indices
from gpu_job_guard_v2 import check_slot

STOP=False
OWNED=None


def stop_request(*_):
    global STOP
    STOP=True
    print('STOP_REQUESTED: finish current epoch; explicit resume only',flush=True)


def small_summary(states):
    return {f:{k:s[k] for k in ('best','best_epoch','bad','done','last_epoch','stop_reason')} for f,s in states.items()}


@torch.no_grad()
def factor_scope(model,data,factor):
    anchors=data['train_raw'][torch.tensor([0,len(data['train_raw'])//2],device='cuda')]
    ix,valid=window_indices(anchors,data['lower'],data['upper'])
    rel=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
    h,outputs,logits,_=model.extract(data['x'][ix],rel,valid,data['absolute'][ix])
    new=model.heads[factor].corrected_members(h,outputs,data['x'][ix,...,206:],valid)
    unchanged=[];changed=[]
    for e,((g,r),(ng,nr)) in enumerate(zip(outputs,new)):
        fields={'gate':(g,ng),'weight':(r[...,0,:],nr[...,0,:]),
                'location':(r[...,1,:],nr[...,1,:]),'scale':(r[...,2,:],nr[...,2,:])}
        for key,(a,b) in fields.items():
            if key!=factor:
                assert torch.equal(a,b),(factor,e,key)
                unchanged.append(f'expert{e}.{key}')
            else:changed.append(float((a.float()-b.float()).abs()[valid].max()))
    return dict(unchanged_expert_factors=unchanged,changed_factor_max_abs=changed,
                scope='two fixed training windows; gate change may reweight merged survival mixture')


def main():
    global OWNED
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--target',choices=tuple(FACTORS),required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();out=args.out.resolve();target=args.target;c=TARGETS.index(target);factors=FACTORS[target]
    if (out/'fold_5_report.json').exists():raise RuntimeError('Completed experiment is immutable')
    if (out/'config.json').exists() and not args.resume:raise RuntimeError('Existing run requires explicit resume')
    out.mkdir(parents=True,exist_ok=True);OWNED=out
    torch.set_num_threads(3);print('ADMISSION',check_slot(min_free_gib=3.),flush=True)
    atomic_json(out/'runtime_status.json',dict(status='running',target=target,pid=os.getpid(),atlas_current_trial=True))
    signal.signal(signal.SIGINT,stop_request);signal.signal(signal.SIGTERM,stop_request)
    start=time.perf_counter();torch.manual_seed(20261010);np.random.seed(20261010);random.seed(20261010)
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
    data,master=load_data();reference=reference_prediction()
    ref=json.loads((PREVIOUS/'CURRENT_RECIPE.json').read_text(encoding='utf-8'))['scores']
    model=Network(target,master).cuda();evaluator=Evaluator(data,master,c,reference)
    fixed=cpu_copy(model.source.state_dict())
    initial_delta={f:cpu_copy(model.heads[f].state_dict()) for f in factors}
    ids=[{id(p) for p in model.heads[f].parameters()} for f in factors]
    assert len(set.union(*ids))==sum(len(i) for i in ids)
    assert not any(p.requires_grad for p in model.source.parameters())
    sources=[Path(__file__)]+[Path(__file__).with_name(n) for n in
             ('champion_all_heads_v26.py','champion_shape_v25.py','independent_channels_v8.py','train_scaling_v21.py','paired_detached_router_sequence.py')]
    cfg=dict(target=target,factors=list(factors),screening_fold=5,model_module='champion_all_heads_v26',
        source_checkpoint=model.source_path,frozen_source_parameters=sum(p.numel() for p in model.source.parameters()),
        per_factor_trainable_parameters={f:sum(p.numel() for p in model.heads[f].parameters()) for f in factors},
        shared_trainable_parameters=False,shared_frozen_inference_within_target=True,backbone_shared_between_targets=False,
        input_features=206,side_features=63,head_input=191,head_hidden=32,
        bounds=dict(gate_logit=.25,mixture_logit=.25,location_std_fraction=.25,raw_scale=.25),
        source_eval_only=True,optimizer_per_factor=True,checkpoint_selection_per_factor=True,
        epochs=80,patience=18,minimum_epochs=10,steps_per_epoch=64,batch_size=64,seed=20261010,
        new_head_seed=20261102,learning_rate=.0003,eta_min=.000001,weight_decay=.01,clip_norm=2.,
        quality=data['qinfo'],reference_scores=ref,validation_rows=144665,training_rows=578891,
        cache='results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',
        amplitude_metadata='results/deep_learning/feature_structure_v20_20260928/feature_manifest.json',
        objective='own target complete predictive likelihood/3, original StudentT/Gaussian and fixedQ50',
        decoder='original target-specific singlecoverage256bins/800actions+default/batch32',
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        torch_version=str(torch.__version__),deterministic_mode='warn_only; not a bitwise repeated-training claim',
        complete_five_folds=False,remaining_folds_started=False)
    if args.resume:assert json.loads((out/'config.json').read_text(encoding='utf-8'))==cfg
    else:atomic_json(out/'config.json',cfg)
    opts={f:torch.optim.AdamW(model.heads[f].parameters(),lr=3e-4,weight_decay=.01) for f in factors}
    scheds={f:torch.optim.lr_scheduler.CosineAnnealingLR(opts[f],80*64,eta_min=1e-6) for f in factors}
    windows=torch.Generator(device='cuda').manual_seed(20261010)
    epoch=0;elapsed=0.;epoch_timing=[]
    if args.resume:
        saved=torch.load(out/'last_state.pt',map_location='cpu',weights_only=False);assert saved['config']==cfg
        model.heads.load_state_dict(saved['current_heads'],strict=True)
        for f in factors:opts[f].load_state_dict(saved['optimizers'][f]);scheds[f].load_state_dict(saved['schedulers'][f])
        states=saved['states'];epoch=saved['epoch'];elapsed=saved['seconds'];epoch_timing=saved['epoch_timing']
        torch.set_rng_state(saved['rng']['torch']);torch.cuda.set_rng_state_all(saved['rng']['cuda'])
        windows.set_state(saved['rng']['windows']);np.random.set_state(saved['rng']['numpy']);random.setstate(saved['rng']['python']);del saved
    else:
        initial=evaluator.many(model,factors);states={}
        for f,(pred,score,total) in initial.items():
            assert np.array_equal(pred,reference[:,c]),(target,f,'initial function changed')
            arm=out/f;arm.mkdir(exist_ok=True)
            atomic_json(arm/'initial_verification.json',dict(target=target,factor=f,score=score,TOTAL=total,all144665_exact=True))
            states[f]=dict(best=score,best_epoch=0,best_state=cpu_copy(model.heads[f].state_dict()),best_pred=pred.copy(),
                best_updated=None,updated_state=None,updated_pred=None,bad=0,history=[],done=False,last_epoch=0,stop_reason=None,proof=None)
        print('INITIAL',json.dumps(dict(target=target,scores={f:v[1] for f,v in initial.items()},TOTAL=ref['TOTAL'])),flush=True)

    def save_arm(f,s):
        arm=out/f;arm.mkdir(exist_ok=True)
        common=dict(target=target,factor=f,config=cfg,source_checkpoint=model.source_path,checkpoint_format='frozen_source_plus_delta')
        atomic_torch(arm/'model_best.pt',dict(delta_state=s['best_state'],best_epoch=s['best_epoch'],target_score=s['best'],**common))
        atomic_array(arm/'channel_pred.npy',s['best_pred'])
        atomic_json(arm/'validation_history.json',s['history'])
        if s['best_updated'] is not None:
            atomic_torch(arm/'model_best_updated.pt',dict(delta_state=s['updated_state'],**s['best_updated'],**common))
            atomic_array(arm/'best_updated_pred.npy',s['updated_pred'])

    def persist():
        for f,s in states.items():save_arm(f,s)
        atomic_torch(out/'last_state.pt',dict(config=cfg,current_heads=cpu_copy(model.heads.state_dict()),
            optimizers={f:cpu_copy(o.state_dict()) for f,o in opts.items()},schedulers={f:s.state_dict() for f,s in scheds.items()},
            states=states,epoch=epoch,epoch_timing=epoch_timing,seconds=elapsed+time.perf_counter()-start,
            rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),windows=windows.get_state(),numpy=np.random.get_state(),python=random.getstate())))
    persist();torch.cuda.reset_peak_memory_stats()
    core=torch.zeros((1,256),device='cuda',dtype=torch.bool);core[:,64:192]=True
    for ep in range(epoch+1,81):
        active=[f for f in factors if not states[f]['done']]
        if STOP or not active:break
        tick=time.perf_counter();model.train();losses={f:0. for f in active}
        for step in range(64):
            anchors=data['train_raw'][torch.randint(len(data['train_raw']),(64,),device='cuda',generator=windows)]
            ix,valid=window_indices(anchors,data['lower'],data['upper'])
            rel=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
            mask=valid&core&data['labeled'][ix]
            h,o,logits,_=model.extract(data['x'][ix],rel,valid,data['absolute'][ix])
            for f in active:
                opts[f].zero_grad(set_to_none=True)
                output=model.heads[f](h,o,logits,data['x'][ix,...,206:],valid)
                loss,_,_=channel_objective(output,data['labels'][ix],data['default'][ix],mask,data['quality'][ix],c)
                if not bool(torch.isfinite(loss)):raise FloatingPointError(f'Nonfinite loss {target}/{f}')
                loss.backward()
                if states[f]['proof'] is None and step==1:
                    def norm(part):
                        v=[p.grad.detach().float().square().sum() for n,p in model.heads[f].named_parameters() if part in n and p.grad is not None]
                        return float(torch.stack(v).sum().sqrt()) if v else 0.
                    proof=dict(input_gradient_norm=norm('.input.'),output_gradient_norm=norm('.output.'),
                               original_gradients_none=all(p.grad is None for p in model.source.parameters()),target_only=target,second_step=True)
                    assert proof['output_gradient_norm']>0 and proof['original_gradients_none']
                    states[f]['proof']=proof;atomic_json(out/f/'gradient_scope.json',proof)
                torch.nn.utils.clip_grad_norm_(model.heads[f].parameters(),2.,error_if_nonfinite=True)
                opts[f].step();scheds[f].step();losses[f]+=float(loss.detach())
            del h,o,logits
        epoch=ep;results=evaluator.many(model,active)
        for f,(pred,score,total) in results.items():
            s=states[f];s['last_epoch']=ep
            if score>s['best']:
                s.update(best=score,best_epoch=ep,best_state=cpu_copy(model.heads[f].state_dict()),best_pred=pred.copy(),bad=0)
            else:s['bad']+=1
            if s['best_updated'] is None or score>s['best_updated']['target_score']:
                s.update(best_updated=dict(epoch=ep,target_score=score,reference_replacement_total=total),
                         updated_state=cpu_copy(model.heads[f].state_dict()),updated_pred=pred.copy())
            s['history'].append(dict(epoch=ep,target_score=score,reference_replacement_total=total,best_score=s['best'],
                best_epoch=s['best_epoch'],bad_epochs=s['bad'],training_loss=losses[f]/64,learning_rate=opts[f].param_groups[0]['lr']))
            if ep>=10 and s['bad']>=18:s.update(done=True,stop_reason='patience18')
            elif ep==80:s.update(done=True,stop_reason='budget80')
        epoch_timing.append(dict(epoch=ep,active_factors=active,seconds=time.perf_counter()-tick))
        persist()
        atomic_json(out/'runtime_status.json',dict(status='running',target=target,pid=os.getpid(),last_completed_epoch=ep,
                    factors=small_summary(states),atlas_current_trial=True))
        print('EPOCH',json.dumps(dict(epoch=ep,target=target,factors=small_summary(states),seconds=epoch_timing[-1]['seconds'])),flush=True)
    if STOP:
        atomic_json(out/'runtime_status.json',dict(status='stopped_by_user',target=target,pid=os.getpid(),resume_available=True,auto_resume=False));return
    assert all(s['done'] for s in states.values())
    assert all(torch.equal(v.detach().cpu(),fixed[n]) for n,v in model.source.state_dict().items())
    last_changes={f:float(sum((v.detach().cpu().double()-initial_delta[f][n].double()).square().sum()
                             for n,v in model.heads[f].state_dict().items()).sqrt()) for f in factors}
    assert all(v>0 for v in last_changes.values())
    for f,s in states.items():model.heads[f].load_state_dict(s['best_state'],strict=True)
    restored=evaluator.many(model,factors);arms=[]
    for f,(pred,score,total) in restored.items():
        s=states[f];assert np.array_equal(pred,s['best_pred'])
        report=dict(target=target,factor=f,screening_fold=5,target_score=score,reference_replacement_total=total,
            delta_reference_target=score-ref[target],best_epoch=s['best_epoch'],stopped_epoch=s['last_epoch'],stop_reason=s['stop_reason'],
            best_after_update=s['best_updated'],trainable_parameters=cfg['per_factor_trainable_parameters'][f],
            frozen_source_parameters=cfg['frozen_source_parameters'],all144665_selected_predictions_restored_exactly=True,
            all_original_state_unchanged=True,gradient_scope=s['proof'],factor_scope=factor_scope(model,data,f),
            last_delta_parameter_change_L2=last_changes[f],checkpoint=str((out/f/'model_best.pt').relative_to(ROOT)),
            complete_five_folds=False,remaining_folds_started=False)
        atomic_json(out/f/'fold_5_report.json',report);arms.append(report)
    report=dict(target=target,screening_fold=5,arms=arms,reference_scores=ref,
        best_factor=max(arms,key=lambda r:r['target_score'])['factor'],
        all_original_state_unchanged=True,all144665_selected_predictions_restored_exactly=True,
        seconds=elapsed+time.perf_counter()-start,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        epoch_timing=epoch_timing,finished_at=datetime.now().astimezone().isoformat(),complete_five_folds=False,remaining_folds_started=False)
    atomic_json(out/'fold_5_report.json',report)
    atomic_json(out/'runtime_status.json',dict(status='complete',target=target,pid=os.getpid(),factors=small_summary(states),atlas_current_trial=False))
    model.close();print('FINAL',json.dumps({k:v for k,v in report.items() if k!='epoch_timing'}),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OWNED is not None:
            failure=dict(status='failed',pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc(),auto_resume=False,atlas_current_trial=False)
            atomic_json(OWNED/'failure.json',failure);atomic_json(OWNED/'runtime_status.json',failure)
        raise
