"""Train the two prespecified posterior contexts for one private target model."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from datetime import datetime
import argparse,hashlib,json,random,signal,sys,time,traceback
import numpy as np
import torch
from champion_posterior_v27 import Network,Evaluator,MODES,SELECTED,ROOT,STUDY,PREVIOUS,load_data,reference_prediction,TARGETS
from train_scaling_v21 import channel_objective
from train_dual8222_heads80 import atomic_json,atomic_array,atomic_torch,cpu_copy
from multiscale_sequence import window_indices
from gpu_job_guard_v2 import check_slot

STOP=False
OWNED=None


def stop_request(*_):
    global STOP
    STOP=True
    print('STOP_REQUESTED: save current epoch, explicit resume only',flush=True)


def summary(states):
    return {m:{k:s[k] for k in ('best','best_epoch','bad','done','last_epoch','stop_reason')} for m,s in states.items()}


@torch.no_grad()
def factor_scope(model,data,mode):
    anchors=data['train_raw'][torch.tensor([0,len(data['train_raw'])//2],device='cuda')]
    ix,valid=window_indices(anchors,data['lower'],data['upper'])
    rel=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
    h,o,logits,_=model.extract(data['x'][ix],rel,valid,data['absolute'][ix])
    new=model.heads[mode].corrected_members(h,o,data['x'][ix,...,206:],valid)
    unchanged=[];changed=[]
    for e,((g,r),(ng,nr)) in enumerate(zip(o,new)):
        fields={'gate':(g,ng),'weight':(r[...,0,:],nr[...,0,:]),
                'location':(r[...,1,:],nr[...,1,:]),'scale':(r[...,2,:],nr[...,2,:])}
        for key,(a,b) in fields.items():
            if key!=model.factor:
                assert torch.equal(a,b),(mode,e,key)
                unchanged.append(f'expert{e}.{key}')
            else:changed.append(float((a.float()-b.float()).abs()[valid].max()))
    return dict(unchanged_factors=unchanged,selected_factor_max_abs=changed,
                scope='two training windows; measured relative to retained v26 outputs')


def main():
    global OWNED
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--target',choices=tuple(SELECTED),required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();out=args.out.resolve();target=args.target;c=TARGETS.index(target)
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
    assert reference.shape==(144665,3) and np.isfinite(reference).all()
    model=Network(target,master).cuda();evaluator=Evaluator(data,master,c,reference)
    fixed=cpu_copy(model.frozen_state());initial={m:cpu_copy(model.heads[m].state_dict()) for m in MODES}
    assert all(torch.equal(initial['self'][k],v) for k,v in initial['pair'].items())
    ids=[{id(p) for p in model.heads[m].parameters()} for m in MODES]
    assert ids[0].isdisjoint(ids[1])
    assert all(not p.requires_grad for n,p in model.named_parameters() if not n.startswith('heads.'))
    sources=[Path(__file__)]+[Path(__file__).with_name(n) for n in
             ('champion_posterior_v27.py','champion_all_heads_v26.py','champion_shape_v25.py','train_scaling_v21.py')]
    cfg=dict(target=target,factor=SELECTED[target],modes=list(MODES),screening_fold=5,
        model_module='champion_posterior_v27',source_checkpoint=model.source_path,retained_head=model.retained_path,
        frozen_parameters=sum(p.numel() for n,p in model.named_parameters() if not n.startswith('heads.')),
        per_candidate_trainable_parameters={m:sum(p.numel() for p in model.heads[m].parameters()) for m in MODES},
        inputs=dict(backbone=206,amplitude_side=63,hidden=128,own_posterior=4,second_posterior=4,head_input=199,head_hidden=32),
        descriptors=['default_probability','normalized_mixture_entropy','clipped_transformed_mean','clipped_log1p_std'],
        matched_change='self repeats own descriptor; pair uses other expert descriptor; all old weights frozen',
        posterior_gradients=False,shared_trainable_parameters=False,shared_backbone_between_targets=False,
        epochs=80,patience=18,minimum_epochs=10,steps_per_epoch=64,batch_size=64,seed=20261010,
        new_head_seed=20261103,learning_rate=.0003,eta_min=.000001,weight_decay=.01,clip_norm=2.,
        correction_bound=.25,location_bound_unit='fixed current component std',quality=data['qinfo'],
        reference_scores=ref,validation_rows=144665,training_rows=578891,
        objective='original own-target complete predictive likelihood/3; Q50 and density families unchanged',
        decoder='original singlecoverage/256bins/800actions+default/batch32',
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        torch_version=str(torch.__version__),deterministic_mode='warn_only; not bitwise retraining claim',
        complete_five_folds=False,remaining_folds_started=False)
    if args.resume:assert json.loads((out/'config.json').read_text(encoding='utf-8'))==cfg
    else:atomic_json(out/'config.json',cfg)
    opts={m:torch.optim.AdamW(model.heads[m].parameters(),lr=3e-4,weight_decay=.01) for m in MODES}
    scheds={m:torch.optim.lr_scheduler.CosineAnnealingLR(opts[m],80*64,eta_min=1e-6) for m in MODES}
    windows=torch.Generator(device='cuda').manual_seed(20261010)
    epoch=0;elapsed=0.;epoch_timing=[]
    if args.resume:
        saved=torch.load(out/'last_state.pt',map_location='cpu',weights_only=False);assert saved['config']==cfg
        model.heads.load_state_dict(saved['current_heads'],strict=True)
        for m in MODES:opts[m].load_state_dict(saved['optimizers'][m]);scheds[m].load_state_dict(saved['schedulers'][m])
        states=saved['states'];epoch=saved['epoch'];elapsed=saved['seconds'];epoch_timing=saved['epoch_timing']
        torch.set_rng_state(saved['rng']['torch']);torch.cuda.set_rng_state_all(saved['rng']['cuda'])
        windows.set_state(saved['rng']['windows']);np.random.set_state(saved['rng']['numpy']);random.setstate(saved['rng']['python']);del saved
    else:
        values=evaluator.many(model,MODES);states={}
        for m,(pred,score,total) in values.items():
            assert np.array_equal(pred,reference[:,c]),(target,m,'initial v26 function mismatch')
            arm=out/m;arm.mkdir(exist_ok=True)
            atomic_json(arm/'initial_verification.json',dict(target=target,mode=m,score=score,TOTAL=total,all144665_exact=True))
            states[m]=dict(best=score,best_epoch=0,best_state=cpu_copy(model.heads[m].state_dict()),best_pred=pred.copy(),
                best_updated=None,updated_state=None,updated_pred=None,bad=0,history=[],done=False,last_epoch=0,stop_reason=None,proof=None)
        print('INITIAL',json.dumps(dict(target=target,scores={m:v[1] for m,v in values.items()},TOTAL=ref['TOTAL'])),flush=True)

    def save_arm(m,s):
        arm=out/m;arm.mkdir(exist_ok=True)
        common=dict(target=target,mode=m,factor=model.factor,config=cfg,source_checkpoint=model.source_path,
                    retained_head=model.retained_path,checkpoint_format='v26_frozen_source_plus_v27_delta')
        atomic_torch(arm/'model_best.pt',dict(delta_state=s['best_state'],best_epoch=s['best_epoch'],target_score=s['best'],**common))
        atomic_array(arm/'channel_pred.npy',s['best_pred']);atomic_json(arm/'validation_history.json',s['history'])
        if s['best_updated'] is not None:
            atomic_torch(arm/'model_best_updated.pt',dict(delta_state=s['updated_state'],**s['best_updated'],**common))
            atomic_array(arm/'best_updated_pred.npy',s['updated_pred'])

    def persist():
        for m,s in states.items():save_arm(m,s)
        atomic_torch(out/'last_state.pt',dict(config=cfg,current_heads=cpu_copy(model.heads.state_dict()),
            optimizers={m:cpu_copy(o.state_dict()) for m,o in opts.items()},schedulers={m:s.state_dict() for m,s in scheds.items()},
            states=states,epoch=epoch,epoch_timing=epoch_timing,seconds=elapsed+time.perf_counter()-start,
            rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),windows=windows.get_state(),numpy=np.random.get_state(),python=random.getstate())))
    persist();torch.cuda.reset_peak_memory_stats()
    core=torch.zeros((1,256),device='cuda',dtype=torch.bool);core[:,64:192]=True
    for ep in range(epoch+1,81):
        active=[m for m in MODES if not states[m]['done']]
        if STOP or not active:break
        tick=time.perf_counter();model.train();losses={m:0. for m in active}
        for step in range(64):
            anchors=data['train_raw'][torch.randint(len(data['train_raw']),(64,),device='cuda',generator=windows)]
            ix,valid=window_indices(anchors,data['lower'],data['upper'])
            rel=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
            mask=valid&core&data['labeled'][ix]
            h,o,logits,_=model.extract(data['x'][ix],rel,valid,data['absolute'][ix])
            for m in active:
                opts[m].zero_grad(set_to_none=True)
                output=model.heads[m](h,o,logits,data['x'][ix,...,206:],valid)
                loss,_,_=channel_objective(output,data['labels'][ix],data['default'][ix],mask,data['quality'][ix],c)
                if not bool(torch.isfinite(loss)):raise FloatingPointError(f'Nonfinite loss {target}/{m}')
                loss.backward()
                if states[m]['proof'] is None and step==1:
                    def norm(part):
                        values=[p.grad.detach().float().square().sum() for n,p in model.heads[m].named_parameters() if part in n and p.grad is not None]
                        return float(torch.stack(values).sum().sqrt()) if values else 0.
                    proof=dict(input_gradient_norm=norm('.input.'),output_gradient_norm=norm('.output.'),
                        frozen_gradients_none=all(p.grad is None for n,p in model.named_parameters() if not n.startswith('heads.')),
                        target_only=target,second_step=True)
                    assert proof['output_gradient_norm']>0 and proof['frozen_gradients_none']
                    states[m]['proof']=proof;atomic_json(out/m/'gradient_scope.json',proof)
                torch.nn.utils.clip_grad_norm_(model.heads[m].parameters(),2.,error_if_nonfinite=True)
                opts[m].step();scheds[m].step();losses[m]+=float(loss.detach())
            del h,o,logits
        epoch=ep;values=evaluator.many(model,active)
        for m,(pred,score,total) in values.items():
            s=states[m];s['last_epoch']=ep
            if score>s['best']:
                s.update(best=score,best_epoch=ep,best_state=cpu_copy(model.heads[m].state_dict()),best_pred=pred.copy(),bad=0)
            else:s['bad']+=1
            if s['best_updated'] is None or score>s['best_updated']['target_score']:
                s.update(best_updated=dict(epoch=ep,target_score=score,reference_replacement_total=total),
                    updated_state=cpu_copy(model.heads[m].state_dict()),updated_pred=pred.copy())
            s['history'].append(dict(epoch=ep,target_score=score,reference_replacement_total=total,best_score=s['best'],
                best_epoch=s['best_epoch'],bad_epochs=s['bad'],training_loss=losses[m]/64,learning_rate=opts[m].param_groups[0]['lr']))
            if ep>=10 and s['bad']>=18:s.update(done=True,stop_reason='patience18')
            elif ep==80:s.update(done=True,stop_reason='budget80')
        epoch_timing.append(dict(epoch=ep,active_modes=active,seconds=time.perf_counter()-tick))
        persist()
        atomic_json(out/'runtime_status.json',dict(status='running',target=target,pid=os.getpid(),last_completed_epoch=ep,
            modes=summary(states),atlas_current_trial=True))
        print('EPOCH',json.dumps(dict(epoch=ep,target=target,modes=summary(states),seconds=epoch_timing[-1]['seconds'])),flush=True)
    if STOP:
        atomic_json(out/'runtime_status.json',dict(status='stopped_by_user',target=target,pid=os.getpid(),resume_available=True,auto_resume=False));return
    assert all(s['done'] for s in states.values())
    assert all(torch.equal(v.detach().cpu(),fixed[n]) for n,v in model.frozen_state().items())
    changes={m:float(sum((v.detach().cpu().double()-initial[m][n].double()).square().sum()
        for n,v in model.heads[m].state_dict().items()).sqrt()) for m in MODES}
    assert all(v>0 for v in changes.values())
    for m,s in states.items():model.heads[m].load_state_dict(s['best_state'],strict=True)
    restored=evaluator.many(model,MODES);arms=[]
    for m,(pred,score,total) in restored.items():
        s=states[m];assert np.array_equal(pred,s['best_pred'])
        report=dict(target=target,mode=m,factor=model.factor,screening_fold=5,target_score=score,
            reference_replacement_total=total,delta_reference_target=score-ref[target],
            best_epoch=s['best_epoch'],stopped_epoch=s['last_epoch'],stop_reason=s['stop_reason'],
            best_after_update=s['best_updated'],trainable_parameters=cfg['per_candidate_trainable_parameters'][m],
            all144665_selected_predictions_restored_exactly=True,all_frozen_state_unchanged=True,
            gradient_scope=s['proof'],factor_scope=factor_scope(model,data,m),last_delta_parameter_change_L2=changes[m],
            checkpoint=str((out/m/'model_best.pt').relative_to(ROOT)),complete_five_folds=False,remaining_folds_started=False)
        atomic_json(out/m/'fold_5_report.json',report);arms.append(report)
    report=dict(target=target,factor=model.factor,screening_fold=5,arms=arms,reference_scores=ref,
        seconds=elapsed+time.perf_counter()-start,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        epoch_timing=epoch_timing,finished_at=datetime.now().astimezone().isoformat(),complete_five_folds=False,remaining_folds_started=False)
    atomic_json(out/'fold_5_report.json',report)
    atomic_json(out/'runtime_status.json',dict(status='complete',target=target,pid=os.getpid(),modes=summary(states),atlas_current_trial=False))
    model.close();print('FINAL',json.dumps({k:v for k,v in report.items() if k!='epoch_timing'}),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OWNED is not None:
            failure=dict(status='failed',pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc(),auto_resume=False,atlas_current_trial=False)
            atomic_json(OWNED/'failure.json',failure);atomic_json(OWNED/'runtime_status.json',failure)
        raise
