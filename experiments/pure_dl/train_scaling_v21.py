"""Explicit single-target end-to-end fitting with a private backbone and checkpoint.

Original fifth-fold population and decoder. Six bounded candidates are launched
explicitly by the planner, never queued. Per-run TOTAL substitution is a diagnostic;
only an assembled three-model prediction receives an actual three-target TOTAL.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,math,random,signal,sys,time,traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from independent_channels_v8 import TARGETS,ARCHITECTURES
from champion_scaling_v21 import make_model,load_data,MODES
from train_dual8222_heads80 import prepare_data,atomic_json,atomic_torch,atomic_array,cpu_copy,SOURCE
from multiscale_sequence import window_indices
from por_t3_patch_loss import probability
from gpu_job_guard_v2 import check_slot
import train_invariant as base

ROOT=Path(__file__).resolve().parents[2]
REFERENCE=ROOT/'outputs/pure_dl_dual8222_v5_location_point3280_fold5'
STOP=False
OWNED=None


def stop_request(_s,_f):
    global STOP
    STOP=True
    print('STOP_REQUESTED: finish epoch and save; explicit resume only',flush=True)


def channel_objective(output,truth,default,mask,quality,target_index):
    # ONLY target_index is read from the label array. No other-target loss.
    gate,raw=output
    g=gate[mask][:,0].float()
    r=raw[mask][:,0].float();d=default[mask][:,target_index]
    y=truth[mask][:,target_index].masked_fill(d,0)
    mu=r[:,1];sd=F.softplus(r[:,2])+.025
    if target_index==0:
        scale=sd/math.sqrt(3)
        logp=math.lgamma(2)-math.lgamma(1.5)-.5*math.log(3*math.pi)-scale.log()-2*torch.log1p(((y[:,None]-mu)/scale).square()/3)
    else:
        logp=-.5*((y[:,None]-mu)/sd).square()-sd.log()-.5*math.log(2*math.pi)
    nll=-(r[:,0].log_softmax(-1)+logp).logsumexp(-1).masked_fill(d,0)
    bce=F.binary_cross_entropy_with_logits(g,d.float(),reduction='none')
    w=quality[mask].detach().float()
    return (w*(bce+nll)).mean()/3,(w*bce).mean(),(w*nll).mean()


class Evaluator:
    def __init__(self,data,master,c,reference):
        self.data=data;self.c=c;self.reference=reference
        ct=np.asarray(master['centers'][c]);name=TARGETS[c]
        assert len(ct)==256
        action=np.linspace(ct[0],ct[-1],800) if c==2 else np.geomspace(max(ct[0]*.9,.001),ct[-1]*1.1,800)
        self.actions=np.unique(np.r_[base.DEFAULTS[name],np.clip(action,*base.BOUNDS[name])])
        self.utilities=torch.tensor(base.kernel_matrix(self.actions,np.r_[base.DEFAULTS[name],ct],name).T,device='cuda',dtype=torch.float32)
        self.edges=torch.tensor(master['density_edges'][c],device='cuda',dtype=torch.float32)
        self.local=np.full(len(data['truth']),-1,np.int64);self.local[data['validation']]=np.arange(len(data['validation']))
        self.centers=[];self.anchors=[]
        for wi,rows in enumerate(data['row_ids']):
            if data['fold'][rows[0]]!=4:continue
            cc=np.arange(data['starts'][wi],data['starts'][wi+1],128)+64
            self.centers.extend(cc);self.anchors.extend(np.minimum(cc,data['starts'][wi+1]-1))

    @torch.no_grad()
    def __call__(self,model):
        d=self.data;c=self.c;model.eval()
        result=np.full(len(d['validation']),np.nan,np.float64);coverage=np.zeros(len(result),np.int8)
        for begin in range(0,len(self.centers),32):
            centers=torch.tensor(self.centers[begin:begin+32],device='cuda')
            anchors=torch.tensor(self.anchors[begin:begin+32],device='cuda')
            ix,valid=window_indices(centers,d['lower'],d['upper'],anchors=anchors)
            relative=torch.tanh((d['depth'][ix]-d['depth'][anchors,None])/10)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                go,raw=model(d['x'][ix],relative,valid,absolute_depth=d['absolute'][ix])
            rows=d['raw_to_valid'][ix[:,64:192].cpu().numpy()]
            use=valid[:,64:192].cpu().numpy()&(rows>=0)
            loc=self.local[rows[use]]
            assert (loc>=0).all() and (coverage[loc]==0).all() and len(np.unique(loc))==len(loc)
            coverage[loc]+=1;take=torch.from_numpy(use).cuda()
            g=go[:,64:192][take][:,0].float().sigmoid();r=raw[:,64:192][take][:,0].float()
            p=probability(c,r[:,0].softmax(-1),r[:,1],F.softplus(r[:,2])+.025,self.edges)
            joint=torch.cat([g[:,None],(1-g[:,None])*p],-1)
            result[loc]=self.actions[(joint@self.utilities).argmax(-1).cpu().numpy()]
        assert (coverage==1).all() and np.isfinite(result).all()
        diagnostic=self.reference.copy();diagnostic[:,c]=result
        scores=base.metric(d['truth'][d['validation']],diagnostic)
        return result,float(scores[TARGETS[c]]),float(scores['TOTAL'])


def main():
    global OWNED
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--target',choices=TARGETS,required=True)
    ap.add_argument('--architecture',choices=ARCHITECTURES,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--mode',choices=tuple(MODES),required=True)
    ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    assert a.target=='POR' and a.architecture=='dual'
    out=a.out.resolve();c=TARGETS.index(a.target)
    if (out/'fold_5_report.json').exists():raise RuntimeError('Completed experiment is immutable')
    if (out/'config.json').exists() and not a.resume:raise RuntimeError('Existing run requires explicit resume')
    out.mkdir(parents=True,exist_ok=True);OWNED=out
    torch.set_num_threads(3);print('GPU_ADMISSION',check_slot(min_free_gib=3.),flush=True)
    atomic_json(out/'runtime_status.json',{'status':'running','pid':os.getpid(),'target':a.target,'architecture':a.architecture,'atlas_current_trial':True})
    signal.signal(signal.SIGINT,stop_request);signal.signal(signal.SIGTERM,stop_request)
    start=time.perf_counter();torch.manual_seed(20261010);np.random.seed(20261010);random.seed(20261010)
    # This installed CUDA build has no deterministic linear-interpolation backward.
    # Retain deterministic kernels where supported, record the warning for others.
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
    master=torch.load(SOURCE,map_location='cpu',weights_only=False)
    previous=torch.load(REFERENCE/'readout_checkpoint.pt',map_location='cpu',weights_only=False)
    reference=np.load(REFERENCE/'fold_5_pred.npy');source_report=json.loads((REFERENCE/'fold_5_report.json').read_text())
    data,_cached_master=load_data();del _cached_master
    model=make_model(a.mode,master).cuda()
    evaluator=Evaluator(data,master,c,reference)
    parameter_names={n for n,p in model.named_parameters() if p.requires_grad}
    assert parameter_names and all(p.requires_grad for p in model.parameters())
    fixed={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if n not in parameter_names}
    original_body={n:p.detach().cpu().clone() for n,p in model.backbone_parameters()}
    cfg={'target':a.target,'architecture':a.architecture,'model_module':'champion_scaling_v21','mode':a.mode,'scaling':model.scaling_manifest,
         'screening_fold':5,'scope':'POR champion function-preserving internal-width scaling; PERM/SW frozen',
         'source_checkpoint':str(SOURCE.relative_to(ROOT)),'head_initialization':str((REFERENCE/'readout_checkpoint.pt').relative_to(ROOT)),
         'source_channel_score':source_report['scores'][a.target],'reference_TOTAL':source_report['scores']['TOTAL'],
         'pretraining':'initialized from old multitarget training on same64trainingwells; independent fine tuning, not from scratch',
         'trainable_parameters':sum(p.numel() for p in model.parameters()),'private_backbone_branches':len(model.members),
         'target_only_objective':True,'backbone_trainable':True,'dropout_during_finetuning':'actual unchanged module behavior; see scaling manifest',
         'deterministic_mode':'warn_only; CUDA interpolation backward can be nondeterministic',
         'bitwise_training_reproducibility_claimed':False,'torch_version':str(torch.__version__),
         'epochs':80,'steps_per_epoch':64,'batch_size':64,'seed':20261010,'patience':18,'minimum_epochs':10,
         'backbone_lr':1e-5,'readout_lr':3e-5,'eta_min':1e-6,'weight_decay':.01,'clip_norm':2,
         'loss':'only assigned-target proper hurdle mixture likelihood, fixedQ50, divided by3 for legacy scaling',
         'quality':data['qinfo'],'decoder':'single coverage,256densitybins,800actions+default,inferencebatch32',
         'validation_rows':144665,'training_rows':578891,'no_test_labels':True,'remaining_folds_authorized':False,
         'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('independent_channels_v8.py'),Path(__file__).with_name('champion_scaling_v21.py')]}}
    if a.resume:assert json.loads((out/'config.json').read_text())==cfg
    else:atomic_json(out/'config.json',cfg)
    optimizer=torch.optim.AdamW(model.parameter_groups(),weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,80*64,eta_min=1e-6)
    windows=torch.Generator(device='cuda').manual_seed(20261010)
    epoch=0;prior_seconds=0.;gradient_proof=None
    if a.resume:
        saved=torch.load(out/'last_state.pt',map_location='cpu',weights_only=False);assert saved['config']==cfg
        model.load_state_dict(saved['current_state'],strict=True);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        epoch=saved['epoch'];best_epoch=saved['best_epoch'];best_score=saved['best_score'];best_state=saved['best_state'];best_prediction=saved['best_prediction']
        initial=saved['initial'];history=saved['history'];bad=saved['bad'];prior_seconds=saved['seconds'];gradient_proof=saved['gradient_proof']
        torch.set_rng_state(saved['rng']['torch']);torch.cuda.set_rng_state_all(saved['rng']['cuda']);windows.set_state(saved['rng']['windows'])
        np.random.set_state(saved['rng']['numpy']);random.setstate(saved['rng']['python']);del saved
    else:
        pred,score,total=evaluator(model)
        initial={'target_score':score,'reference_replacement_total':total,'different_target_elements_vs_shared_reference':int(np.count_nonzero(pred!=reference[:,c])),
                 'comparison':'dual is a private copy of the prior target function; single is a different one-expert initial function'}
        atomic_json(out/'initial_verification.json',initial)
        if a.architecture=='dual':assert np.array_equal(pred,reference[:,c]),initial
        best_score=score;best_prediction=pred.copy();best_state=cpu_copy(model.state_dict());best_epoch=0;bad=0;history=[]
        print('INITIAL',json.dumps(initial),flush=True)
    torch.cuda.reset_peak_memory_stats()
    def persist():
        atomic_torch(out/'model_best.pt',{'state_dict':best_state,'config':cfg,'best_epoch':best_epoch,'target_score':best_score})
        atomic_array(out/'channel_pred.npy',best_prediction)
        atomic_torch(out/'last_state.pt',{'config':cfg,'current_state':cpu_copy(model.state_dict()),'optimizer':cpu_copy(optimizer.state_dict()),'scheduler':scheduler.state_dict(),
            'epoch':epoch,'best_epoch':best_epoch,'best_score':best_score,'best_state':best_state,'best_prediction':best_prediction,'initial':initial,'history':history,'bad':bad,
            'gradient_proof':gradient_proof,'seconds':prior_seconds+time.perf_counter()-start,
            'rng':{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'windows':windows.get_state(),'numpy':np.random.get_state(),'python':random.getstate()}})
        atomic_json(out/'validation_history.json',history)
    persist();core=torch.zeros(1,256,device='cuda',dtype=torch.bool);core[:,64:192]=True
    for current in range(epoch+1,81):
        if STOP or (epoch>=10 and bad>=18):break
        begin=time.perf_counter();model.train();stats=np.zeros(3)
        for step in range(64):
            anchors=data['train_raw'][torch.randint(len(data['train_raw']),(64,),device='cuda',generator=windows)]
            ix,valid=window_indices(anchors,data['lower'],data['upper']);relative=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
            mask=valid&core&data['labeled'][ix];optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(data['x'][ix],relative,valid,absolute_depth=data['absolute'][ix])
            loss,bce,nll=channel_objective(output,data['labels'][ix],data['default'][ix],mask,data['quality'][ix],c)
            assert torch.isfinite(loss);loss.backward()
            if current==1 and step==0:
                body=[(n,p) for n,p in model.backbone_parameters() if p.grad is not None]
                assert body and all(torch.isfinite(p.grad).all() for _,p in body)
                norm=float(torch.stack([p.grad.detach().float().square().sum() for _,p in body]).sum().sqrt())
                assert norm>0
                gradient_proof={'private_backbone_gradient_norm':norm,'body_tensors_with_grad':len(body),'target':a.target,'output_target_axis':1,
                    'private_process_pid':os.getpid(),'other_target_losses_computed':False,
                     'extra_output_gradient_norm':float(sum(p.grad.detach().float().square().sum() for n,p in model.named_parameters() if '.extra_' in n and p.grad is not None).sqrt()) if a.mode!='base1' else 0.}
                atomic_json(out/'gradient_scope.json',gradient_proof);print('PRIVATE_BACKBONE_GRADIENT',json.dumps(gradient_proof),flush=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(),2);optimizer.step();scheduler.step()
            stats+=np.array([float(loss.detach()),float(bce.detach()),float(nll.detach())])
        epoch=current;pred,score,total=evaluator(model)
        if score>best_score:
            best_score=score;best_prediction=pred.copy();best_state=cpu_copy(model.state_dict());best_epoch=epoch;bad=0
        else:bad+=1
        row={'epoch':epoch,'target_score':score,'best_target_score':best_score,'best_epoch':best_epoch,'bad_epochs':bad,
            'reference_replacement_total':total,'score_scope':'TOTAL is single-target replacement diagnostic, not a3modelresult',
            'training_loss':stats[0]/64,'training_bce':stats[1]/64,'training_nll':stats[2]/64,
            'learning_rates':[g['lr'] for g in optimizer.param_groups],'seconds':time.perf_counter()-begin}
        history.append(row);persist()
        atomic_json(out/'runtime_status.json',{'status':'running','pid':os.getpid(),'target':a.target,'architecture':a.architecture,'mode':a.mode,'last_completed_epoch':epoch,
                    'best_target_score':best_score,'best_epoch':best_epoch,'atlas_current_trial':True})
        print('EPOCH',json.dumps(row),flush=True)
    if STOP:
        atomic_json(out/'runtime_status.json',{'status':'stopped_by_user','pid':os.getpid(),'last_completed_epoch':epoch,'auto_resume':False,'resume_available':True,'atlas_current_trial':False});return
    trained_body_change=float(sum((p.detach().cpu().double()-original_body[n].double()).square().sum() for n,p in model.backbone_parameters()).sqrt())
    assert trained_body_change>0
    model.load_state_dict(best_state,strict=True);pred,score,total=evaluator(model)
    assert np.array_equal(pred,best_prediction)
    assert all(torch.equal(model.state_dict()[n].detach().cpu(),v) for n,v in fixed.items())
    selected_body_change=float(sum((p.detach().cpu().double()-original_body[n].double()).square().sum() for n,p in model.backbone_parameters()).sqrt())
    report={'fold':5,'screening_fold':5,'target':a.target,'architecture':a.architecture,'scores':{a.target:score},'target_score':score,'mode':a.mode,'scaling':model.scaling_manifest,
         'best_after_update':max(history,key=lambda r:r['target_score']) if history else None,
        'initial_target_score':initial['target_score'],'source_channel_score':cfg['source_channel_score'],
        'delta_initial':score-initial['target_score'],'delta_shared_reference':score-cfg['source_channel_score'],
        'reference_replacement_total':total,'TOTAL_scope':'diagnostic only; actual combined score requires3independentmodels',
        'best_epoch':best_epoch,'stopped_epoch':epoch,'private_backbone_branches':len(model.members),'trainable_parameters':cfg['trainable_parameters'],
        'private_backbone_gradient_verified':True,'last_backbone_parameter_change_L2':trained_body_change,'selected_backbone_parameter_change_L2':selected_body_change,
        'all144665target_predictions_restored_exactly':True,'frozen_packed_rows_unchanged':True,'independent_checkpoint':True,
        'complete_five_folds':False,'remaining_folds_started':False,'seconds':prior_seconds+time.perf_counter()-start,
        'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'finished_at':datetime.now().astimezone().isoformat()}
    atomic_json(out/'fold_5_report.json',report);atomic_json(out/'runtime_status.json',{'status':'complete','pid':os.getpid(),'target':a.target,'architecture':a.architecture,'atlas_current_trial':False})
    print('TARGET_RESULT',json.dumps(report),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OWNED is not None:
            value={'status':'failed','pid':os.getpid(),'error':repr(exc),'traceback':traceback.format_exc(),'auto_resume':False,'atlas_current_trial':False}
            atomic_json(OWNED/'failure.json',value);atomic_json(OWNED/'runtime_status.json',value)
        raise
