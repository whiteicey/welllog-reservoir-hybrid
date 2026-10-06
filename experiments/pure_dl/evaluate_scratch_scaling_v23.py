"""Final inference-only v23 evaluation, including train/validation populations.

No labels enter model inputs or model selection beyond the already fixed
validation checkpoint. Full training scores and router statistics are only
post-training diagnostics, not additional candidates.
"""
from datetime import datetime
import gc,hashlib,json,sys,time
import numpy as np
import torch
from torch.nn import functional as F
from scratch_scaling_v23 import ROOT,STUDY,MODES,REFERENCE,Network,load_data
from train_scaling_v21 import Evaluator
from train_dual8222_heads80 import atomic_json,atomic_array
from multiscale_sequence import window_indices
from por_t3_patch_loss import probability
from launch_latent_gate_v17 import active_jobs
from gpu_job_guard_v2 import check_slot
import train_invariant as base


def point_score(y,p):
    return 100*np.maximum(0,1-np.abs(p-y)/(.08*(np.abs(y)+1e-5)))


class PopulationEvaluator:
    def __init__(self,data,master,ids,reference):
        self.d=data;self.master=master;self.ids=np.asarray(ids,dtype=np.int64)
        original=Evaluator(data,master,0,reference)
        self.actions=original.actions;self.utilities=original.utilities;self.edges=original.edges
        self.local=np.full(len(data['truth']),-1,np.int64);self.local[self.ids]=np.arange(len(self.ids))
        desired=set(np.unique(data['wells'][self.ids]).tolist())
        self.centers=[];self.anchors=[]
        for well,rows in enumerate(data['row_ids']):
            if well not in desired:continue
            cc=np.arange(data['starts'][well],data['starts'][well+1],128)+64
            self.centers.extend(cc);self.anchors.extend(np.minimum(cc,data['starts'][well+1]-1))

    @torch.no_grad()
    def __call__(self,model):
        d=self.d;model.eval()
        pred=np.full(len(self.ids),np.nan,np.float64);coverage=np.zeros(len(pred),np.int8)
        gate_probs=np.zeros(len(pred),np.float32);route=np.zeros((len(pred),2),np.float32)
        responsibilities=np.zeros((len(pred),2),np.float32)
        nll_sum=0.;bce_sum=0.;numeric_count=0
        captured={}
        handles=[model.router[2].register_forward_hook(lambda _m,_x,y:captured.update(logits=y.detach()))]
        for expert,member in enumerate(model.members):
            def capture(_m,_x,y,expert=expert):
                captured['member'+str(expert)]=(y[0].detach(),y[1].detach())
            handles.append(member.register_forward_hook(capture))
        try:
            for begin in range(0,len(self.centers),32):
                centers=torch.tensor(self.centers[begin:begin+32],device='cuda')
                anchors=torch.tensor(self.anchors[begin:begin+32],device='cuda')
                ix,valid=window_indices(centers,d['lower'],d['upper'],anchors=anchors)
                rel=torch.tanh((d['depth'][ix]-d['depth'][anchors,None])/10)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    go,raw=model(d['x'][ix],rel,valid,absolute_depth=d['absolute'][ix])
                global_rows=d['raw_to_valid'][ix[:,64:192].cpu().numpy()]
                use=valid[:,64:192].cpu().numpy()&(global_rows>=0)
                ids=global_rows[use];loc=self.local[ids]
                assert (loc>=0).all() and (coverage[loc]==0).all() and len(np.unique(loc))==len(loc)
                coverage[loc]+=1;take=torch.as_tensor(use,device='cuda')
                logits=go[:,64:192][take][:,0].float()
                g=logits.sigmoid();r=raw[:,64:192][take][:,0].float()
                cp=probability(0,r[:,0].softmax(-1),r[:,1],F.softplus(r[:,2])+.025,self.edges)
                mass=torch.cat((g[:,None],(1-g[:,None])*cp),-1)
                pred[loc]=self.actions[(mass@self.utilities).argmax(-1).cpu().numpy()]
                gate_probs[loc]=g.cpu().numpy()
                routing=captured.pop('logits').reshape(len(ix),256,3,2)[:,:,0].float().softmax(-1)
                route[loc]=routing[:,64:192][take].cpu().numpy()
                truth=d['truth'][ids]
                dd=torch.as_tensor(base.dmask(truth)[:,0],device='cuda')
                z=torch.as_tensor((np.log1p(truth[:,0])-self.master['regression_median'][0])/self.master['regression_scale'][0],device='cuda',dtype=torch.float32)
                sd=F.softplus(r[:,2])+.025
                logp=np.log(2/np.pi)-sd.log()-2*torch.log1p(((z[:,None]-r[:,1])/sd).square())
                nll=-(r[:,0].log_softmax(-1)+logp).logsumexp(-1)
                nll_sum+=float(nll[~dd].sum());numeric_count+=int((~dd).sum())
                bce_sum+=float(F.binary_cross_entropy_with_logits(logits,dd.float(),reduction='sum'))
                # Explanatory label-conditioned responsibilities, computed only
                # after final checkpoint selection; never fed back to training.
                log_members=[]
                for expert in range(2):
                    mg,mr=captured.pop('member'+str(expert))
                    mg=mg[:,64:192][take][:,0].float();mr=mr[:,64:192][take][:,0].float()
                    msd=F.softplus(mr[:,2])+.025
                    lp=np.log(2/np.pi)-msd.log()-2*torch.log1p(((z[:,None]-mr[:,1])/msd).square())
                    log_density=(mr[:,0].log_softmax(-1)+lp).logsumexp(-1)
                    log_joint=torch.where(dd,F.logsigmoid(mg),F.logsigmoid(-mg)+log_density)
                    log_members.append(routing[:,64:192][take][:,expert].clamp_min(1e-30).log()+log_joint)
                responsibilities[loc]=torch.stack(log_members,-1).softmax(-1).cpu().numpy()
        finally:
            for handle in handles:handle.remove()
        assert (coverage==1).all() and np.isfinite(pred).all()
        truth=d['truth'][self.ids];dd=base.dmask(truth)[:,0];wells=d['wells'][self.ids]
        points=point_score(truth[:,0],pred)
        entropy=-(route*np.log(np.maximum(route,1e-30))).sum(-1)
        stats=dict(rows=len(pred),POR=float(points.mean()),default_POR=float(points[dd].mean()),numeric_POR=float(points[~dd].mean()),
            default_points=int(dd.sum()),numeric_points=int((~dd).sum()),BCE=bce_sum/len(pred),conditional_NLL=nll_sum/max(numeric_count,1),
            gate_default_at_half_fraction=float((gate_probs>=.5).mean()),
            input_router=dict(mean_weights=route.mean(0,dtype=np.float64).tolist(),mean_entropy_nats=float(entropy.mean()),
                max_weight_over_095_fraction=float((route.max(-1)>.95).mean()),
                meaning='input router weights; not posterior responsibilities or proof of expert contribution'),
            posterior_responsibility=dict(nondefault_mean_weights=responsibilities[~dd].mean(0,dtype=np.float64).tolist(),
                nondefault_max_over_095_fraction=float((responsibilities[~dd].max(-1)>.95).mean()),
                nondefault_mean_entropy_nats=float(-(responsibilities[~dd]*np.log(np.maximum(responsibilities[~dd],1e-30))).sum(-1).mean()),
                default_mean_weights=responsibilities[dd].mean(0,dtype=np.float64).tolist(),
                scope='posthoc final-checkpoint label-conditioned diagnostic; no additional model selection'),
            per_well=[dict(well=int(w),n=int((wells==w).sum()),POR=float(points[wells==w].mean())) for w in np.unique(wells)])
        return pred,stats


def paired(y,left,right,defaults,wells):
    diff=point_score(y,right)-point_score(y,left)
    pw=[dict(well=int(w),n=int((wells==w).sum()),POR_delta=float(diff[wells==w].mean())) for w in np.unique(wells)]
    counts=np.array([r['n'] for r in pw]);delta=np.array([r['POR_delta'] for r in pw])
    sample=np.random.default_rng(20261010).integers(0,len(pw),size=(10000,len(pw)))
    boot=.3*(delta[sample]*counts[sample]).sum(1)/counts[sample].sum(1)
    return dict(POR_delta=float(diff.mean()),TOTAL_delta=float(.3*diff.mean()),
        default_POR_delta=float(diff[defaults].mean()),numeric_POR_delta=float(diff[~defaults].mean()),
        default_TOTAL_contribution=float(.3*diff[defaults].sum()/len(diff)),numeric_TOTAL_contribution=float(.3*diff[~defaults].sum()/len(diff)),
        well_wins=sum(r['POR_delta']>0 for r in pw),well_losses=sum(r['POR_delta']<0 for r in pw),per_well=pw,
        exploratory_selection_unadjusted_TOTAL_95=np.quantile(boot,[.025,.975]).tolist())


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if active_jobs():raise RuntimeError('Final GPU inference waits for trainers to finish')
    for mode in MODES:
        if not (ROOT/'outputs'/f'pure_dl_scratch_scaling_v23_{mode}_fold5/fold_5_report.json').exists():raise RuntimeError('Incomplete '+mode)
    torch.set_num_threads(3);check_slot(min_free_gib=3.)
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
    start=time.perf_counter();data,master=load_data();reference=np.load(REFERENCE/'fold_5_pred.npy')
    ids=data['validation'];truth=data['truth'][ids];wells=data['wells'][ids];defaults=base.dmask(truth)[:,0]
    assert truth.shape==reference.shape==(144665,3)
    with np.load(REFERENCE/'oof.npz') as z:
        assert np.array_equal(truth,z['truth'][z['fold']==4]) and np.array_equal(wells,z['well'][z['fold']==4])
    reference_scores=base.metric(truth,reference)
    val=PopulationEvaluator(data,master,ids,reference)
    train=PopulationEvaluator(data,master,np.flatnonzero(data['fold']!=4),reference)
    candidates=[];predictions={}
    for mode in MODES:
        out=ROOT/'outputs'/f'pure_dl_scratch_scaling_v23_{mode}_fold5'
        cp=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False)
        report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8'))
        for name,sha in cp['config']['source_hashes'].items():
            assert hashlib.sha256((ROOT/'experiments/pure_dl'/name).read_bytes()).hexdigest()==sha,name
        # Only this newly-trained checkpoint is loaded into the random builder.
        model=Network(cp['metadata'],mode).cuda();model.load_state_dict(cp['state_dict'],strict=True)
        tick=time.perf_counter();prediction,validation=val(model)
        assert np.array_equal(prediction,np.load(out/'channel_pred.npy'))
        assert abs(validation['POR']-report['target_score'])<1e-8
        _,fit=train(model)
        combined=reference.copy();combined[:,0]=prediction
        assert np.array_equal(combined[:,1:],reference[:,1:])
        scores=base.metric(truth,combined)
        assert abs(scores['TOTAL']-report['reference_replacement_total'])<1e-8
        atomic_array(STUDY/f'assembled_{mode}_pred.npy',combined)
        c=dict(mode=mode,scores=scores,delta_reference_TOTAL=scores['TOTAL']-reference_scores['TOTAL'],
            parameter_count=report['trainable_parameters'],best_epoch=report['best_epoch'],stopped_epoch=report['stopped_epoch'],
            validation=validation,fit=fit,comparison_to_champion=paired(truth[:,0],reference[:,0],prediction,defaults,wells),
            all144665_predictions_disk_reloaded_exactly=True,fixed_PERM_SW_arrays_reused=True,fixed_PERM_SW_fresh_GPU_replay=False,
            independent_target_parameters=True,original_validation_row_order_verified=True,
            inference_train_and_validation_seconds=time.perf_counter()-tick,
            checkpoint=str((out/'model_best.pt').relative_to(ROOT)))
        candidates.append(c);predictions[mode]=prediction
        print('VERIFIED',json.dumps({k:v for k,v in c.items() if k not in ('comparison_to_champion','fit','validation')},ensure_ascii=False),flush=True)
        print('POPULATIONS',mode,json.dumps({'fit':{k:v for k,v in fit.items() if k!='per_well'},'validation':{k:v for k,v in validation.items() if k!='per_well'}},ensure_ascii=False),flush=True)
        del model,cp;gc.collect();torch.cuda.empty_cache()
    comparison=paired(truth[:,0],predictions['base1'],predictions['ffn2'],defaults,wells)
    best=max(candidates,key=lambda r:r['scores']['TOTAL']);adopt=best['scores']['TOTAL']>reference_scores['TOTAL']+1e-9
    prior=json.loads((ROOT/'results/deep_learning/frozen_growth_v22_20260928/CURRENT_RECIPE.json').read_text(encoding='utf-8'))
    choices=[dict(c) for c in prior['choices']]
    if adopt:
        for c in choices:
            if c['target']=='POR':c.update(key='POR_v23_'+best['mode'],architecture='private_scratch_dual',mode=best['mode'],
                checkpoint=best['checkpoint'],model_module='scratch_scaling_v23',builder='Network(checkpoint[metadata],mode)',
                target_score=best['scores']['POR'],best_epoch=best['best_epoch'])
    selected=best['scores'] if adopt else reference_scores
    assembled=dict(fold=5,rows=144665,reference_scores=reference_scores,candidates=candidates,capacity_comparison=comparison,
        selected_scores=selected,selected_new_POR=adopt,all_new_checkpoints_disk_replayed=True,old_other_targets_reused=True,
        old_target_replay_provenance='results/deep_learning/latent_gate_v17_20260928/assembled_report.json',
        complete_five_folds=False,remaining_folds_started=False,seconds=time.perf_counter()-start,finished_at=datetime.now().astimezone().isoformat())
    recipe=dict(scores=selected,choices=choices,selected_new_POR=adopt,best_v23_candidate=best['mode'],
        source_preprocessing='results/deep_learning/champion_scaling_v21_20260928/data_manifest.json',
        complete_five_folds=False,remaining_folds_started=False,old_delivery_unchanged=True)
    atomic_json(STUDY/'assembled_report.json',assembled);atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    print('CAPACITY',json.dumps({k:v for k,v in comparison.items() if k!='per_well'},ensure_ascii=False))
    print('SELECTED',json.dumps(recipe,ensure_ascii=False))


if __name__=='__main__':main()
