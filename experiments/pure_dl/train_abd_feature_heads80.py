"""Frozen A6+B43+D52 form one feature backbone; independent new target heads, no prediction voting."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
import train_invariant as base
from multiscale_sequence import window_indices
from paired_posterior_router_sequence import Network, retained_indices, calibration_indices
from abd_feature_backbone_heads import Network as FeatureHeads
from global_cosine_sequence import Network as SingleNetwork
from frozen_companion_fused_loss import channel_loss
from abd_joint_validation import JointFold5Validation
from gpu_job_guard_v2 import check_slot
ROOT=base.ROOT
SOURCE=ROOT/'outputs/pure_dl_abd_scratch_control80_fold5/fold_5.pt'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();out=args.out
    torch.set_num_threads(3)
    print('GPU_ADMISSION',check_slot(min_free_gib=3.),flush=True)
    if (out/'config.json').exists():raise RuntimeError('Refuse to overwrite existing run')
    out.mkdir(parents=True,exist_ok=True)
    base.write(out/'runtime_status.json',{'status':'running','pid':os.getpid(),'atlas_current_trial':True})
    started=time.perf_counter();torch.manual_seed(20261010);np.random.seed(20261010)
    master=torch.load(SOURCE,map_location='cpu',weights_only=False)
    rec = base.load_split(ROOT.parent/'data', 'train')
    ys = [frame[base.TS].to_numpy(np.float64) for _, frame in rec]
    valid = [np.isfinite(y).all(1) for y in ys]
    truth = np.concatenate([y[m] for y, m in zip(ys, valid)])
    wells = np.concatenate([np.full(m.sum(), i, np.int32) for i, m in enumerate(valid)])
    assert len(rec) == 80 and truth.shape == (723556, 3)
    fold = np.full(len(truth), -1, np.int8)
    for f, (_, ix) in enumerate(GroupKFold(5).split(truth, groups=wells)):
        fold[ix] = f
    validation = np.flatnonzero(fold == 4)
    assert len(validation) == 144665
    row_ids = [np.flatnonzero(wells == i) for i in range(80)]
    tr_wells = [i for i in range(80) if fold[row_ids[i][0]] != 4]
    starts = np.r_[0, np.cumsum([len(frame) for _, frame in rec])]
    nraw = int(starts[-1])
    valid_to_raw = np.concatenate([np.flatnonzero(m)+starts[i] for i, m in enumerate(valid)])
    assert np.array_equal(valid_to_raw[validation], master['validation_raw_rows'])
    raw_to_valid = np.full(nraw, -1, np.int64)
    raw_to_valid[valid_to_raw] = np.arange(len(truth))
    valid_x, builder, fm, fs, _ = base.prepare_features(rec, tr_wells, valid, row_ids)
    assert np.array_equal(fm, master['feature_median'])
    assert np.array_equal(fs, master['feature_scale'])
    contract = base.feature_contract()
    assert contract['whitelist'] == master['selected_feature_names']
    view_indices = retained_indices(contract['whitelist'])
    numeric_indices, conditioning_indices = calibration_indices(contract['whitelist'])
    x = torch.empty(nraw, 206, dtype=torch.float16, device='cuda')
    for i, (_, frame) in enumerate(rec):
        value = builder.transform(frame).enhanced[:, contract['selected_indices']]
        value = np.clip((value-fm)/fs, -8, 8).astype(np.float16)
        x[starts[i]:starts[i+1]] = torch.from_numpy(value).cuda()
    assert torch.equal(x[torch.as_tensor(valid_to_raw, device='cuda')], valid_x)
    del valid_x
    depth = torch.tensor(np.concatenate([frame['DEPTH'].to_numpy(np.float32)
                                         for _, frame in rec]), device='cuda')
    lower = torch.tensor(np.concatenate([np.full(len(frame), starts[i])
                                         for i, (_, frame) in enumerate(rec)]), device='cuda')
    upper = torch.tensor(np.concatenate([np.full(len(frame), starts[i+1])
                                         for i, (_, frame) in enumerate(rec)]), device='cuda')
    absolute = ((depth-master['absolute_depth_median'])/
                master['absolute_depth_scale']).clamp(-8, 8)
    folders=[ROOT/'outputs/pure_dl_abd_scratch_control80_fold5',
             ROOT/'outputs/pure_dl_posterior_three_seed_equal80/seed_20261008',
             ROOT/'outputs/pure_dl_global_cosine_sequence']
    source_models=[];source_checkpoints=[];source_checks=[]
    for i,folder in enumerate(folders):
        cp=torch.load(folder/'fold_5.pt',map_location='cpu',weights_only=False)
        for key in ['feature_median','feature_scale','regression_median','regression_scale','validation_raw_rows']:
            assert np.array_equal(cp[key],master[key]),key
        source=(SingleNetwork() if i==2 else Network(retain_indices=view_indices,numeric_indices=numeric_indices,conditioning_indices=conditioning_indices))
        source.load_state_dict(cp['state_dict'],strict=True);source.cuda().eval().requires_grad_(False)
        source_models.append(source);source_checkpoints.append(cp)
        source_checks.append({'role':['A6','B43','D52'][i],'folder':str(folder.relative_to(ROOT)),
            'sha256':hashlib.sha256((folder/'fold_5.pt').read_bytes()).hexdigest()})
    train_truth=truth[fold!=4];train_defaults=base.dmask(train_truth)
    initial_means=[]
    for j in range(3):
        fit=(base.TRANSFORM[j](train_truth[~train_defaults[:,j],j])-master['regression_median'][j])/master['regression_scale'][j]
        initial_means.append(np.quantile(fit,np.linspace(.05,.95,8)))
    model=FeatureHeads(source_models,np.stack(initial_means),train_defaults.mean(0).clip(.001,.999)).cuda()
    evaluator=JointFold5Validation(x=x,depth=depth,lower=lower,upper=upper,absolute=absolute,starts=starts,
        row_ids=row_ids,raw_to_valid=raw_to_valid,validation=validation,preprocessing={key:master[key] for key in [
        'selected_feature_names','selected_feature_indices','feature_median','feature_scale','regression_median',
        'regression_scale','absolute_depth_median','absolute_depth_scale','validation_raw_rows','centers','density_edges']})
    training=np.flatnonzero(fold!=4);train_raw=torch.as_tensor(valid_to_raw[training],device='cuda')
    labels=torch.zeros(nraw,3,device='cuda');default=torch.zeros(nraw,3,dtype=torch.bool,device='cuda')
    labeled=torch.zeros(nraw,dtype=torch.bool,device='cuda')
    yy=np.column_stack([(base.TRANSFORM[j](truth[training,j])-master['regression_median'][j])/master['regression_scale'][j] for j in range(3)])
    labels[train_raw]=torch.as_tensor(yy,device='cuda',dtype=torch.float32)
    default[train_raw]=torch.as_tensor(base.dmask(truth[training]),device='cuda');labeled[train_raw]=True
    assert not bool(labeled[torch.as_tensor(valid_to_raw[validation],device='cuda')].any())
    # Use identical fixed Q as the completed matched B-only incremental experiment.
    qpath=ROOT/'outputs/pure_dl_abd_scratch_control80_fold5/fold_5_conflict_weights.npz'
    with np.load(qpath) as trace:
        selected=int(np.flatnonzero(trace['epochs']<6)[-1]);qepoch=int(trace['epochs'][selected])
        raw_q=trace['raw_ids'];point_q=trace['weights'][selected].min(axis=1)
        assert np.array_equal(np.sort(raw_q),np.sort(valid_to_raw[training]))
    quality=torch.ones(nraw,device='cuda')
    quality[torch.as_tensor(raw_q,device='cuda')]=torch.as_tensor(point_q,device='cuda')
    cfg={'screening_fold':5,'baseline_total':80.49230853582785,'epochs':80,'steps':64,'batch_size':64,
        'early_stopping':{'maximum_epochs':80,'minimum_epochs':10,'patience':18,'scope':'independent each channel'},
        'source_backbone':str(SOURCE.relative_to(ROOT)),'source_backbone_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'feature_sources':source_checks,'learning_rate':.0006,'eta_min':1e-5,'schedule':'cosine80*64',
        'seed':20261006,'fold_seed':20261010,'trainable_parameters':sum(p.numel() for p in model.heads.parameters()),
        'head_capacity':'three independent640->128GELU->25 MLP heads; gate+8density components per target, random small output initialization; train-only prior biases',
        'trainable_role':'three independent new heads only','backbone_role':'frozen A6+B43+D52 feature extractors,5branches x128features=640','route':'NO original output voting, NO posterior router; normalized intermediate feature concatenation only',
        'loss':'new heads own per-channel NLL, final fixed pointQ; frozen sources provide features only in training and validation; no legacy predictions or auxiliary loss',
        'quality_source_epoch':qepoch,'quality_min':float(point_q.min()),'quality_max':float(point_q.max()),
        'quality_sha256':hashlib.sha256(qpath.read_bytes()).hexdigest(),
        'selection':'each channel selects its new head by its own official score; assemble3fixed best heads and recompute TOTAL; no old-model voting',
        'optimizer':'three AdamW optimizers, per-channel2 clip; no backbone or router optimizer',
        'validation':'same four cosine windows and fine decoder, new predictions only; old80.4923 is reference, not initialization score',
        'source_hashes':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in [
            Path(__file__).name,'abd_feature_backbone_heads.py','frozen_companion_fused_loss.py','abd_joint_validation.py']}}
    base.write(out/'config.json',cfg)
    optimizers=[torch.optim.AdamW(model.heads[j].parameters(),lr=.0006,weight_decay=.01) for j in range(3)]
    schedulers=[torch.optim.lr_scheduler.CosineAnnealingLR(o,80*64,eta_min=1e-5) for o in optimizers]
    def evaluate():
        prediction=evaluator.evaluate(model)
        scores=base.metric(truth[validation],prediction)
        return prediction,scores,scores
    torch.cuda.reset_peak_memory_stats()
    initial_pred,initial_scores,initial_solo=evaluate()
    initial_check={'score_delta':initial_scores['TOTAL']-cfg['baseline_total'],
        'expected_baseline_reproduction':False,'reason':'New feature-fusion MLP heads; old voting graph removed'}
    base.write(out/'initial_verification.json',initial_check)
    best_scores=np.array([initial_scores[c] for c in base.TS]);best_epochs=[0,0,0]
    stopped_epochs=[None,None,None];bad=[0,0,0];active=[True,True,True]
    best_predictions=initial_pred.copy();history=[];best_states=[]
    for j in range(3):best_states.append({k:v.detach().cpu().clone() for k,v in model.heads[j].state_dict().items()})
    def persist():
        torch.save({'head_state_dicts':best_states,'best_epochs':best_epochs.copy(),'channel_scores':best_scores.tolist(),
            'config':cfg,'selection_scope':'three independently selected new heads; all5frozen feature branches referenced by source hashes; no legacy vote'},out/'heads_best.pt')
        np.save(out/'fold_5_pred.npy',best_predictions)
    persist();print('INITIAL_FUSED',json.dumps(initial_scores),flush=True)
    core=torch.zeros(1,256,dtype=torch.bool,device='cuda');core[:,64:192]=True
    for epoch in range(1,81):
        t0=time.perf_counter();model.train();channel_sum=np.zeros(3);bce_sum=np.zeros(3);nll_sum=np.zeros(3)
        epoch_active=active.copy()
        for step in range(64):
            anchors=train_raw[torch.randint(len(train_raw),(64,),device='cuda')]
            ix,exists=window_indices(anchors,lower,upper)
            relative=torch.tanh((depth[ix]-depth[anchors,None])/10);mask=exists & core & labeled[ix]
            for optimizer in optimizers:optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                candidate=model(x[ix],relative,exists,absolute_depth=absolute[ix])
            channels,bce,nll=channel_loss(candidate,labels[ix],default[ix],mask,quality[ix])
            loss=sum(channels[j] for j in range(3) if active[j])/3
            assert torch.isfinite(loss);loss.backward()
            if epoch==1 and step==0:
                assert all(p.grad is None for p in model.backbone.parameters())
                assert all(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in group.parameters()) for group in model.heads)
                print('GRADIENT_SCOPE_VERIFIED new3heads only; frozen5branch feature backbone; NO legacy output fusion',flush=True)
            for j in range(3):
                if active[j]:
                    torch.nn.utils.clip_grad_norm_(model.heads[j].parameters(),2)
                    optimizers[j].step();schedulers[j].step()
            channel_sum+=channels.detach().cpu().numpy();bce_sum+=bce.detach().cpu().numpy();nll_sum+=nll.detach().cpu().numpy()
        pred,scores,solo=evaluate()
        for j,name in enumerate(base.TS):
            if not epoch_active[j]:continue
            if scores[name]>best_scores[j]:
                best_scores[j]=scores[name];best_epochs[j]=epoch;bad[j]=0;best_predictions[:,j]=pred[:,j]
                best_states[j]={k:v.detach().cpu().clone() for k,v in model.heads[j].state_dict().items()}
            else:bad[j]+=1
            if epoch>=10 and bad[j]>=18:
                active[j]=False;stopped_epochs[j]=epoch
                model.heads[j].load_state_dict(best_states[j]);model.heads[j].requires_grad_(False)
        persist()
        item={'epoch':epoch,'official_scores':scores,'official_total':scores['TOTAL'],'solo_scores':solo,
            'channel_train_loss':(channel_sum/64).tolist(),'channel_gate_bce':(bce_sum/64).tolist(),
            'channel_continuous_nll':(nll_sum/64).tolist(),'best_channel_scores':best_scores.tolist(),
            'best_epochs':best_epochs.copy(),'bad_epochs':bad.copy(),'active_next_epoch':active.copy(),
            'best_assembled_total':float(best_scores@np.array([.3,.35,.35])),
            'seconds':time.perf_counter()-t0}
        history.append(item);base.write(out/'fold_5_validation_history.json',history)
        print('EPOCH_VALIDATION',json.dumps(item),flush=True)
        if not any(active):break
    for j in range(3):
        if stopped_epochs[j] is None:stopped_epochs[j]=epoch
        model.heads[j].load_state_dict(best_states[j])
    final_pred,final_scores,final_solo=evaluate()
    assert np.array_equal(final_pred,best_predictions),'Assembled heads must reproduce all selected channel predictions'
    assert np.allclose([final_scores[c] for c in base.TS],best_scores,rtol=0,atol=1e-10)
    for source,cp,item in zip(model.backbone.sources,source_checkpoints,source_checks):
        assert all(torch.equal(v.detach().cpu(),cp['state_dict'][k]) for k,v in source.state_dict().items())
        assert hashlib.sha256((ROOT/item['folder']/'fold_5.pt').read_bytes()).hexdigest()==item['sha256']
    assert all(hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==sha for name,sha in cfg['source_hashes'].items())
    report={'fold':5,'screening_fold':5,'scores':final_scores,'solo_scores':final_solo,'best_epochs':best_epochs,'stopped_epochs':stopped_epochs,
        'initial_scores':initial_scores,'delta_initial':final_scores['TOTAL']-initial_scores['TOTAL'],
        'delta_reference':final_scores['TOTAL']-cfg['baseline_total'],'complete_five_folds':False,
        'assembled_heads_exactly_reproduce_selected_predictions':True,'all_frozen_feature_sources_unchanged':True,'legacy_output_vote':False,
        'all_validation_rows_covered_exactly_four_times':True,'seconds':time.perf_counter()-started,
        'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/1024**2}
    base.write(out/'fold_5_report.json',report)
    base.write(out/'runtime_status.json',{'status':'complete','pid':os.getpid(),'atlas_current_trial':False})
    print('FOLD_RESULT',json.dumps(report),flush=True)


if __name__=='__main__':main()
