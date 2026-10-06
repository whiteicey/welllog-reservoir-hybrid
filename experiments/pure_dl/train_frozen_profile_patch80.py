"""Fine-grained slow vote-router training including the equal-vote initial state."""
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.model_selection import GroupKFold

import train_invariant as base
from gpu_job_guard_v2 import check_slot
from frozen_profile_patch import NumericPatch, apply_patch_values, profile_descriptors
from multiscale_sequence import window_indices
from paired_posterior_router_sequence import Network, retained_indices, calibration_indices
from por_t3_patch_loss import components, probability
from posterior_vote_router import posterior_summary
from paired_calibrated_view_sequence import Network as CalibratedNetwork
from global_cosine_sequence import Network as SingleNetwork

class VoteRouter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits=torch.nn.Parameter(torch.zeros(3,5))
    def forward(self, summaries):
        return self.logits.softmax(-1)[None].expand(len(summaries),-1,-1)



ROOT = base.ROOT
OUT = ROOT/'outputs/pure_dl_frozen_profile_patch80_fold5'
SOURCES = [
    (20261006, ROOT/'outputs/pure_dl_paired_posterior_router80_sequence'),
    (20261007, ROOT/'outputs/pure_dl_posterior_three_seed_equal80/seed_20261007'),
    (20261008, ROOT/'outputs/pure_dl_posterior_three_seed_equal80/seed_20261008'),
    (20261006, ROOT/'outputs/pure_dl_paired_calibrated_view_sequence'),
    (20261006, ROOT/'outputs/pure_dl_global_cosine_sequence'),
]


def main():
    torch.set_num_threads(3)
    started = time.perf_counter()
    print('GPU_ADMISSION', check_slot(), flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    checkpoints = []
    for seed, folder in SOURCES:
        path = folder/'fold_5.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert checkpoint['config']['seed'] == seed
        assert json.loads((folder/'fold_5_report.json').read_text(encoding='utf-8'))['best_checkpoint_predictions_reproduced']
        checkpoints.append((seed, checkpoint, hashlib.sha256(path.read_bytes()).hexdigest()))
    master = checkpoints[0][1]
    for _, checkpoint, _ in checkpoints[1:]:
        for field in ['selected_feature_names', 'selected_feature_indices',
                      'feature_median', 'feature_scale', 'regression_median',
                      'regression_scale', 'absolute_depth_median', 'absolute_depth_scale']:
            assert np.array_equal(checkpoint[field], master[field]), field
        assert np.array_equal(checkpoint['validation_raw_rows'], master['validation_raw_rows'])
    rec = base.load_split(ROOT.parent/'data', 'train')
    ys = [frame[base.TS].to_numpy(np.float64) for _, frame in rec]
    valid = [np.isfinite(y).all(1) for y in ys]
    truth = np.concatenate([y[m] for y, m in zip(ys, valid)])
    wells = np.concatenate([np.full(m.sum(), i, np.int32) for i, m in enumerate(valid)])
    assert len(rec) == 80 and truth.shape == (723556, 3)
    fold = np.full(len(truth), -1, np.int8)
    for f, (_, ix) in enumerate(GroupKFold(5).split(truth, groups=wells)):
        fold[ix] = f
    training = np.flatnonzero(fold != 4)
    validation = np.flatnonzero(fold == 4)
    assert len(validation) == 144665
    row_ids = [np.flatnonzero(wells == i) for i in range(80)]
    tr_wells = [i for i in range(80) if fold[row_ids[i][0]] != 4]
    assert len(tr_wells) == 64
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
    models = []
    for member_index, (_, checkpoint, _) in enumerate(checkpoints):
        cls=CalibratedNetwork if member_index==3 else Network
        model=(SingleNetwork() if member_index==4 else cls(retain_indices=view_indices, numeric_indices=numeric_indices,
                        conditioning_indices=conditioning_indices)).cuda()
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.requires_grad_(False)
        models.append(model.eval())
    reference = ROOT/'outputs/pure_dl_five_learned_slow_vote_fold5'
    router_state = torch.load(reference/'router.pt',map_location='cpu',weights_only=False)
    vote = router_state['state_dict']['logits'].float().softmax(-1).cuda()
    baseline = np.full_like(truth,np.nan)
    baseline[validation] = np.load(reference/'fold_5_pred.npy')
    initial_scores = base.metric(truth[validation],baseline[validation])
    assert abs(initial_scores['TOTAL']-80.38689192147532)<1e-9
    actions, utilities, edges = [], [], []
    for j,name in enumerate(base.TS):
        ct=np.asarray(master['centers'][j])
        action=np.linspace(ct[0],ct[-1],800) if j==2 else np.geomspace(max(ct[0]*.9,.001),ct[-1]*1.1,800)
        fine=np.linspace(ct[0],ct[-1],3197) if j==2 else np.geomspace(max(ct[0]*.9,.001),ct[-1]*1.1,3197)
        action=np.unique(np.r_[base.DEFAULTS[name],np.clip(action,*base.BOUNDS[name]),np.clip(fine,*base.BOUNDS[name])])
        support=np.linspace(base.TRANSFORM[j](ct[0]),base.TRANSFORM[j](ct[-1]),1024)
        boundaries=(np.r_[-np.inf,(support[:-1]+support[1:])/2,np.inf]-master['regression_median'][j])/master['regression_scale'][j]
        actions.append(action)
        utilities.append(torch.tensor(base.kernel_matrix(action,np.r_[base.DEFAULTS[name],base.INVERSE[j](support)],name).T,device='cuda',dtype=torch.float32))
        edges.append(torch.tensor(boundaries,device='cuda',dtype=torch.float32))
    cache=ROOT/'outputs/pure_dl_frozen_numeric_patch80_fold5/frozen_training_predictions.npy'
    assert cache.exists(), 'Reuse the verified frozen prediction cache; do not regenerate'
    if cache.exists():
        cached=np.load(cache)
        assert cached.shape==baseline.shape and np.isfinite(cached).all()
        assert np.array_equal(cached[validation],baseline[validation])
        manifest=json.loads(cache.with_name('cache_manifest.json').read_text())
        assert manifest['checkpoint_sha256']==[sha for _,_,sha in checkpoints]
        assert manifest['router_sha256']==hashlib.sha256((reference/'router.pt').read_bytes()).hexdigest()
        baseline=cached
    else:
        core_weights=torch.sin(torch.pi*(torch.arange(128,device='cuda')+.5)/128).square()
        with torch.no_grad():
            for wi in tr_wells:
                ids_well=row_ids[wi]; local=np.full(len(truth),-1,np.int64);local[ids_well]=np.arange(len(ids_well))
                joint=torch.zeros((3,len(ids_well),5,1025),dtype=torch.float32)
                coverage=np.zeros(len(ids_well),np.int8)
                cc_all=np.arange(starts[wi]-32,starts[wi+1]+64,32)
                for begin in range(0,len(cc_all),32):
                    cc=torch.tensor(cc_all[begin:begin+32],device='cuda')
                    anchor=cc.clamp(int(starts[wi]),int(starts[wi+1]-1))
                    ix,exists=window_indices(cc,lower,upper,anchors=anchor)
                    relative=torch.tanh((depth[ix]-depth[anchor,None])/10)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        outputs=[m(x[ix],relative,exists,absolute_depth=absolute[ix]) for m in models]
                    raw_rows=ix[:,64:192].cpu().numpy(); rows=raw_to_valid[raw_rows]
                    use=exists[:,64:192].cpu().numpy() & (rows>=0)
                    loc=local[rows[use]];assert (loc>=0).all()
                    np.add.at(coverage,loc,1)
                    select=torch.from_numpy(use).cuda();loc_cpu=torch.as_tensor(loc)
                    pw=core_weights[None].expand(len(cc),-1)[select]/2
                    for m,(gate,raw) in enumerate(outputs):
                        g=gate[:,64:192][select].float().sigmoid();r=raw[:,64:192][select].float()
                        for j in range(3):
                            cp=probability(j,r[:,j,0].softmax(-1),r[:,j,1],F.softplus(r[:,j,2])+.025,edges[j])
                            value=torch.cat((g[:,j:j+1],(1-g[:,j:j+1])*cp),-1)
                            joint[j,:,m].index_add_(0,loc_cpu,(value*pw[:,None]).cpu())
                assert (coverage==4).all()
                for j in range(3):
                    for begin in range(0,len(ids_well),2048):
                        end=min(begin+2048,len(ids_well))
                        mixed=(joint[j,begin:end].cuda()*vote[j,:,None]).sum(1)
                        baseline[ids_well[begin:end],j]=actions[j][(mixed@utilities[j]).argmax(-1).cpu().numpy()]
                del joint
                print('CACHE_WELL',wi,'points',len(ids_well),'seconds',round(time.perf_counter()-started,1),flush=True)
        assert np.isfinite(baseline).all()
        base.write(OUT/'cache_manifest.json',{'checkpoint_sha256':[sha for _,_,sha in checkpoints],'router_sha256':hashlib.sha256((reference/'router.pt').read_bytes()).hexdigest(),'training_rows':len(training),'validation_rows':len(validation),'recipe':'frozen learned weights,1024bins,fine actions,4cosine windows; validation cached verified reference'})
        np.save(cache,baseline)
    features=x[torch.as_tensor(valid_to_raw,device='cuda')].float()
    bz=np.column_stack([base.TRANSFORM[j](baseline[:,j]) for j in range(3)])
    bz=(bz-np.asarray(master['regression_median']))/np.asarray(master['regression_scale'])
    fixed_default=base.dmask(baseline)
    features=torch.cat((features,torch.tensor(bz,device='cuda',dtype=torch.float32).clamp(-8,8),torch.tensor(fixed_default,device='cuda',dtype=torch.float32)),1)
    descriptors=profile_descriptors(baseline,fixed_default,row_ids,valid)
    features=torch.cat((features,torch.tensor(descriptors,device='cuda')),1)
    del models,x,depth,lower,upper,absolute,utilities,checkpoints
    gc.collect();torch.cuda.empty_cache()
    target=torch.tensor(truth,device='cuda',dtype=torch.float32)
    original=torch.tensor(baseline,device='cuda',dtype=torch.float32)
    fixed=torch.tensor(fixed_default,device='cuda')
    usable=torch.tensor(~(fixed_default|base.dmask(truth)),device='cuda')
    train_index=torch.tensor(training,device='cuda')
    val_index=torch.tensor(validation,device='cuda')
    q=np.ones(len(truth),np.float32)
    qpath=SOURCES[0][1]/'fold_5_global_point_weights.npz'
    if qpath.exists():
        with np.load(qpath) as pack:
            qids=raw_to_valid[pack['raw_ids']];assert (qids>=0).all() and (fold[qids]!=4).all()
            q[qids]=pack['weight']
    point_q=torch.tensor(q,device='cuda')
    torch.manual_seed(20261103)
    patch=NumericPatch().cuda()
    opt=torch.optim.AdamW(patch.parameters(),lr=1e-4,weight_decay=.01)
    batch_size=8192
    steps=(len(training)+batch_size-1)//batch_size
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,80*steps,eta_min=1e-5)
    best=initial_scores['TOTAL'];best_epoch=0;bad=0
    best_state={k:v.detach().cpu().clone() for k,v in patch.state_dict().items()}
    best_prediction=baseline[validation].copy();history=[]
    def evaluate():
        patch.eval();result=baseline[validation].copy()
        with torch.no_grad():
            for begin in range(0,len(validation),8192):
                ids=val_index[begin:begin+8192]
                delta=patch(features[ids]).double()
                values=apply_patch_values(torch.tensor(baseline[validation[begin:begin+8192]],device='cuda'),delta,fixed[ids])
                for j,name in enumerate(base.TS):values[:,j].clamp_(*base.BOUNDS[name])
                result[begin:begin+len(ids)]=values.cpu().numpy()
        return base.metric(truth[validation],result),result
    scores,prediction=evaluate()
    assert np.array_equal(prediction,best_prediction),'Zero patch must preserve exact80.39 predictions'
    print('INITIAL_PATCH',json.dumps(scores),flush=True)
    base.write(OUT/'config.json',{'screening_fold':5,'initial_scores':scores,'frozen_sources':[str(p.relative_to(ROOT)) for _,p in SOURCES],'parameters':sum(p.numel() for p in patch.parameters()),'input':218,'profile_windows':[9,33],'mechanism':'signed learned coefficients times normalized local frozen-prediction contrasts','hidden':32,'limits':[.04,.03,.025],'loss':'Q-weighted official clipped error with channel weights .30/.35/.35; same nondefault mask as smoothL1 control','lr':1e-4,'batch_size':batch_size,'max_epochs':80,'patience':18,'minimum_epochs':10,'no_validation_gradients':True,'default_decisions_fixed':True,'test_access':False})
    for epoch in range(1,81):
        patch.train();perm=train_index[torch.randperm(len(training),device='cuda')];total=0.
        for begin in range(0,len(perm),batch_size):
            ids=perm[begin:begin+batch_size];opt.zero_grad(set_to_none=True)
            values=apply_patch_values(original[ids],patch(features[ids]),fixed[ids])
            y=target[ids]
            bounded=torch.stack([values[:,j].clamp(*base.BOUNDS[name]) for j,name in enumerate(base.TS)],-1)
            error=torch.stack(((bounded[:,0]-y[:,0]).abs()/(.08*(y[:,0].abs()+1e-5)),
                               (bounded[:,1]/y[:,1].clamp_min(1e-5)).clamp_min(1e-5).log10().abs(),
                               (bounded[:,2]-y[:,2]).abs()/(.05*(y[:,2].abs()+1e-5))),-1)
            channel_weight=error.new_tensor([.3,.35,.35])
            point=(error.clamp(max=1).masked_fill(~usable[ids],0)*channel_weight).sum(-1)
            loss=(point*point_q[ids]).mean();assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(patch.parameters(),1);opt.step();schedule.step()
            total+=float(loss.detach())*len(ids)
        scores,prediction=evaluate()
        if scores['TOTAL']>best:
            best=scores['TOTAL'];best_epoch=epoch;bad=0
            best_state={k:v.detach().cpu().clone() for k,v in patch.state_dict().items()};best_prediction=prediction.copy()
        else:bad+=1
        row={'epoch':epoch,'loss':total/len(training),'scores':scores,'best_score':best,'best_epoch':best_epoch,'bad_epochs':bad}
        history.append(row);base.write(OUT/'validation_history.json',history)
        print('PATCH_EPOCH',json.dumps(row),flush=True)
        if epoch>=10 and bad>=18:break
    patch.load_state_dict(best_state);scores,prediction=evaluate();assert np.array_equal(prediction,best_prediction)
    result={'screening_fold':5,'scores':scores,'initial_scores':initial_scores,'gain':best-initial_scores['TOTAL'],'best_epoch':best_epoch,'stopped_epoch':epoch,'best_checkpoint_predictions_reproduced':True,'screening_passed':best>80.4,'training_points':len(training),'validation_points':len(validation),'validation_points_used_for_gradient':0,'wall_seconds':time.perf_counter()-started,'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'checkpoint_persisted':best>initial_scores['TOTAL']}
    if result['checkpoint_persisted']:torch.save({'state_dict':best_state,'report':result},OUT/'patch.pt')
    np.save(OUT/'fold_5_pred.npy',prediction);base.write(OUT/'report.json',result)
    print('PATCH_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
