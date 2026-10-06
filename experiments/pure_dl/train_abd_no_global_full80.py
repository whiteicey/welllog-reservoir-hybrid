"""Equal predictive-distribution vote of three independently trained neural models."""
import math
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.model_selection import GroupKFold

import train_invariant as base
from multiscale_sequence import window_indices
from paired_posterior_router_sequence import Network, retained_indices, calibration_indices
from por_t3_patch_loss import probability, components
from gpu_job_guard_v2 import check_slot
from abd_module_intervention import disable_full_global_attention
from paired_calibrated_view_sequence import Network as CalibratedNetwork
from global_cosine_sequence import Network as SingleNetwork


ROOT = base.ROOT
SOURCE_OUT = ROOT/'outputs/pure_dl_posterior_three_seed_equal80'
OUT = ROOT/'outputs/pure_dl_abd_no_global_full80_fold5'
SOURCES = [
    (20261006, ROOT/'outputs/pure_dl_paired_posterior_router80_sequence'),
    (20261008, SOURCE_OUT/'seed_20261008'),
    (20261006, ROOT/'outputs/pure_dl_global_cosine_sequence'),
]


def main():
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    admission = check_slot()
    print('GPU_ADMISSION', admission, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'fold_5.pt').exists():
        raise RuntimeError('Refuse to overwrite an existing run')
    base.write(OUT/'runtime_status.json', {'status':'running','atlas_current_trial':True,'pid':os.getpid()})
    torch.manual_seed(20261010)
    np.random.seed(20261010)
    started = time.perf_counter()
    checkpoints = []
    for seed, folder in SOURCES:
        path = folder/'fold_5.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert checkpoint['config']['seed'] == seed
        report = json.loads((folder/'fold_5_report.json').read_text(encoding='utf-8'))
        assert report['best_checkpoint_predictions_reproduced']
        assert abs(report['scores']['TOTAL']-checkpoint['best_validation_score']) < 1e-10
        checkpoints.append((seed, folder, checkpoint, hashlib.sha256(path.read_bytes()).hexdigest()))
    master = checkpoints[0][2]
    for _, _, checkpoint, _ in checkpoints[1:]:
        for field in ['selected_feature_names', 'selected_feature_indices',
                      'feature_median', 'feature_scale', 'regression_median',
                      'regression_scale', 'absolute_depth_median', 'absolute_depth_scale']:
            assert np.array_equal(checkpoint[field], master[field]), field
        for field in ['centers', 'density_edges']:
            for left, right in zip(checkpoint[field], master[field]):
                assert np.array_equal(left, right), field
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
    models = []
    for seed, folder, checkpoint, sha in checkpoints:
        model_class = CalibratedNetwork if 'calibrated_view' in folder.name else Network
        model = (SingleNetwork() if folder.name == 'pure_dl_global_cosine_sequence' else model_class(retain_indices=view_indices, numeric_indices=numeric_indices,
                        conditioning_indices=conditioning_indices)).cuda()
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        models.append(model.eval())
    actions, utilities, edges = [], [], []
    for j, name in enumerate(base.TS):
        ct = np.asarray(master['centers'][j])
        action = np.linspace(ct[0], ct[-1], 800) if j == 2 else np.geomspace(max(ct[0]*.9, .001), ct[-1]*1.1, 800)
        fine_action = np.linspace(ct[0], ct[-1], 3197) if j == 2 else np.geomspace(max(ct[0]*.9, .001), ct[-1]*1.1, 3197)
        action = np.unique(np.r_[base.DEFAULTS[name], np.clip(action, *base.BOUNDS[name]), np.clip(fine_action, *base.BOUNDS[name])])
        actions.append(action)
        transformed_support = np.linspace(base.TRANSFORM[j](ct[0]), base.TRANSFORM[j](ct[-1]), 1024)
        fine_centers = base.INVERSE[j](transformed_support)
        fine_edges = (np.r_[-np.inf, (transformed_support[:-1]+transformed_support[1:])/2, np.inf]-master['regression_median'][j])/master['regression_scale'][j]
        utilities.append(torch.tensor(base.kernel_matrix(action,
                          np.r_[base.DEFAULTS[name], fine_centers], name).T,
                          device='cuda', dtype=torch.float32))
        edges.append(torch.tensor(fine_edges, device='cuda', dtype=torch.float32))
    # B and D stay in eval mode and never enter the optimizer.
    candidate = models[0]
    disable_full_global_attention(candidate)
    teachers = models[1:]
    for teacher in teachers:
        teacher.requires_grad_(False)
    training = np.flatnonzero(fold != 4)
    train_raw = torch.as_tensor(valid_to_raw[training], device='cuda')
    labels = torch.zeros(nraw, 3, device='cuda')
    default = torch.zeros(nraw, 3, dtype=torch.bool, device='cuda')
    labeled = torch.zeros(nraw, dtype=torch.bool, device='cuda')
    targets = np.column_stack([(base.TRANSFORM[j](truth[training,j])-master['regression_median'][j])/master['regression_scale'][j] for j in range(3)])
    labels[train_raw] = torch.as_tensor(targets, dtype=torch.float32, device='cuda')
    default[train_raw] = torch.as_tensor(base.dmask(truth[training]), device='cuda')
    labeled[train_raw] = True
    assert not labeled[torch.as_tensor(valid_to_raw[validation], device='cuda')].any()
    qpath = SOURCES[0][1]/'fold_5_global_point_weights.npz'
    with np.load(qpath) as pack:
        qids = pack['raw_ids']
        assert np.array_equal(np.sort(qids), np.sort(valid_to_raw[training]))
        qvalues = pack['weight'].copy()
        qepoch = np.unique(pack['last_epoch']).tolist()
    quality = torch.ones(nraw, device='cuda')
    quality[torch.as_tensor(qids, device='cuda')] = torch.as_tensor(qvalues, device='cuda')
    cfg = {'seed':20261006, 'screening_fold':5, 'epochs':80, 'steps':64, 'batch_size':64,
           'early_stopping':{'maximum_epochs':80,'minimum_epochs':10,'patience':18},
           'baseline_total':80.40915186344968, 'ensemble':'A+B+D equal predictive distributions',
           'trainable':'A remaining parameters; warm start original best checkpoint; full206 global attention removed',
           'module_ablation':'A.members[0].stem.hierarchy.global_patches output zero; structurally replaced',
           'architecture_loader':'abd_module_intervention.disable_full_global_attention before state_dict load',
           'numerical_execution':'deterministic algorithms and CUBLAS_WORKSPACE_CONFIG=:4096:8; unlike original incremental control',
           'frozen':['B seed20261008 posterior dual','D global cosine single'],
           'loss':'Q times mean over channels of negative log equal three-source joint likelihood',
           'training_context':'all three models use the same sampled 256-point window; central128 supervised',
           'validation':'four cosine windows;1024 bins;original800 union3197 actions',
           'selection':'fused official TOTAL only; solo diagnostic; always save best including epoch0',
           'learning_rate':3e-5,'schedule':'cosine','eta_min':1e-5,'weight_decay':.01,
           'quality':'frozen original A final train-only Q table, not best-epoch Q; no auxiliary restart',
           'quality_last_epochs':qepoch,'quality_sha256':hashlib.sha256(qpath.read_bytes()).hexdigest(),
           'source_checks':[{'folder':str(folder.relative_to(ROOT)), 'sha256':sha} for _,folder,_,sha in checkpoints]}
    base.write(OUT/'config.json',cfg)
    local = np.full(len(truth), -1, np.int64)
    local[validation] = np.arange(len(validation))
    core_weights = torch.sin(torch.pi*(torch.arange(128,device='cuda')+.5)/128).square()
    val_wells = [wi for wi in range(80) if fold[row_ids[wi][0]] == 4]

    @torch.no_grad()
    def well_joint(model, wi):
        model.eval()
        rows = row_ids[wi]
        offset = int(rows[0])
        joint_sum = torch.zeros(len(rows),3,1025,device='cuda')
        coverage = np.zeros(len(rows),np.int8)
        weight_sum = torch.zeros(len(rows),device='cuda')
        cc_all = np.arange(starts[wi]-32, starts[wi+1]+64,32)
        for begin in range(0,len(cc_all),32):
            cc = torch.as_tensor(cc_all[begin:begin+32],device='cuda')
            anchor = cc.clamp(int(starts[wi]),int(starts[wi+1]-1))
            ix,exists = window_indices(cc,lower,upper,anchors=anchor)
            relative = torch.tanh((depth[ix]-depth[anchor,None])/10)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                go,raw = model(x[ix],relative,exists,absolute_depth=absolute[ix])
            ids = raw_to_valid[ix[:,64:192].cpu().numpy()]
            use = exists[:,64:192].cpu().numpy() & (ids>=0)
            loc = ids[use]-offset
            assert ((loc>=0)&(loc<len(rows))).all()
            np.add.at(coverage,loc,1)
            loc_t = torch.as_tensor(loc,device='cuda')
            select = torch.as_tensor(use,device='cuda')
            ww = core_weights[None,:].expand(len(cc),-1)[select]
            weight_sum.index_add_(0,loc_t,ww)
            gate = go[:,64:192][select].float().sigmoid()
            rr = raw[:,64:192][select].float()
            for j in range(3):
                cp = probability(j,rr[:,j,0].softmax(-1),rr[:,j,1],F.softplus(rr[:,j,2])+.025,edges[j])
                joint = torch.cat([gate[:,j:j+1],(1-gate[:,j:j+1])*cp],-1)
                joint_sum[:,j].index_add_(0,loc_t,joint*ww[:,None])
        assert (coverage==4).all()
        assert torch.allclose(weight_sum,torch.full_like(weight_sum,2.),atol=1e-5)
        return joint_sum/weight_sum[:,None,None]

    # Only frozen validation distributions are cached, on CPU (~1.7 GiB).
    # No validation target enters training labels or training likelihood.
    frozen_joint = {}
    for wi in val_wells:
        frozen_joint[wi] = (well_joint(teachers[0],wi)+well_joint(teachers[1],wi)).cpu()
    print('FROZEN_VALIDATION_CACHE_READY', flush=True)

    @torch.no_grad()
    def evaluate():
        fused_pred = np.empty((len(validation),3),np.float64)
        solo_pred = np.empty_like(fused_pred)
        for wi in val_wells:
            current = well_joint(candidate,wi)
            loc = local[row_ids[wi]]
            for begin in range(0,len(loc),1024):
                end = min(begin+1024,len(loc))
                aa = current[begin:end]
                mixed = (aa+frozen_joint[wi][begin:end].cuda())/3
                for j in range(3):
                    fused_pred[loc[begin:end],j] = actions[j][(mixed[:,j]@utilities[j]).argmax(-1).cpu().numpy()]
                    solo_pred[loc[begin:end],j] = actions[j][(aa[:,j]@utilities[j]).argmax(-1).cpu().numpy()]
        assert np.isfinite(fused_pred).all()
        return fused_pred,base.metric(truth[validation],fused_pred),base.metric(truth[validation],solo_pred)

    def log_likelihood(output, mask, yy, dd):
        gg,rr = output
        gg,rr = gg[mask].float(),rr[mask].float()
        conditional = (rr[:,:,0].log_softmax(-1)+components(rr,yy.masked_fill(dd,0))).logsumexp(-1)
        return torch.where(dd,F.logsigmoid(gg),F.logsigmoid(-gg)+conditional)

    optimizer = torch.optim.AdamW(candidate.parameters(),lr=cfg['learning_rate'],weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,80*64,eta_min=1e-5)
    core = torch.zeros(1,256,dtype=torch.bool,device='cuda');core[:,64:192]=True
    torch.cuda.reset_peak_memory_stats()
    initial_pred, initial_scores, initial_solo = evaluate()
    reference_report=json.loads((ROOT/'outputs/pure_dl_abd_backbone_module_ablation/report.json').read_text(encoding='utf-8'))
    reference_score=next(r['scores']['TOTAL'] for r in reference_report['arms'] if r['arm']=='A_member0_no_global_attention')
    reference_check={'score_delta':initial_scores['TOTAL']-reference_score,
                     'reference':'matched inference removal of full206 global attention',
                     'delta_original_ABD':initial_scores['TOTAL']-cfg['baseline_total']}
    base.write(OUT/'initial_verification.json',reference_check)
    assert abs(reference_check['score_delta'])<1e-4,reference_check
    best_score = initial_scores['TOTAL'];best_epoch=0;bad=0;history=[]
    best_pred = initial_pred.copy()
    def save_best(epoch, scores, solo):
        checkpoint = dict(master)
        checkpoint.update(state_dict={k:v.detach().cpu().clone() for k,v in candidate.state_dict().items()},
                          config=cfg,best_epoch=epoch,best_validation_score=scores['TOTAL'],
                          selection_scope='A+B+D fused score; state_dict is A only',solo_scores=solo)
        torch.save(checkpoint,OUT/'fold_5.pt')
        np.save(OUT/'fold_5_pred.npy',best_pred)
    save_best(0,initial_scores,initial_solo)
    print('INITIAL_FUSED',json.dumps({'scores':initial_scores,'solo':initial_solo,'verification':reference_check}),flush=True)
    frozen_states = [{k:v.detach().cpu().clone() for k,v in m.state_dict().items()} for m in teachers]
    gradient_checked=False
    for epoch in range(1,81):
        epoch_start=time.perf_counter();candidate.train();total=0.
        for step in range(64):
            anchors = train_raw[torch.randint(len(train_raw),(64,),device='cuda')]
            ix,exists = window_indices(anchors,lower,upper)
            relative = torch.tanh((depth[ix]-depth[anchors,None])/10)
            mask = exists & core & labeled[ix]
            yy,dd = labels[ix][mask],default[ix][mask]
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                frozen_outputs = [m(x[ix],relative,exists,absolute_depth=absolute[ix]) for m in teachers]
            frozen_logs = [log_likelihood(out,mask,yy,dd) for out in frozen_outputs]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output = candidate(x[ix],relative,exists,absolute_depth=absolute[ix])
            current_log = log_likelihood(output,mask,yy,dd)
            mixed_log = torch.logsumexp(torch.stack([current_log,*frozen_logs]),0)-math.log(3.)
            loss = (quality[ix][mask]*(-mixed_log).mean(-1)).mean()
            assert torch.isfinite(loss)
            loss.backward()
            if not gradient_checked:
                assert all(p.grad is None for m in teachers for p in m.parameters())
                for module in [*candidate.members,candidate.router]:
                    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
                gradient_checked=True
                print('GRADIENT_CHECK_A_ONLY_PASSED',flush=True)
            torch.nn.utils.clip_grad_norm_(candidate.parameters(),2.)
            optimizer.step();schedule.step();total+=float(loss.detach())
        prediction,scores,solo = evaluate()
        if scores['TOTAL']>best_score:
            best_score=scores['TOTAL'];best_epoch=epoch;bad=0;best_pred=prediction.copy()
            save_best(epoch,scores,solo)
        else:bad+=1
        item={'epoch':epoch,'official_total':scores['TOTAL'],'official_scores':scores,'solo_scores':solo,
              'best_score':best_score,'best_epoch':best_epoch,'bad_epochs':bad,'train_loss':total/64,
              'next_lr':optimizer.param_groups[0]['lr'],'seconds':time.perf_counter()-epoch_start}
        history.append(item);base.write(OUT/'fold_5_validation_history.json',history)
        print('EPOCH_VALIDATION',json.dumps(item),flush=True)
        if epoch>=10 and bad>=18:break
    best = torch.load(OUT/'fold_5.pt',map_location='cpu',weights_only=False)
    candidate.load_state_dict(best['state_dict'])
    final_pred,final_scores,final_solo=evaluate()
    assert np.array_equal(final_pred,best_pred)
    assert all(torch.equal(value.detach().cpu(),before[key]) for m,before in zip(teachers,frozen_states) for key,value in m.state_dict().items())
    report={'fold':5,'screening_fold':5,'scores':final_scores,'solo_scores':final_solo,'best_epoch':best_epoch,
            'stopped_epoch':epoch,'best_checkpoint_predictions_reproduced':True,'checkpoint_persisted':True,
            'all_validation_rows_covered_exactly_four_times':True,'frozen_models_unchanged':True,
            'initial_scores':initial_scores,'delta_initial':best_score-initial_scores['TOTAL'],
            'delta_adopted_baseline':best_score-cfg['baseline_total'],'complete_five_folds':False,
            'seconds':time.perf_counter()-started,'timing_scope':'startup features frozen-cache train validation restore; partially concurrent if another GPU job active',
            'gpu_admission':admission,'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/1024**2}
    base.write(OUT/'fold_5_report.json',report)
    base.write(OUT/'runtime_status.json',{'status':'complete','atlas_current_trial':False,'pid':os.getpid()})
    print('FOLD_RESULT',json.dumps(report),flush=True)


if __name__=='__main__':
    main()
