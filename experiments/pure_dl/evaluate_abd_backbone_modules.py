"""Equal predictive-distribution vote of three independently trained neural models."""
import math
import os
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
from paired_calibrated_view_sequence import Network as CalibratedNetwork
from global_cosine_sequence import Network as SingleNetwork


ROOT = base.ROOT
SOURCE_OUT = ROOT/'outputs/pure_dl_posterior_three_seed_equal80'
OUT = ROOT/'outputs/pure_dl_abd_backbone_module_ablation'
SOURCES = [
    (20261006, ROOT/'outputs/pure_dl_paired_posterior_router80_sequence'),
    (20261008, SOURCE_OUT/'seed_20261008'),
    (20261006, ROOT/'outputs/pure_dl_global_cosine_sequence'),
]


def main():
    torch.set_num_threads(3)
    admission = check_slot()
    print('GPU_ADMISSION', admission, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'report.json').exists():
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
    cfg = {'screening_fold':5,'baseline_total':80.40915186344968,
           'scope':'post-training module intervention in A; B/D frozen; not retraining causal gain',
           'ensemble':'A+B+D equal distributions;four cosine windows;fixed fine decoder',
           'source_checks':[{'folder':str(folder.relative_to(ROOT)),'sha256':sha} for _,folder,_,sha in checkpoints]}
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

    # Each arm starts from the identical original model; hooks never change weights.
    candidate.requires_grad_(False)
    original_state = {k:v.detach().cpu().clone() for k,v in candidate.state_dict().items()}
    initial_pred,initial_scores,initial_solo = evaluate()
    with np.load(ROOT/'outputs/pure_dl_8040_pair_ablation/ablation_predictions.npz') as pack:
        assert np.array_equal(pack['validation_rows'],validation)
        reference = pack['four_fine__triple_drop_3']
    assert abs(initial_scores['TOTAL']-cfg['baseline_total'])<1e-4
    reference_check={'exact_predictions':bool(np.array_equal(initial_pred,reference)),
                     'different_elements':int(np.count_nonzero(initial_pred!=reference)),
                     'score_delta':initial_scores['TOTAL']-cfg['baseline_total']}
    def per_well(pred):
        return [{'well_id':int(wi),'scores':base.metric(truth[row_ids[wi]],pred[local[row_ids[wi]]])} for wi in val_wells]
    rows=[{'arm':'baseline','scores':initial_scores,'solo_scores':initial_solo,'delta_fused':0.,'delta_solo':0.,'per_well':per_well(initial_pred)}]
    print('BASELINE',json.dumps(rows[0]),flush=True)
    def zero_output(module,args,output):
        return torch.zeros_like(output)
    def identity_output(module,args,output):
        return args[0]
    def drop_skip(main_width):
        def hook(module,args):
            xx=args[0]
            return (torch.cat([xx[:,:main_width],torch.zeros_like(xx[:,main_width:])],1),)
        return hook
    def attach(member,kind):
        if kind=='local_attention':
            return [member.stem.hierarchy.local.register_forward_hook(zero_output)]
        if kind=='global_attention':
            return [member.stem.hierarchy.global_patches.register_forward_hook(zero_output)]
        if kind=='whole_hierarchy':
            return [member.stem.hierarchy.register_forward_hook(zero_output)]
        if kind=='point_mlp':
            return [member.point.register_forward_hook(zero_output)]
        if kind=='bottleneck_residuals':
            return [member.bottleneck.register_forward_hook(identity_output)]
        if kind=='unet_skips':
            return [member.merge2.register_forward_pre_hook(drop_skip(192)),
                    member.merge1.register_forward_pre_hook(drop_skip(128)),
                    member.merge0.register_forward_pre_hook(drop_skip(96))]
        raise ValueError(kind)
    kinds=['local_attention','global_attention','whole_hierarchy','point_mlp','bottleneck_residuals','unet_skips']
    for member_index in range(2):
        for kind in kinds:
            t0=time.perf_counter()
            hooks=attach(candidate.members[member_index],kind)
            try:
                prediction,scores,solo=evaluate()
            finally:
                for hook in hooks:hook.remove()
            item={'arm':f'A_member{member_index}_no_{kind}','member':member_index,'module':kind,
                  'scores':scores,'solo_scores':solo,'delta_fused':scores['TOTAL']-initial_scores['TOTAL'],
                  'delta_solo':solo['TOTAL']-initial_solo['TOTAL'],'per_well':per_well(prediction),
                  'seconds':time.perf_counter()-t0}
            rows.append(item)
            base.write(OUT/'partial_results.json',rows)
            print('MODULE_ABLATION',json.dumps({k:v for k,v in item.items() if k!='per_well'}),flush=True)
    assert all(torch.equal(v.detach().cpu(),original_state[k]) for k,v in candidate.state_dict().items())
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in candidate.modules())
    report={'screening_fold':5,'scope':cfg['scope'],'baseline_scores':initial_scores,'reference_check':reference_check,
            'arms':rows,'source_checks':cfg['source_checks'],'rows':len(validation),'weights_unchanged':True,
            'all_validation_rows_covered_exactly_four_times':True,'seconds':time.perf_counter()-started,
            'timing_scope':'features frozen-cache and13arms; concurrent with ABD finetune if active',
            'gpu_admission':admission,'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/1024**2}
    base.write(OUT/'report.json',report)
    base.write(OUT/'runtime_status.json',{'status':'complete','atlas_current_trial':False,'pid':os.getpid()})
    print('ABLATION_COMPLETE',json.dumps({k:v for k,v in report.items() if k!='arms'}),flush=True)


if __name__=='__main__':
    main()

