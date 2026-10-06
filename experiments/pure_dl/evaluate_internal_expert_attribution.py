"""Equal predictive-distribution vote of three independently trained neural models."""
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
from multiscale_sequence import window_indices
from paired_posterior_router_sequence import Network, retained_indices, calibration_indices
from por_t3_patch_loss import probability
from paired_calibrated_view_sequence import Network as CalibratedNetwork
from global_cosine_sequence import Network as SingleNetwork


ROOT = base.ROOT
SOURCE_OUT = ROOT/'outputs/pure_dl_posterior_three_seed_equal80'
OUT = ROOT/'outputs/pure_dl_internal_expert_attribution'
SOURCES = [
    (20261006, ROOT/'outputs/pure_dl_paired_posterior_router80_sequence'),
    (20261007, SOURCE_OUT/'seed_20261007'),
    (20261008, SOURCE_OUT/'seed_20261008'),
    (20261006, ROOT/'outputs/pure_dl_paired_calibrated_view_sequence'),
    (20261006, ROOT/'outputs/pure_dl_global_cosine_sequence'),
]
SOURCES=[SOURCES[0],SOURCES[2]]


def main():
    torch.set_num_threads(3)
    print('GPU_ADMISSION',check_slot(),flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
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
    arms={}
    for source,offset in [('A',0),('B',3)]:
        for label,i in [('full206',0),('calibrated68',1),('routed',2)]:
            w=torch.zeros(3,6);w[:,offset+i]=1.;arms[source+'_'+label]=w
        w=torch.zeros(3,6);w[:,offset:offset+2]=.5;arms[source+'_fixed_equal']=w
    regimes={'four_fine':(edges,utilities,actions,arms)}
    predictions={reg+'__'+arm:np.full((len(validation),3),np.nan) for reg,(_,_,_,aa) in regimes.items() for arm in aa}
    local=np.full(len(truth),-1,np.int64);local[validation]=np.arange(len(validation))
    coverage_evidence=[]
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for wi in range(80):
            if fold[row_ids[wi][0]]!=4:continue
            ids=row_ids[wi];loc=local[ids];n=len(ids)
            within=np.full(len(truth),-1,np.int64);within[ids]=np.arange(n)
            for reg,(ee,uu,aa,votes) in regimes.items():
                count=np.zeros(n,np.int8)
                joint=torch.zeros((3,n,6,len(ee[0])),dtype=torch.float32)
                cc=(np.arange(starts[wi]-32,starts[wi+1]+64,32) if reg=='four_fine'
                    else np.arange(starts[wi],starts[wi+1],128)+64)
                anchors=np.clip(cc,starts[wi],starts[wi+1]-1)
                for begin in range(0,len(cc),32):
                    centers_t=torch.tensor(cc[begin:begin+32],device='cuda')
                    anchor=torch.tensor(anchors[begin:begin+32],device='cuda')
                    ix,exists=window_indices(centers_t,lower,upper,anchors=anchor)
                    relative=torch.tanh((depth[ix]-depth[anchor,None])/10)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        outputs=[]
                        for m in models:
                            members,route,merged=m.prediction_parts(x[ix],relative,exists,absolute_depth=absolute[ix])
                            outputs.extend([members[0],members[1],merged])
                    rows=raw_to_valid[ix[:,64:192].cpu().numpy()]
                    use=exists[:,64:192].cpu().numpy() & (rows>=0)
                    dest=within[rows[use]];assert (dest>=0).all()
                    np.add.at(count,dest,1)
                    select=torch.from_numpy(use).cuda()
                    pw=(torch.sin(torch.pi*(torch.arange(128,device='cuda')+.5)/128).square()[None].expand(len(anchor),-1)[select]/2
                        if reg=='four_fine' else torch.ones(len(dest),device='cuda'))
                    for k,(go,raw) in enumerate(outputs):
                        g=go[:,64:192][select].float().sigmoid();r=raw[:,64:192][select].float()
                        for j in range(3):
                            cp=probability(j,r[:,j,0].softmax(-1),r[:,j,1],F.softplus(r[:,j,2])+.025,ee[j])
                            values=torch.cat([g[:,j:j+1],(1-g[:,j:j+1])*cp],-1)
                            joint[j,:,k].index_add_(0,torch.as_tensor(dest),(values*pw[:,None]).cpu())
                assert (count==(4 if reg=='four_fine' else 1)).all()
                coverage_evidence.append(dict(well=rec[wi][0],regime=reg,points=n,coverage=int(count[0])))
                for j in range(3):
                    for begin in range(0,n,512):
                        end=min(begin+512,n);channel=joint[j,begin:end].cuda()
                        for key,vote in votes.items():
                            mixed=(channel*vote[j].cuda()[None,:,None]).sum(1)
                            chosen=(mixed@uu[j]).argmax(-1).cpu().numpy()
                            predictions[reg+'__'+key][loc[begin:end],j]=aa[j][chosen]
                del joint
            print('WELL_COMPLETE',wi,rec[wi][0],round(time.perf_counter()-started,2),flush=True)
    rows=[]
    for key,pred in predictions.items():
        assert np.isfinite(pred).all()
        scores=base.metric(truth[validation],pred)
        per_well=[dict(well=rec[wi][0],points=len(row_ids[wi]),scores=base.metric(truth[row_ids[wi]],pred[local[row_ids[wi]]])) for wi in range(80) if fold[row_ids[wi][0]]==4]
        rows.append(dict(arm=key,scores=scores,per_well=per_well))
    reference=np.load(ROOT/'outputs/pure_dl_8039_source_ablation/ablation_predictions.npz')['four_fine__single_0']
    reproduced=predictions['four_fine__A_routed']
    verification=dict(reference_score=base.metric(truth[validation],reference),reproduced_score=base.metric(truth[validation],reproduced),
                      exactly_equal=bool(np.array_equal(reference,reproduced)),max_abs_difference=np.abs(reference-reproduced).max(0).tolist(),
                      changed_elements=int(np.count_nonzero(reference!=reproduced)))
    np.savez_compressed(OUT/'ablation_predictions.npz',validation_rows=validation,raw_rows=valid_to_raw[validation],well_ids=wells[validation],**predictions)
    report=dict(screening_fold=5,scope='Post-training expert intervention on original seed06(A) and seed08(B); not a retrained architecture ablation; fixed windows/decoder, no label-based weights',
                sources=[dict(index=k,folder=str(folder.relative_to(ROOT)),sha256=sha) for k,(_,folder,_,sha) in enumerate(checkpoints)],
                arms=rows,reference_verification=verification,coverage=coverage_evidence,
                wall_seconds=time.perf_counter()-started,peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2)
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('ABLATION_COMPLETE',json.dumps(dict(results=[dict(arm=v['arm'],scores=v['scores']) for v in rows],verification=verification)),flush=True)


if __name__=='__main__':main()
