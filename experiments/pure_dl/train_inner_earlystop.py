"""Fixed grouped CV for dense multiscale well-depth networks and loss ablation."""
import argparse
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
from multiscale_sequence import objective, head_loss, window_indices
from depth_norm_joint_sequence import Network
from fold_screening import qualifies, THRESHOLD
from gpu_job_guard import check_slot


from train_invariant import FeatureBuilder, select_features, FEATURES, SEGMENTS

def prepare_inner_features(rec, train_wells, valid, row_ids):
    assert len(train_wells) == 56 and len(set(train_wells)) == 56
    builder = FeatureBuilder().fit([rec[i][1] for i in train_wells])
    indices = select_features(builder.enhanced_feature_names_)
    features = [builder.transform(frame).enhanced[:,indices] for _, frame in rec]
    fm = np.empty(FEATURES, np.float32); fs = np.empty(FEATURES, np.float32)
    edges = np.empty((FEATURES, SEGMENTS+1), np.float32)
    for c in range(FEATURES):
        values = np.concatenate([features[i][:, c] for i in train_wells])
        assert len(values) == sum(len(rec[i][1]) for i in train_wells)
        quantiles = np.quantile(values, np.linspace(0, 1, SEGMENTS+1))
        fm[c] = quantiles[SEGMENTS//2]
        fs[c] = max(quantiles[3*SEGMENTS//4]-quantiles[SEGMENTS//4], 1e-3)
        edges[c] = np.clip((quantiles-fm[c])/fs[c], -8, 8)
    n = sum(len(rows) for rows in row_ids)
    x = torch.empty((n, FEATURES), dtype=torch.float16, device='cuda')
    for i, feature in enumerate(features):
        xx = np.clip((feature[valid[i]]-fm)/fs, -8, 8).astype(np.float16)
        assert xx.shape == (len(row_ids[i]), FEATURES) and np.isfinite(xx).all()
        lo, hi = int(row_ids[i][0]), int(row_ids[i][-1])+1
        assert hi-lo == len(row_ids[i])
        x[lo:hi] = torch.from_numpy(xx).cuda()
    del features, values, feature, xx
    gc.collect()
    return x, builder, fm, fs, edges


def main():
    torch.set_num_threads(3)
    print('GPU_ADMISSION',check_slot(),flush=True)
    out=base.ROOT/'outputs/pure_dl_inner_earlystop';out.mkdir(exist_ok=True)
    start=time.perf_counter()
    rec = base.load_split(base.ROOT.parent/'data', 'train')
    ys = [frame[base.TS].to_numpy(np.float64) for _, frame in rec]
    valid = [np.isfinite(v).all(1) for v in ys]
    truth = np.concatenate([v[m] for v, m in zip(ys, valid)])
    well = np.concatenate([np.full(m.sum(), i, np.int32) for i, m in enumerate(valid)])
    assert len(rec) == 80 and truth.shape == (723556, 3)
    fold = np.full(len(truth), -1, np.int8)
    for f, (_, ix) in enumerate(GroupKFold(5).split(truth, groups=well)):
        fold[ix] = f
    row_ids = [np.flatnonzero(well == i) for i in range(80)]
    starts = np.r_[0, np.cumsum([len(frame) for _, frame in rec])]
    nraw = int(starts[-1])
    valid_to_raw = np.concatenate([np.flatnonzero(m)+starts[i] for i, m in enumerate(valid)])
    raw_to_valid = np.full(nraw, -1, np.int64); raw_to_valid[valid_to_raw] = np.arange(len(truth))
    lower = torch.tensor(np.concatenate([np.full(len(frame), starts[i]) for i, (_, frame) in enumerate(rec)]), device='cuda')
    upper = torch.tensor(np.concatenate([np.full(len(frame), starts[i+1]) for i, (_, frame) in enumerate(rec)]), device='cuda')
    depth = torch.tensor(np.concatenate([frame['DEPTH'].to_numpy(np.float32) for _, frame in rec]), device='cuda')
    assert torch.isfinite(depth).all()

    outer_train=np.unique(well[fold!=0]);outer_hold=np.unique(well[fold==0])
    inner_val=np.sort(np.random.default_rng(20260925).permutation(outer_train)[:8])
    tr_wells=sorted(set(outer_train.tolist())-set(inner_val.tolist()))
    assert len(tr_wells)==56 and len(inner_val)==8 and len(outer_hold)==16
    training=np.flatnonzero(np.isin(well,tr_wells));validation=np.flatnonzero(np.isin(well,inner_val))
    defaults=base.dmask(truth);contract=base.feature_contract()
    cfg={'seed':20261006,'epochs':40,'steps':64,'windows_per_step':64,'patience':5,'min_delta':.002,'minimum_epochs':10,'criterion':'inner all-valid-row main-head BCE plus conditional NLL','outer_fold':0,'split_seed':20260925}
    source_files=[Path(__file__)]+[Path(__file__).with_name(n) for n in ['depth_norm_joint_sequence.py','absolute_depth_sequence.py','depthwise_norm_sequence.py','multiscale_sequence.py','train_invariant.py','train.py','gpu_job_guard.py']]+[base.ROOT/'package/src'/n for n in ['feature.py','data.py','utils.py']]
    cfg['sha256']={str(q.relative_to(base.ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in source_files}
    base.write(out/'config.json',cfg)
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);t0=time.perf_counter()
    core=torch.zeros(1,256,device='cuda',dtype=torch.bool);core[:,64:192]=True
    args=argparse.Namespace(deep_supervision=True)
    valid_x, builder, fm, fs, edges = prepare_inner_features(rec, tr_wells, valid, row_ids)
    x = torch.empty(nraw, 206, dtype=torch.float16, device='cuda')
    for i, (_, frame) in enumerate(rec):
        value = builder.transform(frame).enhanced[:, contract['selected_indices']]
        value = np.clip((value-fm)/fs, -8, 8).astype(np.float16)
        x[starts[i]:starts[i+1]] = torch.from_numpy(value).cuda()
    assert torch.equal(x[torch.tensor(valid_to_raw, device='cuda')], valid_x)
    del valid_x, value
    med = []; scale = []; z = []; centers = []; density_edges = []; initial = []
    for j in range(3):
        transformed = base.TRANSFORM[j](truth[:, j])
        fit = transformed[np.isin(well,tr_wells) & ~defaults[:, j]]
        q25, q50, q75 = np.percentile(fit, [25, 50, 75]); sc = max(float(q75-q25), .1)
        med.append(q50); scale.append(sc); z.append((transformed-q50)/sc)
        initial.append(np.quantile((fit-q50)/sc, np.linspace(.05, .95, 8)))
        ct = np.linspace(fit.min(), fit.max(), 256); centers.append(base.INVERSE[j](ct))
        density_edges.append((np.r_[-np.inf, (ct[:-1]+ct[1:])/2, np.inf]-q50)/sc)
    train_raw_np = valid_to_raw[training]
    train_raw = torch.tensor(train_raw_np, device='cuda')
    labels = torch.zeros(nraw, 3, device='cuda')
    state = torch.zeros(nraw, 3, dtype=torch.bool, device='cuda')
    labeled = torch.zeros(nraw, dtype=torch.bool, device='cuda')
    labels[train_raw] = torch.tensor(np.column_stack(z)[training], device='cuda', dtype=torch.float32)
    state[train_raw] = torch.tensor(defaults[training], device='cuda'); labeled[train_raw] = True
    assert not bool(labeled[torch.tensor(valid_to_raw[validation], device='cuda')].any())
    depth_fit = depth[train_raw].cpu().numpy().astype(np.float64)
    depth_q = np.percentile(depth_fit, [25,50,75])
    depth_median = float(depth_q[1]); depth_scale = max(float(depth_q[2]-depth_q[0]), .1)
    absolute = ((depth-depth_median)/depth_scale).clamp(-8,8)
    model = Network(initial_means=np.stack(initial)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 40*cfg['steps'], eta_min=.00001)
    history = []; torch.cuda.reset_peak_memory_stats()

    evaluation=[];best=float('inf');best_epoch=0;bad=0
    inner_y=torch.tensor(np.column_stack(z)[validation],device='cuda',dtype=torch.float32)
    inner_d=torch.tensor(defaults[validation],device='cuda')
    local=np.full(len(truth),-1,np.int64);local[validation]=np.arange(len(validation))
    ec=[];ea=[]
    for wi in inner_val:
        cc=np.arange(starts[wi],starts[wi+1],128)+64
        ec.extend(cc);ea.extend(np.minimum(cc,starts[wi+1]-1))
    def validate_inner():
        model.eval();covered=np.zeros(len(validation),np.int8)
        gates=torch.empty(len(validation),3,device='cuda');density=torch.empty(len(validation),3,3,8,device='cuda')
        with torch.no_grad():
            for begin in range(0,len(ec),32):
                cc=torch.tensor(ec[begin:begin+32],device='cuda');anchor=torch.tensor(ea[begin:begin+32],device='cuda')
                ix,exists=window_indices(cc,lower,upper,anchors=anchor)
                relative=torch.tanh((depth[ix]-depth[anchor,None])/10)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    go,rr=model(x[ix],relative,exists,absolute_depth=absolute[ix])
                rows=raw_to_valid[ix[:,64:192].cpu().numpy()]
                use=exists[:,64:192].cpu().numpy()&(rows>=0);loc=local[rows[use]]
                assert (loc>=0).all() and (covered[loc]==0).all() and len(np.unique(loc))==len(loc)
                covered[loc]+=1;mask=torch.from_numpy(use).cuda();ii=torch.tensor(loc,device='cuda')
                gates[ii]=go[:,64:192][mask].float();density[ii]=rr[:,64:192][mask].float()
            assert (covered==1).all()
            loss=head_loss((gates[None],density[None]),inner_y[None],inner_d[None],torch.ones(1,len(validation),device='cuda',dtype=torch.bool))
            assert torch.isfinite(loss)
            return float(loss)
    for epoch in range(40):
        model.train(); total = 0.
        for _ in range(cfg['steps']):
            anchors = train_raw[torch.randint(len(train_raw), (cfg['windows_per_step'],), device='cuda')]
            ix, exists = window_indices(anchors, lower, upper)
            relative = torch.tanh((depth[ix]-depth[anchors, None])/10)
            supervised = exists & core & labeled[ix]
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(x[ix], relative, exists, auxiliary=args.deep_supervision, absolute_depth=absolute[ix])
            loss = objective(outputs, labels[ix], state[ix], supervised, args.deep_supervision)
            assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            opt.step(); sched.step(); total += float(loss.detach())
        history.append(total/cfg['steps'])
        if epoch == 0 or (epoch+1) % 5 == 0:
            print(f'inner_pilot epoch={epoch+1}/40 loss={history[-1]:.5f} seconds={time.perf_counter()-t0:.1f}', flush=True)

        inner_loss=validate_inner()
        if inner_loss < best-cfg['min_delta']:
            best=inner_loss;best_epoch=epoch+1;bad=0
        else:bad+=1
        evaluation.append({'epoch':epoch+1,'training_loss':history[-1],'inner_loss':inner_loss,'best_epoch':best_epoch,'bad_epochs':bad})
        base.write(out/'history.json',evaluation)
        print('INNER_VALIDATION',json.dumps(evaluation[-1]),flush=True)
        if epoch+1>=cfg['minimum_epochs'] and bad>=cfg['patience']:break
    assert all(hashlib.sha256(q.read_bytes()).hexdigest()==cfg['sha256'][str(q.relative_to(base.ROOT))] for q in source_files)
    np.savez_compressed(out/'scope.npz',training_raw_rows=train_raw_np,inner_validation_raw_rows=valid_to_raw[validation],outer_validation_raw_rows=valid_to_raw[fold==0],regression_median=med,regression_scale=scale,feature_median=fm,feature_scale=fs,depth_median=depth_median,depth_scale=depth_scale)
    result={'config':cfg,'training_wells':tr_wells,'inner_validation_wells':inner_val.tolist(),'outer_validation_wells':outer_hold.tolist(),'best_epoch':best_epoch,'stopped_epoch':epoch+1,'early_stopped':epoch+1<40,'best_inner_loss':best,'history':evaluation,'seconds':time.perf_counter()-start,'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'outer_predictions_produced':False,'official_score':None,'requires_refit_on_all64_training_wells':True}
    base.write(out/'pilot_report.json',result);print('PILOT_RESULT',json.dumps({k:v for k,v in result.items() if k not in ['config','history']}),flush=True)
if __name__=='__main__':main()
