"""Fixed grouped CV for dense multiscale well-depth networks and loss ablation."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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
from multiscale_sequence import window_indices
from por_t3_patch_loss import probability, head_loss
from paired_posterior_router_sequence import Network, retained_indices, VIEW_NAMES, calibration_indices
from paired_backbone_sequence import merge_predictions
from paired_backbone_sequence import objective
from abd_module_intervention import disable_full_global_attention
from abd_frozen_ad_validation import FrozenADValidation
from frozen_companion_fused_loss import FrozenCompanions, channel_gradient_snapshot
from global_point_weights import GlobalPointWeights
from compact_memory_dynamics_probe import CompactMemoryDynamicsProbe
from conflict_recovery2_policy import PersistentConflictPolicy
from fold5_screening_804 import qualifies, THRESHOLD, FOLD_INDEX, fold_plan
from gpu_job_guard_v2 import check_slot


def main():
    ap = argparse.ArgumentParser()
    ap.set_defaults(deep_supervision=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--learning-rate', type=float, choices=[.0006], required=True)
    ap.add_argument('--schedule', choices=['cosine'], required=True)
    ap.add_argument('--fold-count', type=int, choices=[1], default=1)
    ap.add_argument('--ablation',choices=['control'],default='control')
    args = ap.parse_args()
    def make_network(*positional,**keywords):
        model=Network(*positional,**keywords)
        if args.ablation=='no_global_full':disable_full_global_attention(model)
        return model
    if args.fold_count == 5:
        proof = json.loads((args.out/'screening_verification.json').read_text())
        assert proof['passed'] and proof['screening_fold']==5 and qualifies(proof['screening_fold_score']), 'Fifth fold must be independently verified above80.4'
    torch.set_num_threads(3)
    print('GPU_ADMISSION', check_slot(min_free_gib=3.), flush=True)
    start = time.perf_counter(); args.out.mkdir(parents=True, exist_ok=True)
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
    sources = [Path(__file__).with_name('frozen_companion_fused_loss.py'),Path(__file__).with_name('paired_detached_router_sequence.py'),Path(__file__).with_name('abd_frozen_ad_validation.py'),Path(__file__).with_name('abd_module_intervention.py'),Path(__file__).with_name('paired_posterior_router_sequence.py'), Path(__file__).with_name('paired_detached_router_sequence.py'), Path(__file__).with_name('paired_calibrated_view_sequence.py'), Path(__file__).with_name('paired_point_view_sequence.py'), Path(__file__).with_name('paired_well_view_sequence.py'), Path(__file__).with_name('paired_backbone_sequence.py'), Path(__file__).with_name('global_cosine_sequence.py'), Path(__file__).with_name('patch_hierarchy_sequence.py'), Path(__file__).with_name('por_student3_sequence.py'), Path(__file__).with_name('cosine_patch_sequence.py'), Path(__file__).with_name('compact_memory_dynamics_probe.py'), Path(__file__).with_name('training_por_t3_full.py'), Path(__file__).with_name('por_t3_patch_loss.py'), Path(__file__).with_name('global_point_loss.py'), Path(__file__).with_name('global_point_weights.py'), Path(__file__).with_name('training_conflict_probe.py'), Path(__file__).with_name('conflict_recovery2_policy.py'), Path(__file__).with_name('relative_only_sequence.py'), Path(__file__).with_name('blur_ungated_sequence.py'), Path(__file__).with_name('blur_no_point_sequence.py'), Path(__file__).with_name('blur_no_dropout_sequence.py'), Path(__file__).with_name('blur_down_sequence.py'), Path(__file__).with_name('population_likelihood_sequence.py'), Path(__file__), Path(__file__).with_name('multiscale_sequence.py'),
               Path(__file__).with_name('absolute_depth_sequence.py'), Path(__file__).with_name('depthwise_norm_sequence.py'), Path(__file__).with_name('depth_norm_joint_sequence.py'), Path(__file__).with_name('fold5_screening_804.py'), Path(__file__).with_name('gpu_job_guard_v2.py'), Path(base.__file__),
               Path(__file__).with_name('train.py'), base.ROOT/'package/src/feature.py',
               base.ROOT/'package/src/data.py', base.ROOT/'package/src/utils.py']
    hashes = {str(p.relative_to(base.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    contract = base.feature_contract(); base.write(args.out/'feature_contract.json', contract)
    view_indices = retained_indices(contract['whitelist'])
    view_names = [contract['whitelist'][i] for i in view_indices]
    numeric_indices, conditioning_indices = calibration_indices(contract['whitelist'])
    cfg = {'baseline_control':'exact original posterior-router Network; compact probe storage and below-gate artifact policy only', 'probe_storage':'float32 prior CDF in RAM; compact final point state', 'negative_artifacts':'no checkpoint or full distributions when fifth<=80.4', 'posterior_descriptors':{'count':24,'order':'member then target then gate/component_entropy/mean/log1p_total_std','source':'detached main posterior once per member','bounds':'gate entropy0..1; mean[-8,8]; log1p_std<=8','variance':'sum weight*(component_std^2+(mu-mixture_mean)^2); PORsd is actual t3 std'}, 'detached_router':{'input':92,'hidden':32,'outputs':[3,2],'initialization':'zero last layer; after experts fork_rng','expert_objective':'original member NLL mean','router_objective':'main full posterior NLL from detached expert tensors; same Q','optimizer':'separate AdamW and separate clip2; matched80 cosine','aux_routing':False}, 'input_calibration':{'numeric_indices':numeric_indices,'numeric_names':[contract['whitelist'][i] for i in numeric_indices],'conditioning_indices':conditioning_indices,'conditioning_names':[contract['whitelist'][i] for i in conditioning_indices],'formula':'x*(1+tanh(gamma))+tanh(beta)','hidden':32,'final_zero':True}, 'well_view': {'member0':'full206','member1':'instantaneous41 plus input-only within-well rank/ziqr/qnorm27 total68','indices':view_indices,'names_in_input_order':view_names,'required_names':list(VIEW_NAMES),'excluded_value':0,'space':'original train-only standardized input'}, 'seed': 20261008, 'epochs': 80, 'initial_lr': args.learning_rate, 'lr_schedule': args.schedule, 'steps': 64, 'windows_per_step': 64,
           'window_length': 256, 'supervised_core': [64, 192], 'deep_supervision': args.deep_supervision,
           'auxiliary_weights': [.2, .1] if args.deep_supervision else [], 'auxiliary_schedule': 'linear factor max(0,1-epoch_zero_based/40)',
           'architecture': 'physical absolute depth plus nine channel-only per-depth LayerNorm blocks',
           'normalization': 'LayerNorm across channels independently at each depth; nine blocks',
           'absolute_depth': 'train labeled rows median/IQR; clipped [-8,8]; zero initialized extra stem channel', 'screening_threshold': THRESHOLD, 'screening_fold': 5,
           'downsample_filter': 'fixed [.25,.5,.25] replicate padding before three stride2 convolutions', 'dropout_probability': 0., 'point_bypass': False, 'absolute_depth_enabled': False, 'residual_gate': 'constant0.5', 'initialization': 'retain value projections; remove gate projections', 'features': 206, 'input_encoding': 'robust scaling only; no PLE or learned per-feature embedding',
           'parameters': sum(p.numel() for p in make_network(retain_indices=view_indices, numeric_indices=numeric_indices, conditioning_indices=conditioning_indices).parameters()), 'dependency_sha256': hashes,
           'test_access': False, 'label_scope': 'only64 training wells allocated for loss; validation labels used after inference',
           'loss_reduction': 'Global point weight times completed BCE plus masked NLL; original population denominator', 'probe_coverage':'all valid training rows once per observation', 'conflict_entropy_quantile':0.40, 'conflict_tail_threshold':0.02, 'recovery_consecutive_records':2, 'density_family':'POR StudentT df3 scale=std/sqrt3; PERM/SW Gaussian; synchronized loss decode probe', 'attention_scores':'local dot product; global cosine QK float32 scale init sqrt8 capped100', 'independent_pair':{'members':2,'full_predictive_weights':'input-dependent channelwise softmax; initial equal','training':'mean of individual member likelihood losses','quality':'one full-training-point table from aggregate predictive distribution','initialization':'sequential independent from fixed fold seed','components_per_target':16}, 'patch_hierarchy':{'patch_size':16,'width':32,'heads':4,'local_points':16,'global_tokens':16,'zero_init_output':True}, 'dynamic_policy':'train-probe persistent narrow-distribution target-tail conflict; difficult neutral; applies next epoch; Q in[.1,1]',
           'lr_policy': {'minimum_lr': .00001, 'plateau_factor': .5, 'plateau_patience': 3, 'plateau_threshold': .01},
           'selection': 'user-requested highest whole-fold official validation checkpoint; no refit; each outer fold selects its own checkpoint',
           'early_stopping': {'patience': 18, 'minimum_epochs': 10, 'maximum_epochs': 80, 'criterion': 'official_TOTAL', 'mode': 'max'},
           'outer_wells': {str(f): [rec[i][0] for i in range(80) if fold[row_ids[i][0]] == f] for f in range(5)}}
    cfg['attribution_ablation']='B_from_scratch_final_fused_loss_frozen_A6_D'
    cfg['trainable_role']='B'
    cfg['frozen_roles']=['A_best_epoch6','D_original']
    cfg['baseline_control']='From-scratch B seed08; original expert/router NLL and auxiliary/dynamic Q; best A6 and D frozen validation companions'
    cfg['negative_artifacts']='retain best B checkpoint until fused assessment; no epoch checkpoint stack'
    cfg['ensemble_baseline']=80.49230853582785
    cfg['selection']='highest fixed ABD equal posterior fused official TOTAL; B solo diagnostic'
    cfg['early_stopping']['criterion']='ABD_fused_official_TOTAL'
    cfg['validation']='deterministic four cosine windows;1024 bins;original800 union3197 actions'
    cfg['training_loss_scope']='Final equal A/B/D predictive likelihood; A6 and D frozen in every training forward; gradient into B backbones and router only. Original B auxiliary losses retained.'
    cfg['baseline_control']='Compare B scratch solo-loss training: same initialization, samples, LR, aux, Q algorithm, frozen companions and validation; replace independent main and detached router objectives by final fused main objective'
    cfg['detached_router']['expert_objective']='final ABD fused main NLL plus original annealed B auxiliary losses'
    cfg['detached_router']['router_objective']='same final ABD fused main NLL; no separate detached expert main loss'
    cfg['detached_router']['optimizer']='original separate AdamW/clip2 for B experts and B router; common fused backward'
    cfg['independent_pair']['training']='B internal members coupled through final ABD predictive likelihood'
    cfg['quality_probe_scope']='unchanged original B posterior on all training points; Q applied to final fused loss and original B auxiliary loss'
    cfg['channel_monitor']='per-epoch3channel fused BCE/NLL; first minibatch gradient norms/cosines, no dynamic channel reweighting yet'
    cfg['initialization']='B seed20261008+4 random initialization; no B checkpoint warm start'
    cfg['architecture']='dual relative-depth U-Net; local+global patch attention; full206 plus calibrated68; no point MLP; ungated residual blocks'
    cfg['module_intervention']='none' if args.ablation=='control' else 'only member0 global_patches replaced by ZeroAttention after identical initialization' 
    if (args.out/'config.json').exists():
        assert json.loads((args.out/'config.json').read_text()) == cfg
    base.write(args.out/'config.json', cfg)
    base.write(args.out/'runtime_status.json',{'status':'running','atlas_current_trial':True,'pid':os.getpid()})
    defaults = base.dmask(truth); pred = np.full_like(truth, np.nan); reports = []
    core = torch.zeros(1, 256, device='cuda', dtype=torch.bool); core[:, 64:192] = True
    for f in fold_plan(args.fold_count):
        if f != FOLD_INDEX:
            assert qualifies(base.metric(truth[fold == FOLD_INDEX], pred[fold == FOLD_INDEX])['TOTAL']), 'Cached fifth-fold score fails gate'
        t0 = time.perf_counter(); torch.manual_seed(cfg['seed']+f); np.random.seed(cfg['seed']+f)
        validation = np.flatnonzero(fold == f); training = np.flatnonzero(fold != f)
        cache = args.out/f'fold_{f+1}_pred.npy'
        if cache.exists():
            for suffix in ['_report.json', '_distributions.npz', '.pt']:
                assert (args.out/f'fold_{f+1}{suffix}').exists()
            cached_checkpoint = torch.load(args.out/f'fold_{f+1}.pt', map_location='cpu', weights_only=False)
            assert cached_checkpoint['config'] == cfg
            del cached_checkpoint
            print(f'REUSE_VERIFIED_CONFIG_FOLD {f+1}', flush=True)
            pred[validation] = np.load(cache)
            reports.append(json.loads((args.out/f'fold_{f+1}_report.json').read_text())); continue
        tr_wells = [i for i in range(80) if fold[row_ids[i][0]] != f]
        print(f'FOLD {f+1} train-only preprocessing; deep_supervision={args.deep_supervision}', flush=True)
        valid_x, builder, fm, fs, edges = base.prepare_features(rec, tr_wells, valid, row_ids)
        actual_names = [builder.enhanced_feature_names_[i] for i in contract['selected_indices']]
        assert actual_names == contract['whitelist']
        assert retained_indices(actual_names) == view_indices
        assert calibration_indices(actual_names) == (numeric_indices, conditioning_indices)
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
            fit = transformed[(fold != f) & ~defaults[:, j]]
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
        model = make_network(initial_means=np.stack(initial), retain_indices=view_indices, numeric_indices=numeric_indices, conditioning_indices=conditioning_indices).cuda()
        expert_parameters=list(model.members.parameters())
        router_parameters=list(model.router.parameters())
        assert set(map(id,expert_parameters)).isdisjoint(map(id,router_parameters))
        assert set(map(id,expert_parameters+router_parameters))==set(map(id,model.parameters()))
        opt = torch.optim.AdamW(expert_parameters, lr=args.learning_rate, weight_decay=.01)
        router_opt = torch.optim.AdamW(router_parameters, lr=args.learning_rate, weight_decay=.01)
        router_sched = torch.optim.lr_scheduler.CosineAnnealingLR(router_opt,cfg['epochs']*cfg['steps'],eta_min=.00001)
        sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['epochs']*cfg['steps'], eta_min=.00001) if args.schedule=='cosine' else torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=3,threshold=.01,threshold_mode='abs',min_lr=.00001))
        actions = []; utility = []
        for j, name in enumerate(base.TS):
            ct = centers[j]
            action = np.linspace(ct[0], ct[-1], 800) if j == 2 else np.geomspace(max(ct[0]*.9, .001), ct[-1]*1.1, 800)
            action = np.unique(np.r_[base.DEFAULTS[name], np.clip(action, *base.BOUNDS[name])])
            actions.append(action)
            utility.append(torch.tensor(base.kernel_matrix(action, np.r_[base.DEFAULTS[name], ct], name).T, device='cuda', dtype=torch.float32))
        probe=CompactMemoryDynamicsProbe(args.out/f'fold_{f+1}_dynamics',tr_wells,starts,labeled,lower,upper,depth,absolute,labels,state,density_edges,med,scale,base.TS)
        assert np.array_equal(np.sort(probe.raw),np.sort(train_raw_np)), "Full training coverage required"
        policy=PersistentConflictPolicy(len(probe.raw))
        table=GlobalPointWeights(nraw,train_raw_np,starts)
        supervision_weights=torch.ones(nraw,device='cuda')
        table.save(args.out/f'fold_{f+1}_global_point_weights.npz')
        weight_trace=[];update_epochs=[]
        evaluator=FrozenADValidation(x=x,depth=depth,lower=lower,upper=upper,absolute=absolute,starts=starts,
            row_ids=row_ids,raw_to_valid=raw_to_valid,validation=validation,preprocessing={
            'selected_feature_names':contract['whitelist'],'selected_feature_indices':contract['selected_indices'],
            'feature_median':fm,'feature_scale':fs,'regression_median':med,'regression_scale':scale,
            'absolute_depth_median':depth_median,'absolute_depth_scale':depth_scale,
            'validation_raw_rows':valid_to_raw[validation],'centers':centers,'density_edges':density_edges})
        cfg['frozen_companions']=evaluator.source_checks
        base.write(args.out/'config.json',cfg)
        companions=FrozenCompanions(evaluator.source_checks)
        print('FROZEN_AD_TRAINING_READY',flush=True)
        last_solo_scores={}
        def evaluate_current():
            nonlocal last_solo_scores
            fused,solo=evaluator.evaluate(model)
            pred[validation]=fused
            last_solo_scores=base.metric(truth[validation],solo)
            return {'indices':validation,'fused_prediction':fused,'solo_prediction':solo}
        history = []; torch.cuda.reset_peak_memory_stats()
        best_score=-float('inf'); best_epoch=0; bad_epochs=0; validation_history=[]; best_state=None; best_predictions=None
        for epoch in range(cfg['epochs']):
            model.train(); total=0.; fused_total=0.; auxiliary_total=0.; channel_total=np.zeros(3); bce_total=np.zeros(3); nll_total=np.zeros(3); epoch_lr=opt.param_groups[0]['lr']
            for step in range(cfg['steps']):
                anchors = train_raw[torch.randint(len(train_raw), (64,), device='cuda')]
                ix, exists = window_indices(anchors, lower, upper)
                relative = torch.tanh((depth[ix]-depth[anchors, None])/10)
                supervised = exists & core & labeled[ix]
                opt.zero_grad(set_to_none=True)
                router_opt.zero_grad(set_to_none=True)
                loss,channel_losses,channel_bce,channel_nll,auxiliary_loss=companions.forward_loss(
                    model,x[ix],relative,exists,absolute[ix],labels[ix],state[ix],supervised,supervision_weights[ix],epoch)
                if step==0:
                    gradient_snapshot=channel_gradient_snapshot(channel_losses,model)
                assert torch.isfinite(loss)
                loss.backward()
                if epoch==0 and step==0:
                    assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in expert_parameters)
                    assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in router_parameters)
                    companions.verify()
                    print('GRADIENT_SCOPE_VERIFIED frozen A/D; trainable B experts+router',flush=True)
                torch.nn.utils.clip_grad_norm_(expert_parameters,2)
                torch.nn.utils.clip_grad_norm_(router_parameters,2)
                last_lr=opt.param_groups[0]['lr']
                opt.step();router_opt.step()
                if args.schedule=='cosine':sched.step();router_sched.step()
                total += float(loss.detach())
                fused_total += float(channel_losses.detach().mean()); auxiliary_total += float(auxiliary_loss.detach())
                channel_total += channel_losses.detach().cpu().numpy(); bce_total += channel_bce.detach().cpu().numpy(); nll_total += channel_nll.detach().cpu().numpy()
            history.append(total/cfg['steps'])
            if epoch == 0 or (epoch+1) % 5 == 0:
                print(f"fold={f+1} epoch={epoch+1}/{cfg['epochs']} loss={history[-1]:.5f} seconds={time.perf_counter()-t0:.1f}", flush=True)
            current_pack=evaluate_current()
            channel_scores=base.metric(truth[validation],pred[validation])
            score=channel_scores['TOTAL']
            assert np.isfinite(score)
            if args.schedule=='plateau':sched.step(score)
            if score>best_score:
                best_score=score;best_epoch=epoch+1;bad_epochs=0
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                best_predictions=pred[validation].copy()
            else:bad_epochs+=1
            validation_history.append({'epoch':epoch+1,'first_lr':epoch_lr,'last_lr':last_lr,'next_lr':opt.param_groups[0]['lr'],'fused_train_loss':fused_total/cfg['steps'],'auxiliary_train_loss':auxiliary_total/cfg['steps'],'channel_fused_loss':(channel_total/cfg['steps']).tolist(),'channel_gate_bce':(bce_total/cfg['steps']).tolist(),'channel_continuous_nll':(nll_total/cfg['steps']).tolist(),'channel_gradients':gradient_snapshot,'router_next_lr':router_opt.param_groups[0]['lr'],'official_total':score,'official_scores':channel_scores,'solo_scores':last_solo_scores,'best_epoch':best_epoch,'best_score':best_score,'bad_epochs':bad_epochs})
            base.write(args.out/f'fold_{f+1}_validation_history.json',validation_history)
            print('EPOCH_VALIDATION',json.dumps(validation_history[-1]),flush=True)
            del current_pack
            if epoch==0 or (epoch+1)%2==0:
                probe.record(model,x,epoch+1)
                q=policy.update(epoch+1,probe.history['nll'][-1],probe.history['conditional_entropy'][-1],probe.history['target_tail'][-1],probe.history['gate_probability'][-1],probe.history['cdf_l1_change'][-1],~probe.default)
                assert np.isfinite(q).all() and (q>=.1).all() and (q<=1).all()
                point_q=table.update(probe.raw,q,epoch+1)
                supervision_weights[torch.as_tensor(probe.raw,device='cuda')]=torch.as_tensor(point_q,device='cuda')
                table.save(args.out/f'fold_{f+1}_global_point_weights.npz')
                assert torch.equal(supervision_weights[torch.as_tensor(valid_to_raw[validation],device='cuda')],torch.ones(len(validation),device='cuda'))
                weight_trace.append(q.astype(np.float32));update_epochs.append(epoch+1)
                np.savez_compressed(args.out/f'fold_{f+1}_conflict_weights.npz',epochs=np.asarray(update_epochs),raw_ids=probe.raw,weights=np.stack(weight_trace))
                base.write(args.out/f'fold_{f+1}_conflict_snapshot.json',policy.snapshot())
                print('CONFLICT_UPDATE',json.dumps({'observed_epoch':epoch+1,'applies_from_epoch':epoch+2,'downweighted_by_channel':(q<1).sum(0).tolist(),'minimum':float(q.min())}),flush=True)
            if epoch+1>=cfg['early_stopping']['minimum_epochs'] and bad_epochs>=cfg['early_stopping']['patience']:break
        stopped_epoch=epoch+1
        probe.finalize()
        companions.verify()
        model.load_state_dict(best_state)
        pack=evaluate_current()
        assert np.array_equal(pred[validation],best_predictions), 'Restored best checkpoint must reproduce saved best predictions'
        assert abs(base.metric(truth[validation],pred[validation])['TOTAL']-best_score)<1e-10
        keep_weights = True
        if keep_weights:
            np.savez_compressed(args.out/f'fold_{f+1}_distributions.npz', **pack)
        checkpoint = {'state_dict': model.cpu().state_dict() if keep_weights else None, 'config': cfg, 'best_epoch': best_epoch, 'stopped_epoch': stopped_epoch, 'best_validation_score': best_score, 'absolute_depth_median': depth_median, 'absolute_depth_scale': depth_scale,
            'feature_median': fm.tolist(), 'feature_scale': fs.tolist(), 'ple_edges_normalized': edges.tolist(),
            'medians': builder.medians_, 'selected_feature_names': contract['whitelist'],
            'selected_feature_indices': contract['selected_indices'], 'regression_median': med,
            'regression_scale': scale, 'centers': centers, 'density_edges': density_edges,
            'training_raw_rows': train_raw_np, 'validation_raw_rows': valid_to_raw[validation]}
        checkpoint['selection_scope']='ABD fused TRAINING and validation; checkpoint contains B only; frozen A6/D source paths+hashes in config'
        if keep_weights: torch.save(checkpoint, args.out/f'fold_{f+1}.pt')
        fr = {'fold': f+1, 'frozen_A_D_unchanged':True, 'training_loss_scope':'final ABD fused', 'best_epoch': best_epoch, 'stopped_epoch': stopped_epoch, 'best_checkpoint_predictions_reproduced': True, 'checkpoint_persisted': bool(keep_weights), 'scores': base.metric(truth[validation], pred[validation]), 'solo_scores':last_solo_scores, 'ensemble_delta':best_score-cfg['ensemble_baseline'],
              'seconds': time.perf_counter()-t0, 'loss': history, 'all_validation_rows_covered_exactly_four_times': True,
              'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/1024**2}
        base.write(args.out/f'fold_{f+1}_report.json', fr); reports.append(fr)
        np.save(cache, pred[validation]); np.savez_compressed(args.out/'oof.npz', truth=truth, pred=pred, fold=fold, well=well)
        print('FOLD_RESULT', json.dumps({k: v for k, v in fr.items() if k != 'loss'}), flush=True)
        del probe,policy,table,supervision_weights,weight_trace
        del model, opt, sched, x, labels, state, labeled, train_raw, loss, utility, pack, checkpoint, best_state, best_predictions, companions
        gc.collect(); torch.cuda.empty_cache()
    assert np.isfinite(pred[np.isin(fold,fold_plan(args.fold_count))]).all()
    reports.sort(key=lambda item:item['fold'])
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(base.ROOT))] for p in sources)
    elapsed = time.perf_counter()-start
    screen_scores = base.metric(truth[fold == FOLD_INDEX], pred[fold == FOLD_INDEX])
    screening = {'config': cfg, 'complete_five_folds': args.fold_count == 5,
                 'screening_fold': 5, 'screening_scores': screen_scores, 'screening_passed': qualifies(screen_scores['TOTAL']),
                 'threshold': THRESHOLD, 'folds': reports, 'seconds_this_invocation': elapsed}
    base.write(args.out/'screening_report.json', screening)
    base.write(args.out/'runtime_status.json',{'status':'complete','atlas_current_trial':False,'pid':os.getpid()})
    if args.fold_count == 1:
        assert np.isnan(pred[fold != FOLD_INDEX]).all()
        print('FIFTH_FOLD_SCREEN_ONLY', json.dumps(screen_scores), flush=True)
        return
    assert np.isfinite(pred).all()
    report = {'config': cfg, 'oof_scores': base.metric(truth, pred), 'folds': reports,
              'seconds': sum(r['seconds'] for r in reports), 'seconds_this_invocation': elapsed,
              'timing_scope': 'sum of fold work across screening and continuation; outside-fold startup excluded'}
    base.write(args.out/'report.json', report)
    print('FINAL', json.dumps(report['oof_scores']), flush=True)


if __name__ == '__main__':
    main()
