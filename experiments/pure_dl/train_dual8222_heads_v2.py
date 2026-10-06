"""Fixed fifth-fold independent-head architecture study on retained dual80.221861.

Explicit launches only. New runs never overwrite old experiments. --resume restores
an epoch-boundary training state; it does not load best heads as though they were
last-epoch parameters. No external ensemble or remaining-fold queue is used.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import hashlib
import json
import random
import signal
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.model_selection import GroupKFold
import train_invariant as base
from multiscale_sequence import window_indices
from paired_local_context_router_sequence import Network as Original, retained_indices, calibration_indices
from dual8222_head_structure_v2 import Network
from frozen_companion_fused_loss import channel_loss
from por_t3_patch_loss import probability
from fold5_screening_805 import THRESHOLD
from gpu_job_guard_v2 import check_slot

ROOT = base.ROOT
SOURCE_DIR = ROOT / 'outputs/pure_dl_paired_local_context_router80_sequence'
SOURCE = SOURCE_DIR / 'fold_5.pt'
STOP_REQUESTED = False
OWNED_OUTPUT: Path | None = None


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def atomic_torch(path: Path, value: Any) -> None:
    temp = path.with_name(path.name + '.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def atomic_array(path: Path, value: np.ndarray) -> None:
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream:
        np.save(stream, value)
    os.replace(temp, path)


def cpu_copy(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def stop_after_epoch(_signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print('STOP_REQUESTED: finish current epoch and save resumable state', flush=True)


class SourceEvaluator:
    """The source's single-coverage256-bin/800-action evaluation, without large caches."""
    def __init__(self, data: dict, checkpoint: dict):
        self.data = data
        self.actions, self.utilities, self.edges = [], [], []
        for j, name in enumerate(base.TS):
            ct = np.asarray(checkpoint['centers'][j])
            assert len(ct) == 256
            action = (np.linspace(ct[0], ct[-1], 800) if j == 2 else
                      np.geomspace(max(ct[0]*.9, .001), ct[-1]*1.1, 800))
            action = np.unique(np.r_[base.DEFAULTS[name], np.clip(action, *base.BOUNDS[name])])
            self.actions.append(action)
            utility = base.kernel_matrix(action, np.r_[base.DEFAULTS[name], ct], name).T
            self.utilities.append(torch.tensor(utility, device='cuda', dtype=torch.float32))
            self.edges.append(torch.tensor(checkpoint['density_edges'][j], device='cuda', dtype=torch.float32))
        self.local = np.full(len(data['truth']), -1, np.int64)
        self.local[data['validation']] = np.arange(len(data['validation']))
        self.centers, self.anchors = [], []
        for wi, rows in enumerate(data['row_ids']):
            if data['fold'][rows[0]] != 4:
                continue
            cc = np.arange(data['starts'][wi], data['starts'][wi+1], 128) + 64
            self.centers.extend(cc)
            self.anchors.extend(np.minimum(cc, data['starts'][wi+1]-1))

    @torch.no_grad()
    def __call__(self, model):
        d = self.data
        pred = np.full((len(d['validation']), 3), np.nan, np.float64)
        covered = np.zeros(len(pred), np.int8)
        model.eval()
        for begin in range(0, len(self.centers), 32):
            cc = torch.tensor(self.centers[begin:begin+32], device='cuda')
            anchor = torch.tensor(self.anchors[begin:begin+32], device='cuda')
            ix, exists = window_indices(cc, d['lower'], d['upper'], anchors=anchor)
            relative = torch.tanh((d['depth'][ix]-d['depth'][anchor, None])/10)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                go, raw = model(d['x'][ix], relative, exists, absolute_depth=d['absolute'][ix])
            original = ix[:, 64:192].cpu().numpy()
            rows = d['raw_to_valid'][original]
            use = exists[:, 64:192].cpu().numpy() & (rows >= 0)
            ids = rows[use]
            loc = self.local[ids]
            assert (loc >= 0).all() and (covered[loc] == 0).all()
            assert len(np.unique(loc)) == len(loc)
            covered[loc] += 1
            select = torch.from_numpy(use).cuda()
            gate = go[:, 64:192][select].float().sigmoid()
            rr = raw[:, 64:192][select].float()
            weight, mu = rr[:, :, 0].softmax(-1), rr[:, :, 1]
            sd = F.softplus(rr[:, :, 2]) + .025
            for j in range(3):
                prob = probability(j, weight[:, j], mu[:, j], sd[:, j], self.edges[j])
                joint = torch.cat([gate[:, j:j+1], (1-gate[:, j:j+1])*prob], -1)
                pred[loc, j] = self.actions[j][(joint@self.utilities[j]).argmax(-1).cpu().numpy()]
        assert (covered == 1).all() and np.isfinite(pred).all()
        return pred, base.metric(d['truth'][d['validation']], pred)


def prepare_data(master: dict) -> dict:
    rec = base.load_split(ROOT.parent/'data', 'train')
    ys = [frame[base.TS].to_numpy(np.float64) for _, frame in rec]
    valid = [np.isfinite(y).all(1) for y in ys]
    truth = np.concatenate([y[m] for y, m in zip(ys, valid)])
    wells = np.concatenate([np.full(m.sum(), i, np.int32) for i, m in enumerate(valid)])
    assert len(rec) == 80 and truth.shape == (723556, 3)
    fold = np.full(len(truth), -1, np.int8)
    for f, (_, ix) in enumerate(GroupKFold(5).split(truth, groups=wells)):
        fold[ix] = f
    validation, training = np.flatnonzero(fold == 4), np.flatnonzero(fold != 4)
    assert len(validation) == 144665 and len(training) == 578891
    row_ids = [np.flatnonzero(wells == i) for i in range(80)]
    tr_wells = [i for i in range(80) if fold[row_ids[i][0]] != 4]
    starts = np.r_[0, np.cumsum([len(frame) for _, frame in rec])]
    nraw = int(starts[-1])
    valid_to_raw = np.concatenate([np.flatnonzero(m)+starts[i] for i, m in enumerate(valid)])
    assert np.array_equal(valid_to_raw[validation], master['validation_raw_rows'])
    assert np.array_equal(valid_to_raw[training], master['training_raw_rows'])
    raw_to_valid = np.full(nraw, -1, np.int64)
    raw_to_valid[valid_to_raw] = np.arange(len(truth))
    valid_x, builder, fm, fs, _ = base.prepare_features(rec, tr_wells, valid, row_ids)
    assert np.array_equal(fm, master['feature_median']) and np.array_equal(fs, master['feature_scale'])
    contract = base.feature_contract()
    assert contract['whitelist'] == master['selected_feature_names']
    x = torch.empty(nraw, 206, dtype=torch.float16, device='cuda')
    for i, (_, frame) in enumerate(rec):
        value = builder.transform(frame).enhanced[:, contract['selected_indices']]
        value = np.clip((value-fm)/fs, -8, 8).astype(np.float16)
        x[starts[i]:starts[i+1]] = torch.from_numpy(value).cuda()
    assert torch.equal(x[torch.as_tensor(valid_to_raw, device='cuda')], valid_x)
    del valid_x
    depth = torch.tensor(np.concatenate([frame['DEPTH'].to_numpy(np.float32) for _, frame in rec]), device='cuda')
    lower = torch.tensor(np.concatenate([np.full(len(frame), starts[i]) for i, (_, frame) in enumerate(rec)]), device='cuda')
    upper = torch.tensor(np.concatenate([np.full(len(frame), starts[i+1]) for i, (_, frame) in enumerate(rec)]), device='cuda')
    absolute = ((depth-master['absolute_depth_median'])/master['absolute_depth_scale']).clamp(-8, 8)
    train_raw = torch.as_tensor(valid_to_raw[training], device='cuda')
    labels = torch.zeros(nraw, 3, device='cuda')
    default = torch.zeros(nraw, 3, dtype=torch.bool, device='cuda')
    labeled = torch.zeros(nraw, dtype=torch.bool, device='cuda')
    yy = np.column_stack([(base.TRANSFORM[j](truth[training, j])-master['regression_median'][j])/master['regression_scale'][j] for j in range(3)])
    labels[train_raw] = torch.as_tensor(yy, device='cuda', dtype=torch.float32)
    default[train_raw] = torch.as_tensor(base.dmask(truth[training]), device='cuda')
    labeled[train_raw] = True
    assert not bool(labeled[torch.as_tensor(valid_to_raw[validation], device='cuda')].any())
    qpath = SOURCE_DIR/'fold_5_conflict_weights.npz'
    with np.load(qpath) as trace:
        selected = int(np.flatnonzero(trace['epochs'] < master['best_epoch'])[-1])
        qepoch = int(trace['epochs'][selected])
        raw_q = trace['raw_ids']
        point_q = trace['weights'][selected].min(axis=1)
        assert np.array_equal(np.sort(raw_q), np.sort(valid_to_raw[training]))
    quality = torch.ones(nraw, device='cuda')
    quality[torch.as_tensor(raw_q, device='cuda')] = torch.as_tensor(point_q, device='cuda')
    assert torch.isfinite(quality).all() and float(quality.min()) >= .09999
    qinfo = {'epoch': qepoch, 'minimum': float(point_q.min()), 'maximum': float(point_q.max()),
             'downweighted_training_rows': int((point_q < 1).sum()), 'dynamic_updates': False,
             'sha256': hashlib.sha256(qpath.read_bytes()).hexdigest()}
    return dict(x=x, depth=depth, lower=lower, upper=upper, absolute=absolute, labels=labels,
                default=default, labeled=labeled, quality=quality, qinfo=qinfo, train_raw=train_raw,
                truth=truth, fold=fold, wells=wells, starts=starts, row_ids=row_ids,
                raw_to_valid=raw_to_valid, validation=validation, contract=contract)


def main() -> None:
    global OWNED_OUTPUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--architecture', choices=['linear', 'cross80'], required=True)
    ap.add_argument('--resume', action='store_true', help='Explicit epoch-boundary resume from this run last_state.pt')
    args = ap.parse_args()
    out = args.out.resolve()
    if (out/'fold_5_report.json').exists():
        raise RuntimeError('Completed experiment is immutable')
    if (out/'config.json').exists() and not args.resume:
        raise RuntimeError('Refuse to overwrite existing run; explicit --resume required')
    if args.resume and not (out/'last_state.pt').exists():
        raise RuntimeError('No complete epoch-boundary state to resume')
    out.mkdir(parents=True, exist_ok=True)
    OWNED_OUTPUT = out
    torch.set_num_threads(3)
    admission = check_slot(min_free_gib=3.)
    print('GPU_ADMISSION', json.dumps(admission), flush=True)
    atomic_json(out/'runtime_status.json', {'status':'running', 'pid':os.getpid(), 'atlas_current_trial':True,
                                          'architecture':args.architecture, 'resuming':args.resume})
    signal.signal(signal.SIGINT, stop_after_epoch)
    signal.signal(signal.SIGTERM, stop_after_epoch)
    started = time.perf_counter()
    torch.manual_seed(20261010)
    np.random.seed(20261010)
    random.seed(20261010)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    master = torch.load(SOURCE, map_location='cpu', weights_only=False)
    assert master['best_epoch'] == 52 and abs(master['best_validation_score']-80.22186108303967) < 1e-10
    data = prepare_data(master)
    numeric, conditioning = calibration_indices(master['selected_feature_names'])
    original = Original(retain_indices=retained_indices(master['selected_feature_names']),
                        numeric_indices=numeric, conditioning_indices=conditioning)
    original.load_state_dict(master['state_dict'], strict=True)
    original.cuda().eval().requires_grad_(False)
    model = Network(original, args.architecture).cuda()
    trainable_parameters = Network.parameter_count(args.architecture)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == trainable_parameters
    evaluator = SourceEvaluator(data, master)
    dependency_names = [Path(__file__).name, 'dual8222_head_structure_v2.py', 'train_dual8222_heads80.py', 'frozen_channel_heads.py',
                        'paired_local_context_router_sequence.py', 'paired_detached_router_sequence.py',
                        'frozen_companion_fused_loss.py', 'por_t3_patch_loss.py', 'fold5_screening_805.py']
    cfg = {'screening_fold':5, 'screening_threshold':THRESHOLD, 'baseline_total':master['best_validation_score'],
           'source_backbone':str(SOURCE.relative_to(ROOT)), 'source_backbone_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
           'source_epoch':52, 'model':'dual8222_head_structure_v2', 'architecture':args.architecture,
           'epochs':80, 'steps':64, 'batch_size':64, 'seed':20261006, 'fold_seed':20261010,
           'early_stopping':{'maximum_epochs':80,'minimum_epochs':10,'patience':18,'scope':'independent target'},
           'head_width':0 if args.architecture=='linear' else 80,
           'local_kernel':1, 'trainable_parameters':trainable_parameters,
           'trainable_parameters_per_target':trainable_parameters//3, 'residual_parameters_per_target':trainable_parameters//3-6450,
           'initialization':'copied source linear rows; cross80 adds zero-output two-expert residual; original router fixed',
           'feature_input':'linear:original128 per expert;cross80 also sees concatenated256;no added normalization',
           'loss':'own target dual predictive BCE+masked NLL;fixed source epoch50 pointQ;no auxiliary restart',
           'quality':data['qinfo'], 'learning_rate':0.00003, 'eta_min':0.00001, 'weight_decay':0.01,
           'schedule':'each target cosine80*64; independent AdamW and clip2',
           'validation':'source single coverage;256probability bins;800actions+default;batch32;no external ensemble',
           'comparison':'linear controls independent training;cross80 is93180params versus old pointwise93420;near-budget internal expert interaction',
           'test_access':False, 'remaining_folds_authorized':False,
           'resume':'explicit complete epoch-boundary state; no automatic launcher',
           'source_hashes':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in dependency_names}}
    if args.resume:
        assert json.loads((out/'config.json').read_text(encoding='utf-8')) == cfg, 'Resume config/source changed'
    else:
        atomic_json(out/'config.json', cfg)
    optimizers = [torch.optim.AdamW(model.heads[j].parameters(), lr=3e-5, weight_decay=.01) for j in range(3)]
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(o, 80*64, eta_min=1e-5) for o in optimizers]
    window_rng = torch.Generator(device='cuda').manual_seed(20261010)
    torch.cuda.reset_peak_memory_stats()
    previous_seconds = 0.
    if args.resume:
        state = torch.load(out/'last_state.pt', map_location='cpu', weights_only=False)
        assert state['config'] == cfg
        model.heads.load_state_dict(state['current_head_state_dict'])
        for optimizer, saved in zip(optimizers, state['optimizers']): optimizer.load_state_dict(saved)
        for scheduler, saved in zip(schedulers, state['schedulers']): scheduler.load_state_dict(saved)
        best_states, best_predictions = state['best_states'], state['best_predictions']
        best_scores = np.asarray(state['best_scores'])
        best_epochs, stopped_epochs, bad, active = [state[k] for k in ['best_epochs','stopped_epochs','bad_epochs','active']]
        history, initial_scores = state['history'], state['initial_scores']
        previous_seconds, epoch = state['seconds'], state['epoch']
        torch.set_rng_state(state['rng']['torch_cpu'])
        torch.cuda.set_rng_state_all(state['rng']['torch_cuda'])
        np.random.set_state(state['rng']['numpy'])
        random.setstate(state['rng']['python'])
        window_rng.set_state(state['rng']['window'])
        for j in range(3): model.heads[j].requires_grad_(active[j])
        print('RESUMED_EPOCH_BOUNDARY', epoch, flush=True)
        del state
    else:
        initial_pred, initial_scores = evaluator(model)
        reference = np.load(SOURCE_DIR/'fold_5_pred.npy')
        check = {'different_elements':int(np.count_nonzero(initial_pred != reference)),
                 'score_delta_historical':initial_scores['TOTAL']-cfg['baseline_total'], 'scores':initial_scores}
        # A changed head must not silently discard the declared source function.
        if check['different_elements']:
            source_pred, source_scores = evaluator(original)
            check['different_elements_current_source'] = int(np.count_nonzero(initial_pred != source_pred))
            check['source_replay_scores'] = source_scores
            assert np.array_equal(initial_pred, source_pred), check
        assert abs(check['score_delta_historical']) < 1e-3, check
        atomic_json(out/'initial_verification.json', check)
        best_scores = np.array([initial_scores[c] for c in base.TS])
        best_epochs, stopped_epochs, bad, active = [0]*3, [None]*3, [0]*3, [True]*3
        best_predictions = initial_pred.copy()
        best_states = [cpu_copy(group.state_dict()) for group in model.heads]
        history, epoch = [], 0
        print('INITIAL_SOURCE_REPLAY', json.dumps(check), flush=True)

    def persist(completed_epoch: int) -> None:
        best = {'head_state_dicts':best_states, 'best_epochs':best_epochs.copy(), 'channel_scores':best_scores.tolist(),
                'config':cfg, 'selection_scope':'three fixed independently selected target groups;two internal experts'}
        atomic_torch(out/'heads_best.pt', best)
        atomic_array(out/'fold_5_pred.npy', best_predictions)
        state = {'format_version':1, 'config':cfg, 'epoch':completed_epoch,
                 'current_head_state_dict':cpu_copy(model.heads.state_dict()),
                 'optimizers':[cpu_copy(o.state_dict()) for o in optimizers],
                 'schedulers':[s.state_dict() for s in schedulers], 'best_states':best_states,
                 'best_predictions':best_predictions, 'best_scores':best_scores.tolist(),
                 'best_epochs':best_epochs.copy(), 'stopped_epochs':stopped_epochs.copy(),
                 'bad_epochs':bad.copy(), 'active':active.copy(), 'history':history,
                 'initial_scores':initial_scores, 'seconds':previous_seconds+time.perf_counter()-started,
                 'rng':{'torch_cpu':torch.get_rng_state(), 'torch_cuda':torch.cuda.get_rng_state_all(),
                        'numpy':np.random.get_state(), 'python':random.getstate(), 'window':window_rng.get_state()}}
        atomic_torch(out/'last_state.pt', state)
        atomic_json(out/'fold_5_validation_history.json', history)

    persist(epoch)
    core = torch.zeros(1, 256, dtype=torch.bool, device='cuda'); core[:, 64:192] = True
    for epoch in range(epoch+1, 81):
        if STOP_REQUESTED or not any(active): break
        t0 = time.perf_counter(); model.train()
        sums = np.zeros((3,3)); epoch_active = active.copy()
        for step in range(64):
            anchors = data['train_raw'][torch.randint(len(data['train_raw']), (64,), device='cuda', generator=window_rng)]
            ix, exists = window_indices(anchors, data['lower'], data['upper'])
            relative = torch.tanh((data['depth'][ix]-data['depth'][anchors, None])/10)
            mask = exists & core & data['labeled'][ix]
            for optimizer in optimizers: optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(data['x'][ix], relative, exists, absolute_depth=data['absolute'][ix])
            channels, bce, nll = channel_loss(prediction, data['labels'][ix], data['default'][ix], mask, data['quality'][ix])
            loss = sum(channels[j] for j in range(3) if active[j])/3
            assert torch.isfinite(loss)
            loss.backward()
            if epoch == 1 and step == 0:
                assert all(p.grad is None and not p.requires_grad for p in model.backbone.parameters())
                assert all(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in group.parameters()) for group in model.heads)
                print('GRADIENT_SCOPE_VERIFIED independent heads only;two experts/router frozen;no ABD', flush=True)
            for j in range(3):
                if active[j]:
                    torch.nn.utils.clip_grad_norm_(model.heads[j].parameters(), 2)
                    optimizers[j].step(); schedulers[j].step()
            sums += np.stack([channels.detach().cpu().numpy(), bce.detach().cpu().numpy(), nll.detach().cpu().numpy()])
        pred, scores = evaluator(model)
        for j, name in enumerate(base.TS):
            if not epoch_active[j]: continue
            if scores[name] > best_scores[j]:
                best_scores[j], best_epochs[j], bad[j] = scores[name], epoch, 0
                best_predictions[:, j] = pred[:, j]
                best_states[j] = cpu_copy(model.heads[j].state_dict())
            else: bad[j] += 1
            if epoch >= 10 and bad[j] >= 18:
                active[j], stopped_epochs[j] = False, epoch
                model.heads[j].load_state_dict(best_states[j]); model.heads[j].requires_grad_(False)
        item = {'epoch':epoch, 'official_scores':scores, 'official_total':scores['TOTAL'],
                'channel_train_loss':(sums[0]/64).tolist(), 'channel_gate_bce':(sums[1]/64).tolist(),
                'channel_continuous_nll':(sums[2]/64).tolist(), 'best_channel_scores':best_scores.tolist(),
                'best_epochs':best_epochs.copy(), 'bad_epochs':bad.copy(), 'active_next_epoch':active.copy(),
                'best_assembled_total':float(best_scores@np.array([.3,.35,.35])),
                'next_learning_rates':[o.param_groups[0]['lr'] for o in optimizers], 'seconds':time.perf_counter()-t0}
        history.append(item)
        persist(epoch)
        print('EPOCH_VALIDATION', json.dumps(item), flush=True)
        atomic_json(out/'runtime_status.json', {'status':'running','pid':os.getpid(),'atlas_current_trial':True,
                    'architecture':args.architecture,'last_completed_epoch':epoch,
                    'best_assembled_candidate':item['best_assembled_total']})
        if not any(active): break
    if STOP_REQUESTED:
        atomic_json(out/'runtime_status.json', {'status':'stopped_by_user','pid':os.getpid(),'atlas_current_trial':False,
                    'last_completed_epoch':history[-1]['epoch'] if history else 0,'resume_available':True,'auto_resume':False})
        print('STOPPED_WITH_EPOCH_STATE', flush=True)
        return
    for j in range(3):
        if stopped_epochs[j] is None: stopped_epochs[j] = history[-1]['epoch'] if history else 0
        model.heads[j].load_state_dict(best_states[j])
    final_pred, final_scores = evaluator(model)
    assert np.array_equal(final_pred, best_predictions), 'Fixed three-head assembly must replay selected predictions'
    assert np.allclose([final_scores[c] for c in base.TS], best_scores, rtol=0, atol=1e-10)
    assert all(torch.equal(v.detach().cpu(), master['state_dict'][k]) for k,v in model.backbone.state_dict().items())
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == cfg['source_backbone_sha256']
    assert all(hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==sha for name,sha in cfg['source_hashes'].items())
    report = {'fold':5,'screening_fold':5,'scores':final_scores,'best_epochs':best_epochs,'stopped_epochs':stopped_epochs,
              'initial_scores':initial_scores,'delta_initial':final_scores['TOTAL']-initial_scores['TOTAL'],
              'delta_historical':final_scores['TOTAL']-cfg['baseline_total'],'screening_threshold':THRESHOLD,
              'screening_passed':bool(final_scores['TOTAL']>THRESHOLD),'remaining_folds_started':False,
              'complete_five_folds':False,'assembled_heads_exactly_reproduce_selected_predictions':True,
              'frozen_source_unchanged':True,'all_validation_rows_covered_once':True,
              'architecture':args.architecture,'trainable_parameters':trainable_parameters,
              'seconds':previous_seconds+time.perf_counter()-started,'timing_scope':'per-process setup/train/validation/replay, concurrent work reported separately',
              'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/1024**2}
    atomic_json(out/'fold_5_report.json', report)
    atomic_json(out/'runtime_status.json', {'status':'complete','pid':os.getpid(),'atlas_current_trial':False})
    print('FOLD_RESULT', json.dumps(report), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if OWNED_OUTPUT is not None:
            detail = {'status':'failed','pid':os.getpid(),'atlas_current_trial':False,
                      'error':repr(exc),'traceback':traceback.format_exc(),
                      'resume_available':(OWNED_OUTPUT/'last_state.pt').exists(),'auto_resume':False}
            atomic_json(OWNED_OUTPUT/'failure.json', detail)
            atomic_json(OWNED_OUTPUT/'runtime_status.json', detail)
        raise
