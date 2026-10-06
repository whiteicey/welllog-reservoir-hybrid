"""Train two predeclared private-SW readout routes on original fold 5.

Uses the unchanged v16 train-only preprocessing cache and v14 evaluator.
No extra folds, prediction blending, test data, or automatic restarts.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json
import random
import signal
import time
import traceback
import numpy as np
import torch
from latent_gate_routing_v17 import Network, MODES
from train_latent_fold5_v16 import load_outer_data, STUDY as DATA_STUDY
from train_sw_group_v14 import Evaluator, loss_terms
from multiscale_sequence import window_indices
from train_dual8222_heads80 import atomic_json, atomic_torch, atomic_array, cpu_copy
from gpu_job_guard_v2 import check_slot

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "results/deep_learning/latent_gate_v17_20260928"
REFERENCE = ROOT / "outputs/pure_dl_independent_v8_selected_fold5"
STOP = False
OWNED = None


def request_stop(*_):
    global STOP
    STOP = True
    print("STOP_REQUESTED: save at epoch boundary; explicit resume only", flush=True)


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    global OWNED
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    out = args.out.resolve()
    if (out / "fold_5_report.json").exists():
        raise RuntimeError("Completed run is immutable")
    if (out / "config.json").exists() and not args.resume:
        raise RuntimeError("Existing run requires explicit --resume")
    out.mkdir(parents=True, exist_ok=True)
    OWNED = out
    torch.set_num_threads(3)
    print("ADMISSION", check_slot(min_free_gib=3.0), flush=True)
    atomic_json(out / "runtime_status.json", dict(status="running", mode=args.mode, pid=os.getpid(), atlas_current_trial=True))
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started = time.perf_counter()
    random.seed(20261010)
    np.random.seed(20261010)
    torch.manual_seed(20261010)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    data = load_outer_data()
    manifest_path = DATA_STUDY / "data_outer_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["original_validation_row_order_verified"]
    assert not manifest["outer_fifth_used_for_fitting"]
    model = Network(args.mode, data["centers"], float(data["target_median"]), float(data["target_scale"])).cuda()
    prior = model.initialize_training_priors(dict(train_raw=data["fit_raw"], labels=data["labels"], default=data["default"], quality=torch.ones(len(data["truth"]), device="cuda")))
    initial_hash = state_hash(model)
    initial_body = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if not n.startswith("output.")}
    reference = json.loads((REFERENCE / "fold_5_report.json").read_text(encoding="utf-8"))["scores"]
    parameter_count = sum(p.numel() for p in model.parameters())
    assert parameter_count == 174849
    source_paths = [Path(__file__), Path(__file__).with_name("latent_gate_routing_v17.py"), Path(__file__).with_name("architecture_zoo_v15.py"), Path(__file__).with_name("train_sw_group_v14.py"), Path(__file__).with_name("independent_sw_scratch_v10.py"), Path(__file__).with_name("train_latent_fold5_v16.py")]
    cfg = dict(mode=args.mode, target="SW", screening_fold=5, model_module="latent_gate_routing_v17", trainable_parameters=parameter_count,
        hidden_width=64, latent_count=16, attention_heads=4, input_features=206, relative_depth=1, window_length=256,
        default_features="local_before_and_after_convs" if args.mode == "local_gate" else "global_latent_path",
        density_features="global_latent_path", local_convolution_weights_shared_within_SW=True,
        both_feature_paths_computed=True, backward_paths_intentionally_differ=True,
        parameter_matched_comparison=True, initial_state_sha256=initial_hash, initialization=prior,
        pretrained_network_used=False, other_target_parameters_used=False, seed=20261010,
        epochs=80, patience=18, minimum_epochs=10, steps_per_epoch=64, batch_size=64,
        lr=0.0006, eta_min=0.00001, weight_decay=0.01, clip_norm=2.0, Q=1,
        loss="mean(BCE(default)+nondefault soft2bin CE)/3", learning_weight="point1",
        decoder="unchanged256support/800linear-actions+default/singlecoverage/batch32",
        fit_wells=manifest["fit_wells"], monitor_wells=manifest["monitor_wells"], fit_points=578891, monitor_points=144665,
        cache=str((DATA_STUDY / "data_outer.npz").relative_to(ROOT)), data_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        reference_scores=reference, selection="original fifth SW; actual best checkpoint", remaining_folds_started=False,
        deterministic_mode="warn_only; no bitwise training reproducibility claim", torch_version=str(torch.__version__))
    if args.resume:
        assert json.loads((out / "config.json").read_text(encoding="utf-8")) == cfg
    else:
        atomic_json(out / "config.json", cfg)
    train_eval = Evaluator(data, data["fit_raw"].cpu().numpy())
    val_eval = Evaluator(data, data["monitor_raw"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0006, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 80 * 64, eta_min=0.00001)
    windows = torch.Generator(device="cuda").manual_seed(20261010)
    epoch, elapsed, proof = 0, 0.0, None
    if args.resume:
        saved = torch.load(out / "last_state.pt", map_location="cpu", weights_only=False)
        assert saved["config"] == cfg
        model.load_state_dict(saved["current"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        epoch, elapsed, proof = saved["epoch"], saved["seconds"], saved["proof"]
        best, best_epoch = saved["best"], saved["best_epoch"]
        best_state, best_pred = saved["best_state"], saved["best_pred"]
        history, bad = saved["history"], saved["bad"]
        torch.set_rng_state(saved["rng"]["torch"])
        torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
        windows.set_state(saved["rng"]["windows"])
        np.random.set_state(saved["rng"]["numpy"])
        random.setstate(saved["rng"]["python"])
        del saved
    else:
        pred, initial = val_eval(model)
        atomic_json(out / "initial.json", dict(validation=initial, initial_state_sha256=initial_hash))
        best, best_epoch, best_state, best_pred = initial["SW"], 0, cpu_copy(model.state_dict()), pred.copy()
        history, bad = [], 0

    def persist():
        atomic_torch(out / "model_best.pt", dict(state_dict=best_state, config=cfg, best_epoch=best_epoch, target_score=best,
            centers=data["centers"], target_median=model.target_median, target_scale=model.target_scale))
        atomic_array(out / "channel_pred.npy", best_pred)
        atomic_json(out / "validation_history.json", history)
        atomic_torch(out / "last_state.pt", dict(config=cfg, current=cpu_copy(model.state_dict()), optimizer=cpu_copy(optimizer.state_dict()),
            scheduler=scheduler.state_dict(), epoch=epoch, best=best, best_epoch=best_epoch, best_state=best_state, best_pred=best_pred,
            history=history, bad=bad, proof=proof, seconds=elapsed+time.perf_counter()-started,
            rng=dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(), windows=windows.get_state(), numpy=np.random.get_state(), python=random.getstate())))

    persist()
    torch.cuda.reset_peak_memory_stats()
    core = torch.zeros((1, 256), device="cuda", dtype=torch.bool)
    core[:, 64:192] = True
    for ep in range(epoch + 1, 81):
        if STOP or (epoch >= 10 and bad >= 18):
            break
        model.train()
        tick, loss_sum = time.perf_counter(), 0.0
        for step in range(64):
            anchors = data["fit_raw"][torch.randint(len(data["fit_raw"]), (64,), device="cuda", generator=windows)]
            ix, valid = window_indices(anchors, data["lower"], data["upper"])
            relative = torch.tanh((data["depth"][ix] - data["depth"][anchors, None]) / 10)
            mask = valid & core & data["fitmask"][ix]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(data["x"][ix], relative, valid)
            loss = loss_terms(model, output, data, ix, mask, "point")
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            if proof is None and step == 1:
                grads = [p.grad.detach().float().square().sum() for n, p in model.named_parameters() if n.startswith("blocks.") and p.grad is not None]
                memory_grads = [p.grad.detach().float().square().sum() for n, p in model.named_parameters() if n.startswith("blocks.1.") and p.grad is not None]
                norm = float(torch.stack(grads).sum().sqrt())
                memory_norm = float(torch.stack(memory_grads).sum().sqrt())
                assert norm > 0 and memory_norm > 0 and np.isfinite(norm)
                proof = dict(backbone_gradient_norm=norm, memory_gradient_norm=memory_norm,
                    only_original64_training_labels=True, private_SW=True, other_target_losses=False)
                atomic_json(out / "gradient_scope.json", proof)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach())
        epoch = ep
        pred, vs = val_eval(model)
        _, ts = train_eval(model)
        if vs["SW"] > best:
            best, best_epoch, best_state, best_pred, bad = vs["SW"], epoch, cpu_copy(model.state_dict()), pred.copy(), 0
        else:
            bad += 1
        row = dict(epoch=epoch, fit=ts, validation=vs, best_target_score=best, best_epoch=best_epoch,
            bad_epochs=bad, training_loss=loss_sum/64, seconds=time.perf_counter()-tick,
            reference_replacement_total=reference["TOTAL"] + 0.35*(vs["SW"]-reference["SW"]))
        history.append(row)
        persist()
        atomic_json(out / "runtime_status.json", dict(status="running", pid=os.getpid(), mode=args.mode, last_completed_epoch=epoch,
            best_target_score=best, best_epoch=best_epoch, reference_replacement_total=reference["TOTAL"]+0.35*(best-reference["SW"]), atlas_current_trial=True))
        print("EPOCH", json.dumps(dict(mode=args.mode, epoch=epoch, SW=vs["SW"], best_SW=best, best_epoch=best_epoch,
            default_SW=vs["default_SW"], numeric_SW=vs["numeric_SW"], seconds=row["seconds"])), flush=True)
    if STOP:
        atomic_json(out / "runtime_status.json", dict(status="stopped_by_user", pid=os.getpid(), mode=args.mode, resume_available=True, auto_resume=False))
        return
    model.load_state_dict(best_state, strict=True)
    pred, vs = val_eval(model)
    fitpred, ts = train_eval(model)
    assert np.array_equal(pred, best_pred)
    atomic_array(out / "fit_pred.npy", fitpred)
    body_change = float(sum((p.detach().cpu().double()-initial_body[n].double()).square().sum() for n, p in model.named_parameters() if n in initial_body).sqrt())
    report = dict(fold=5, screening_fold=5, target="SW", mode=args.mode, scores={"SW": vs["SW"]}, target_score=vs["SW"],
        validation=vs, fit=ts, best_epoch=best_epoch, stopped_epoch=epoch, trainable_parameters=parameter_count,
        reference_replacement_total=reference["TOTAL"]+0.35*(vs["SW"]-reference["SW"]),
        TOTAL_scope="fixed verified POR/PERM; pending actual new SW disk reload and combined scoring",
        delta_old_SW=vs["SW"]-reference["SW"], gradient_scope=proof, selected_backbone_change_L2=body_change,
        mechanism=model.mechanism_statistics(), all144665rows_restored_exactly=True,
        seconds=elapsed+time.perf_counter()-started, peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        complete_five_folds=False, remaining_folds_started=False, finished_at=datetime.now().astimezone().isoformat())
    atomic_json(out / "fold_5_report.json", report)
    atomic_json(out / "runtime_status.json", dict(status="complete", mode=args.mode, pid=os.getpid(), atlas_current_trial=False))
    print("FINAL", json.dumps({k: v for k, v in report.items() if k not in ("fit", "validation")}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if OWNED is not None:
            error = dict(status="failed", pid=os.getpid(), error=repr(exc), traceback=traceback.format_exc(), auto_resume=False)
            atomic_json(OWNED / "failure.json", error)
            atomic_json(OWNED / "runtime_status.json", error)
        raise
