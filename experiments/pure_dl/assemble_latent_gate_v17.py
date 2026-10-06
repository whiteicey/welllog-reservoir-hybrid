"""Final disk replay, three-private-model scoring and routing diagnostics."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import gc
import json
import sys
import time
import numpy as np
import torch
from latent_gate_routing_v17 import Network, MODES
from train_latent_gate_v17 import ROOT, STUDY, REFERENCE
from train_latent_fold5_v16 import load_outer_data
from train_sw_group_v14 import Evaluator as GridEvaluator
from train_independent_channels_v8 import Evaluator as OriginalEvaluator, REFERENCE as HEAD_REFERENCE
from train_dual8222_heads80 import SOURCE, prepare_data, atomic_json, atomic_array
from independent_channels_v8 import Network as PrivateModel
from multiscale_sequence import window_indices
from launch_latent_gate_v17 import active_jobs
from gpu_job_guard_v2 import check_slot
import train_invariant as base


def point_scores(y, pred):
    return 100*np.maximum(0, 1-np.abs(pred-y)/(.05*(np.abs(y)+1e-5)))


@torch.no_grad()
def mechanism_check(model, data):
    anchors = data["fit_raw"][:2]
    ix, valid = window_indices(anchors, data["lower"], data["upper"])
    x = data["x"][ix]
    rel = torch.tanh((data["depth"][ix]-data["depth"][anchors, None])/10)
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        gate, logits = model(x, rel, valid)
        corrupted_x = x.float().masked_fill(~valid[..., None], 100000)
        corrupted_rel = rel.masked_fill(~valid, -100000)
        changed = model(corrupted_x, corrupted_rel, valid)
        padding_error = max(float((a-b).abs().max()) for a, b in zip((gate, logits), changed))
        empty = model(x, rel, torch.zeros_like(valid))
        assert padding_error == 0 and all(bool(torch.isfinite(t).all()) and bool((t==0).all()) for t in empty)
        memory = model.blocks[1]
        memory.disable_global_for_diagnostic = True
        try:
            no_memory_gate, no_memory_logits = model(x, rel, valid)
        finally:
            memory.disable_global_for_diagnostic = False
    gate_delta = float((gate-no_memory_gate)[valid].abs().max())
    density_delta = float((logits-no_memory_logits)[valid].abs().mean())
    if model.mode == "local_gate":
        assert gate_delta == 0
    return dict(training_input_windows=2, padding_invariance_max_error=padding_error,
        empty_window_finite_and_zero=True, memory_bypass_gate_max_error=gate_delta,
        memory_bypass_density_mean_absolute_logit_change=density_delta,
        note="Post-training forward dependency check only, not a retrained ablation or scoring candidate")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if active_jobs():
        raise RuntimeError("Final replay waits until all project trainers have ended")
    for mode in MODES:
        if not (ROOT / "outputs" / f"pure_dl_latent_gate_v17_{mode}_fold5/fold_5_report.json").exists():
            raise RuntimeError(f"Unfinished candidate: {mode}")
    torch.set_num_threads(3)
    check_slot(min_free_gib=3.)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    started = time.perf_counter()
    data = load_outer_data()
    evaluator = GridEvaluator(data, data["monitor_raw"])
    reference_pred = np.load(REFERENCE / "fold_5_pred.npy")
    with np.load(REFERENCE / "oof.npz") as z:
        truth = z["truth"][z["fold"]==4].copy()
        wells = z["well"][z["fold"]==4].copy()
    assert truth.shape == reference_pred.shape == (144665, 3)
    assert np.array_equal(truth[:, 2], data["truth"][data["monitor_raw"]])
    assert np.array_equal(wells, data["wells"][data["monitor_raw"]])
    defaults = data["default_truth"][data["monitor_raw"]].copy()
    models, predictions, reports, timings, checks = {}, {}, {}, [], {}
    for mode in MODES:
        out = ROOT / "outputs" / f"pure_dl_latent_gate_v17_{mode}_fold5"
        cp = torch.load(out / "model_best.pt", map_location="cpu", weights_only=False)
        model = Network(mode, cp["centers"], cp["target_median"], cp["target_scale"]).cuda()
        model.load_state_dict(cp["state_dict"], strict=True)
        model.eval()
        tick = time.perf_counter()
        pred, stats = evaluator(model)
        assert np.array_equal(pred, np.load(out / "channel_pred.npy"))
        report = json.loads((out / "fold_5_report.json").read_text(encoding="utf-8"))
        assert abs(stats["SW"]-report["target_score"]) < 1e-9
        checks[mode] = mechanism_check(model, data)
        models[mode], predictions[mode], reports[mode] = model, pred, report
        timings.append(dict(target="SW", mode=mode, replay_and_small_checks_seconds=time.perf_counter()-tick, all144665rows_exact=True))
        print("SW_RELOADED", mode, stats["SW"], flush=True)
        del cp
    del evaluator, data
    gc.collect()
    torch.cuda.empty_cache()
    master = torch.load(SOURCE, map_location="cpu", weights_only=False)
    heads = torch.load(HEAD_REFERENCE / "readout_checkpoint.pt", map_location="cpu", weights_only=False)
    legacy = prepare_data(master)
    assert np.array_equal(legacy["truth"][legacy["validation"]], truth)
    assert np.array_equal(legacy["wells"][legacy["validation"]], wells)
    manifest = json.loads((REFERENCE / "fold_5_report.json").read_text(encoding="utf-8"))
    fixed = np.zeros((144665, 2), np.float64)
    for c, name in enumerate(("POR", "PERM")):
        choice = next(v for v in manifest["choices"] if v["target"]==name)
        cp = torch.load(ROOT / choice["checkpoint"], map_location="cpu", weights_only=False)
        model = PrivateModel(master, heads, name, choice["architecture"]).cuda()
        model.load_state_dict(cp["state_dict"], strict=True)
        model.eval()
        ev = OriginalEvaluator(legacy, master, c, reference_pred)
        tick = time.perf_counter()
        pred, score, _ = ev(model)
        assert np.array_equal(pred, reference_pred[:, c])
        models[name], fixed[:, c] = model, pred
        timings.append(dict(target=name, replay_seconds=time.perf_counter()-tick, all144665rows_exact=True))
        print("FIXED_RELOADED", name, score, flush=True)
        del ev, cp
    reference_scores = base.metric(truth, reference_pred)
    candidates = []
    for mode in MODES:
        relevant = [models["POR"], models["PERM"], models[mode]]
        stores = [{p.untyped_storage().data_ptr() for p in m.parameters() if p.numel()} for m in relevant]
        assert not any(stores[i] & stores[j] for i in range(3) for j in range(i))
        combined = np.column_stack((fixed, predictions[mode]))
        scores = base.metric(truth, combined)
        assert abs(scores["TOTAL"]-reports[mode]["reference_replacement_total"]) < 1e-8
        atomic_array(STUDY / f"assembled_{mode}_pred.npy", combined)
        candidates.append(dict(mode=mode, scores=scores, delta_v5_TOTAL=scores["TOTAL"]-reference_scores["TOTAL"],
            best_epoch=reports[mode]["best_epoch"], stopped_epoch=reports[mode]["stopped_epoch"],
            three_private_models_reloaded=True, no_shared_parameter_storage=True, POR_PERM_exactly_unchanged=True,
            all144665SW_predictions_exact=True, checkpoint=f"outputs/pure_dl_latent_gate_v17_{mode}_fold5/model_best.pt"))
    diff = point_scores(truth[:, 2], predictions["local_gate"])-point_scores(truth[:, 2], predictions["shared_global"])
    perwell = [dict(well=int(w), n=int((wells==w).sum()), SW_delta=float(diff[wells==w].mean())) for w in np.unique(wells)]
    rng = np.random.default_rng(20261010)
    sampled = rng.integers(0, len(perwell), (10000, len(perwell)))
    n = np.array([v["n"] for v in perwell])
    delta = np.array([v["SW_delta"] for v in perwell])
    replicates = (delta[sampled]*n[sampled]).sum(1)/n[sampled].sum(1)*0.35
    comparison = dict(local_minus_shared_SW=float(diff.mean()), local_minus_shared_TOTAL=float(diff.mean()*0.35),
        default_SW_delta=float(diff[defaults].mean()), nondefault_SW_delta=float(diff[~defaults].mean()),
        default_TOTAL_contribution=float(diff[defaults].sum()/len(diff)*0.35),
        nondefault_TOTAL_contribution=float(diff[~defaults].sum()/len(diff)*0.35),
        well_wins=sum(v["SW_delta"]>0 for v in perwell), well_losses=sum(v["SW_delta"]<0 for v in perwell), per_well=perwell,
        exploratory_selection_unadjusted_TOTAL_95=np.quantile(replicates, [.025, .975]).tolist())
    best = max(candidates, key=lambda v: v["scores"]["TOTAL"])
    adopted = best["scores"]["TOTAL"] > reference_scores["TOTAL"] + 1e-9
    choices = [dict(v) for v in manifest["choices"]]
    if adopted:
        for choice in choices:
            if choice["target"]=="SW":
                choice.update(key="SW_v17_"+best["mode"], architecture=best["mode"], model_module="latent_gate_routing_v17", builder="Network",
                    checkpoint=best["checkpoint"], target_score=best["scores"]["SW"], best_epoch=best["best_epoch"])
    selected = best["scores"] if adopted else reference_scores
    recipe = dict(scores=selected, choices=choices, selected_new_SW=adopted, best_new_candidate=best["mode"],
        preprocessing_metadata="results/deep_learning/latent_fold5_v16_20260928/data_outer_manifest.json" if adopted else None,
        original_fifth_development_selection=True, complete_five_folds=False, remaining_folds_started=False, old_delivery_unchanged=True)
    report = dict(fold=5, rows=144665, reference_scores=reference_scores, candidates=candidates, comparison=comparison,
        selected_scores=selected, selected_new_SW=adopted, all_new_SW_and_fixed_POR_PERM_reloaded=True,
        all_original_row_order_verified=True, implementation_checks=checks, timings=timings,
        setup_and_replay_seconds=time.perf_counter()-started, finished_at=datetime.now().astimezone().isoformat(),
        complete_five_folds=False, remaining_folds_started=False)
    atomic_json(STUDY / "CURRENT_RECIPE.json", recipe)
    atomic_json(STUDY / "assembled_report.json", report)
    print("ASSEMBLED", json.dumps({k: v for k, v in report.items() if k != "comparison"}, ensure_ascii=False), flush=True)
    print("COMPARISON", json.dumps(comparison, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
