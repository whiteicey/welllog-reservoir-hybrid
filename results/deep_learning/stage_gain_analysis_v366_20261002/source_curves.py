"""CPU-only saved-history comparison; do not evaluate labels or alter trainers."""
from pathlib import Path
import hashlib, json, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
paths = {'original': ROOT/'outputs/pure_dl_paired_local_context_router80_sequence'}
paths.update({str(seed): ROOT/'outputs/pure_dl_complete_seeds_v362_fold5'/f'seed_{seed}'/'SOURCE'
              for seed in range(20261010, 20261015)})
original = json.loads((paths['original']/'fold_5_validation_history.json').read_text(encoding='utf-8'))
records = {}; fig, ax = plt.subplots(figsize=(9, 5))
for label, path in paths.items():
    hp = path/'fold_5_validation_history.json'
    if not hp.is_file():
        continue
    history = json.loads(hp.read_text(encoding='utf-8'))
    if not history:
        continue
    cfg = json.loads((path/'config.json').read_text(encoding='utf-8'))
    epochs = [row['epoch'] for row in history]
    assert epochs == list(range(1, len(history)+1))
    best = max(history, key=lambda row: row['official_total'])
    report = path/'fold_5_report.json'
    final = json.loads(report.read_text(encoding='utf-8')) if report.is_file() else None
    if final:
        assert abs(best['official_total']-final['scores']['TOTAL']) < 1e-9
    comparisons = [dict(epoch=a['epoch'], total_gap=a['official_total']-b['official_total'],
                        expert_loss_gap=a['expert_train_loss']-b['expert_train_loss'])
                   for a,b in zip(history, original)]
    first = next((d for d in comparisons if abs(d['total_gap'])>1e-12 or abs(d['expert_loss_gap'])>1e-12), None)
    common = min(len(history), len(original))
    common_old_best = max(original[:common], key=lambda row: row['official_total'])
    stderr = path/'stderr.log'
    records[label] = dict(observed_epochs=len(history), terminal_report_available=bool(final),
                          best_epoch=best['epoch'], best_TOTAL=best['official_total'],
                          base_seed=cfg['seed'], actual_fold_seed=cfg['seed']+4,
                          first_observed_difference=first, matched_epoch_comparisons=comparisons,
                          best_original_at_same_observed_horizon=common_old_best['official_total'],
                          delta_best_same_horizon=best['official_total']-common_old_best['official_total'],
                          historical_trainer_sha256=cfg.get('historical_trainer_sha256'),
                          history_sha256=hashlib.sha256(hp.read_bytes()).hexdigest(),
                          nondeterministic_warning_recorded=stderr.is_file() and 'does not have a deterministic implementation' in stderr.read_text(encoding='utf-8'),
                          history_source=hp.relative_to(ROOT).as_posix())
    running=[]; current=-float('inf')
    for row in history:
        current=max(current,row['official_total']); running.append(current)
    ax.plot(epochs,running,label=label+(' (running snapshot)' if not final else ''))
ax.set(xlabel='Epoch', ylabel='SOURCE best official TOTAL so far',
       title='Original fifth fold: SOURCE trajectories, not final eight-stage scores')
ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
fig.savefig(HERE/'source_learning_curves.png',dpi=160); plt.close(fig)
result=dict(observed_epoch=time.time(), records=records, screening_fold=5,
            complete_five_folds=False, CPU_history_only=True, extra_training_stages=0,
            interpretation=[
                'The first new SOURCE has the same actual fold seed as original; first two epochs agree, epoch3 diverges before the different stopping time.',
                'Both original and first new SOURCE attain their retained best at epoch52; truncating to62 epochs does not remove a better observed historical checkpoint.',
                'CUDA nondeterministic warnings establish that same-seed bitwise reproducibility is not guaranteed, not that this is the sole cause of score differences.',
                'Second-seed future after its stop52 is unobserved; cannot infer that patience18 would or would not recover the original score.',
                'Records without terminal reports are running snapshots; all SOURCE scores are prerequisites, not complete-model improvement.',
                'Config seed is the base seed; actual fixed fifth uses base seed+4. Batch sources have distinct actual seeds20261010..20261014, first matching historical SOURCE.'
            ])
(HERE/'SOURCE_CURVE_ANALYSIS.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:{x:v[x] for x in ('observed_epochs','terminal_report_available','best_epoch','best_TOTAL','actual_fold_seed','first_observed_difference','delta_best_same_horizon')} for k,v in records.items()},indent=2))
