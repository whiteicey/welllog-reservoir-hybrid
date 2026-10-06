"""CPU-only compact milestone records; never launch or resume training."""
from pathlib import Path
import json, sys, time
import psutil

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
seed, stage = int(sys.argv[1]), sys.argv[2]
own = ROOT / 'outputs/pure_dl_complete_seeds_v362_fold5' / f'seed_{seed}'
dest_stage = own / ('SW351_fixed' if stage == 'SW351' else stage)
report = json.loads((dest_stage / ('fold_5_report.json' if stage=='SOURCE' else 'report.json')).read_text(encoding='utf-8-sig'))
timing = json.loads((dest_stage / 'TIMING.json').read_text(encoding='utf-8-sig'))
assert timing['terminal'] and timing['exit_code'] == 0
assert not psutil.pid_exists(timing['pid'])
batch = json.loads((STUDY / 'BATCH_STATE.json').read_text(encoding='utf-8-sig'))
pid = batch.get('active_pid')
cmd = psutil.Process(pid).cmdline() if pid and psutil.pid_exists(pid) else []
next_own = ROOT / 'outputs/pure_dl_complete_seeds_v362_fold5' / f'seed_{batch.get("seed")}'
next_stage = batch.get('stage', 'none')
config_path = next_own / ('SW351_fixed' if next_stage == 'SW351' else next_stage) / 'config.json'
config = json.loads(config_path.read_text(encoding='utf-8-sig')) if config_path.exists() else {}
stages = ['SOURCE', 'POR113', 'POR151', 'PERM154', 'SW104', 'POR341', 'PERM336', 'SW351']
value = dict(seed=seed, stage=stage, report=report, timing=timing,
             terminal_pid_missing=True, observed_epoch=time.time(),
             next_batch_snapshot=batch, next_actual_cmdline=cmd,
             next_actual_checkpoint_loads=config.get('actual_checkpoint_loads', []),
             completed_stages=stages.index(stage)+1, total_stages=8,
             complete_seed_model=(own/'COMPLETE_MODEL.json').exists(),
             complete_five_folds=False, remaining_folds_started=False)
dest = STUDY/f'SEED{seed}_{stage}_TERMINAL_HANDOFF.json'
dest.write_text(json.dumps(value, indent=2), encoding='utf-8')
plan_path = STUDY/'PLAN.json'
plan = json.loads(plan_path.read_text(encoding='utf-8-sig'))
plan.update(current_stage=batch.get('stage'), current_seed=batch.get('seed'), current_actual_pid=pid)
plan[f'seed{seed}_{stage}_terminal'] = True
plan_path.write_text(json.dumps(plan, indent=2), encoding='utf-8')
if stage=='SOURCE':
    text=(f'\nV362 seed{seed} SOURCE prerequisite completed: scores{json.dumps(report["scores"])}, '
          f'best{report["best_epoch"]}/stop{report["stopped_epoch"]}, '
          f'{report["seconds"]:.3f}s solo/{report["peak_gpu_allocated_mib"]:.3f}MiB, '
          f'processwall{timing["process_wall_seconds"]:.3f}s, bestRAMreplay/allvalidationonce. '
          f'Not complete8stage score. Next observed {batch.get("seed")}/{batch.get("stage")}/PID{pid}; '
          'own SOURCE weights/Q only, no historical/otherseed learnedweights. '
          'Five-seed batch pending/nootherfolds/noindependentdiskreload claim.\n')
else:
    text = (f'\nV362 seed{seed} completed {stage}: target{report["target_score"]}, '
        f'best{report["best_epoch"]}/stop{report["stopped_epoch"]}, '
        f'owninitialdelta{report["against_initial"]}; '
        f'{report["seconds"]:.3f}s solo/{report["peak_MiB"]:.3f}MiB. '
        f'RAMbest{report["validation_rows"]}/frozenexact, not independent diskreload. '
        f'{value["completed_stages"]}/8 stages; next observed '
        f'{batch.get("seed")}/{batch.get("stage")}/PID{pid}. '
        'Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.\n')
for p in (STUDY/'README.md', STUDY/'PROJECTS.md', ROOT/'PROJECTS.md', ROOT.parent/'PROJECTS.md'):
    with p.open('a', encoding='utf-8') as h: h.write(text)
print(json.dumps(value, indent=2))
