"""Wait for one exact training process, audit, then execute a predeclared blend.

Never starts or restarts training and never chooses weights from scores.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--training-script', help='Exact script basename expected in the live process command line')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--verifier', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--ensemble-out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    assert Path.cwd().resolve() == root
    previous = json.loads((args.previous/'report.json').read_text())['config']['ensemble']
    assert json.loads((args.previous/'verification.json').read_text())['passed']
    runs = previous['runs']+[str(args.out)]
    assert len(set(str(Path(p).resolve()) for p in runs)) == len(runs)
    command = [sys.executable, 'experiments/pure_dl/ensemble.py', '--runs', *runs,
               '--out', str(args.ensemble_out)]
    if previous.get('specialists'):
        command += ['--specialists', *[previous['specialists'][k] for k in ('POR', 'PERM', 'SW')]]
    for target, paths in previous.get('target_sources', {}).items():
        for path in paths:
            command += ['--target-source', target+'='+path]
    plan = {'training_pid': args.pid, 'previous_verified_ensemble': str(args.previous),
            'new_source': str(args.out), 'blend_command': command,
            'selection': 'predeclared add-one equal weighting; no score-dependent branch'}
    destination = args.out/'postprocess_plan.json'
    if destination.exists():
        assert json.loads(destination.read_text()) == plan
    else:
        destination.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    try:
        process = psutil.Process(args.pid)
        expected_script = args.training_script or ('train_neighborhood_ple.py' if args.out.name.endswith('_ple') else 'train_neighborhood.py')
        assert any(Path(arg).name == expected_script for arg in process.cmdline()), 'PID belongs to a different job'
        print('WAITING_FOR', args.pid, expected_script, flush=True)
        while True:
            try:
                process.wait(timeout=30)
                break
            except psutil.TimeoutExpired:
                continue
    except psutil.NoSuchProcess:
        pass
    # A dead process alone is not evidence of success.
    assert (args.out/'report.json').is_file(), 'Training terminated without a complete report; do not restart automatically'
    for fold in range(1, 6):
        for suffix in ('_pred.npy', '_report.json', '_distributions.npz', '.pt'):
            assert (args.out/f'fold_{fold}{suffix}').is_file()
    print('TRAINING_TERMINAL_WITH_COMPLETE_ARTIFACTS', flush=True)
    subprocess.run([sys.executable, 'experiments/pure_dl/verify.py', '--out', str(args.out)], check=True)
    subprocess.run([sys.executable, str(args.verifier)], check=True)
    subprocess.run(command, check=True)
    subprocess.run([sys.executable, 'experiments/pure_dl/verify.py', '--out', str(args.ensemble_out)], check=True)
    subprocess.run([sys.executable, 'experiments/pure_dl/summarize.py'], check=True)
    print('POSTPROCESS_COMPLETE', str(args.ensemble_out), flush=True)


if __name__ == '__main__':
    main()
