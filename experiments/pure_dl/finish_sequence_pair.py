"""Complete the predeclared architecture/loss pair sequentially, without fusion."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import psutil


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--baseline-pid', type=int, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    assert Path.cwd().resolve() == root
    a = root/'outputs/pure_dl_multiscale_sequence'
    b = root/'outputs/pure_dl_multiscale_sequence_deep'
    out = root/'outputs/pure_dl_sequence_pair'; out.mkdir(exist_ok=True)
    def status(phase, **extra):
        state = {'phase': phase, 'baseline_pid': args.baseline_pid, **extra}
        (out/'state.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
        print(json.dumps(state), flush=True)
    def verify(directory):
        assert (directory/'report.json').exists(), 'Incomplete run; never restart automatically'
        subprocess.run([sys.executable, 'experiments/pure_dl/verify.py', '--out', str(directory)], check=True)
        subprocess.run([sys.executable, 'experiments/pure_dl/verify_multiscale_sequence.py', '--out', str(directory)], check=True)
    status('waiting_for_baseline')
    try:
        process = psutil.Process(args.baseline_pid)
        assert any(Path(arg).name == 'train_multiscale_sequence.py' for arg in process.cmdline())
        while True:
            try:
                process.wait(timeout=30); break
            except psutil.TimeoutExpired:
                continue
    except psutil.NoSuchProcess:
        pass
    status('verifying_baseline'); verify(a)
    b.mkdir(exist_ok=True)
    assert not (b/'config.json').exists(), 'A loss run already exists; inspect before scheduling another'
    with (b/'train.log').open('w', encoding='utf-8') as stdout, (b/'train.err').open('w', encoding='utf-8') as stderr:
        child = subprocess.Popen([sys.executable, '-u', 'experiments/pure_dl/train_multiscale_sequence.py',
                                  '--deep-supervision', '--out', str(b)], stdout=stdout, stderr=stderr)
        status('training_deep_supervision', training_pid=child.pid)
        code = child.wait()
    assert code == 0, f'Loss run exited with code {code}; no automatic restart'
    status('verifying_deep_supervision'); verify(b)
    ca = json.loads((a/'config.json').read_text()); cb = json.loads((b/'config.json').read_text())
    assert not ca['deep_supervision'] and cb['deep_supervision']
    assert ca['auxiliary_weights'] == [] and cb['auxiliary_weights'] == [.2, .1]
    for key in ca:
        if key not in ('deep_supervision', 'auxiliary_weights'):
            assert ca[key] == cb[key], key
    sa = json.loads((a/'verification.json').read_text())['score']
    sb = json.loads((b/'verification.json').read_text())['score']
    result = {'passed': True, 'same_architecture_seed_sampling_budget_and_decoder': True,
              'baseline': sa, 'deep_supervision': sb,
              'score_delta': sb['TOTAL']-sa['TOTAL'], 'target': 82,
              'goal_achieved_by_this_pair': max(sa['TOTAL'], sb['TOTAL']) > 82,
              'selection': 'Two predeclared standalone models; no post-hoc fusion or weights'}
    (out/'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    subprocess.run([sys.executable, 'experiments/pure_dl/summarize.py'], check=True)
    status('complete', comparison=result)


if __name__ == '__main__':
    main()
