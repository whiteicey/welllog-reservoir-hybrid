"""Reject incompatible cached warm-start runs before the trainer writes anything."""
import hashlib
import json
from pathlib import Path


def check_resume_config(actual, expected):
    for key, value in expected.items():
        if key not in actual or actual[key] != value:
            raise ValueError(f'Incompatible warm-start cache: {key}; use a new output directory')


def validate_resume(out, expected):
    out = Path(out)
    config = out / 'config.json'
    if config.exists():
        check_resume_config(json.loads(config.read_text(encoding='utf-8')), expected)
    elif out.exists() and any(out.glob('fold_*')):
        raise ValueError('Orphan fold cache without config.json; use a new output directory')


def build_contract(source, wrapper, base_file, args, digest_key, lr_key, lr):
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    directory = Path(wrapper).resolve().parent
    root = directory.parents[1]
    dependencies = {Path(base_file).resolve(), directory/'train.py', directory/'train_invariant.py'}
    dependencies.update(root/'package/src'/name for name in ['data.py', 'feature.py', 'model.py', 'utils.py'])
    return {
        'warmstart_source': str(source),
        'warmstart_sha256': {str(f): digest(source / f'fold_{f}.pt') for f in range(1, 6)},
        digest_key: digest(wrapper), lr_key: lr,
        'warmstart_base_sha256': digest(base_file),
        'warmstart_guard_sha256': digest(__file__),
        'warmstart_dependencies_sha256': {
            p.relative_to(root).as_posix(): digest(p) for p in sorted(dependencies)
        },
        'epochs': args.epochs, 'steps': args.steps,
        'batch_points': args.batch_points, 'seed': args.seed,
    }
