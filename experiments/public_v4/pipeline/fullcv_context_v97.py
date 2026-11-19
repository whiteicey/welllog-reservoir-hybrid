"""Minimal public context contract for the original stage trainer."""
from pathlib import Path
import hashlib

ROOT = Path('<PROJECT_ROOT>')
STUDY = ROOT / 'paper_workspace/v4_volve/results/v4_root_20261113/stages/study'
RECIPE = ROOT / 'paper_workspace/v4_volve/results/v4_root_20261113/stages/recipe.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_training_context(data):
    if not data.get('quality_ready') or data.get('quality') is None:
        raise ValueError('Public training quality/context was not prepared')
