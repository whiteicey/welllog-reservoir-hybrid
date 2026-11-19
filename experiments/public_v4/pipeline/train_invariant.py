"""Public Volve adapter over the original v4 invariants.

All model classes and losses are re-exported. Only target names, default-state
semantics, public RMSE and the auxiliary tolerance kernel are changed.
"""
from pathlib import Path
import importlib.util
import numpy as np

_here = Path(__file__).resolve().parent
import sys
sys.path.insert(0, '<PROJECT_ROOT>')
_spec = importlib.util.spec_from_file_location("v4_invariant_original", _here / "train_invariant_original.py")
_original = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_original)
for _name in dir(_original):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_original, _name)

# The adapter lives deeper than the original runtime. Bind the project root
# explicitly so no paper_workspace/package surrogate is inferred.
import sys
ROOT = Path('<PROJECT_ROOT>')
_original.ROOT = ROOT
sys.path.insert(0, str(ROOT / 'package'))

TS = ["PHIF", "SW", "VSH"]
TRANSFORM = [np.log1p, lambda x: x, np.log1p]
INVERSE = [np.expm1, lambda x: x, np.expm1]
DEFAULTS = {"PHIF": 0.0, "SW": 1.0, "VSH": 0.0}
BOUNDS = {"PHIF": (0.0, 1.0), "SW": (0.0, 1.0), "VSH": (0.0, 3.0)}


def dmask(y):
    y = np.asarray(y)
    d = np.zeros(y.shape, dtype=bool)
    d[:, 1] = np.nan_to_num(y[:, 1], nan=0.0) >= .999
    return d


def metric(y, p):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    scores = [float(np.sqrt(np.mean((p[:, j] - y[:, j]) ** 2))) for j in range(3)]
    total = float(np.sqrt(np.mean((p - y) ** 2)))
    return {"TOTAL": -total, "PHIF": -scores[0], "SW": -scores[1], "VSH": -scores[2],
            "rmse_all3": total, "rmse_PHIF": scores[0], "rmse_SW": scores[1], "rmse_VSH": scores[2]}


def kernel_matrix(action, truth, target):
    tol = {"PHIF": .08, "SW": .05, "VSH": .08}[target]
    action = np.asarray(action, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return np.maximum(0.0, 1.0 - np.abs(action[:, None] - truth[None, :]) /
                      (tol * (np.abs(truth[None, :]) + 1e-7)))


def load_split(*args, **kwargs):
    raise RuntimeError("Internal data access is forbidden in the public Volve experiment")


def prepare_features(*args, **kwargs):
    raise RuntimeError("Use prepare_public_fold.py for public feature preparation")
