"""Capture the software and source dependencies used by the public reproduction.

Run in the same SSH virtual environment after the experiment. This does not
train a model or alter predictions.
"""
from pathlib import Path
import hashlib
import json
import platform
from datetime import datetime, timezone

import catboost
import matplotlib
import numpy
import pandas
import sklearn
import torch

root = Path("<PROJECT_ROOT>")
out = root / "paper_workspace" / "public_volve" / "environment.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


result = {
    "collected_utc": datetime.now(timezone.utc).isoformat(),
    "collected_after_run": True,
    "python": platform.python_version(),
    "platform": platform.platform(),
    "packages": {
        "catboost": catboost.__version__,
        "scikit-learn": sklearn.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "matplotlib": matplotlib.__version__,
        "torch": torch.__version__,
    },
    "cuda": {
        "available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    },
    "source_dependencies": {},
}
for rel in ["run_transfer_pdda2021.py", "package/src/feature.py", "package/src/utils.py"]:
    p = root / rel
    result["source_dependencies"][rel] = {"sha256": sha(p), "bytes": p.stat().st_size}
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
