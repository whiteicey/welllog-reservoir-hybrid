"""Read-only evidence collection for the v17 local-gate experiment."""
from pathlib import Path
import datetime
import json
import sys

import psutil

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]


def show(path: Path, limit: int = 12000) -> None:
    print("\nFILE", path.relative_to(ROOT))
    if not path.exists():
        print("MISSING")
        return
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix == ".json":
        obj = json.loads(text)
        if isinstance(obj, dict):
            obj = {k: v for k, v in obj.items() if k not in {"per_well", "history", "candidate_diagnostics"}}
            text = json.dumps(obj, ensure_ascii=False, indent=2)
    print(text[:limit])
    if len(text) > limit:
        print("TRUNCATED", len(text), "characters")


print("OBSERVED", datetime.datetime.now().astimezone().isoformat())
for name in ["assembled_report.json", "CURRENT_RECIPE.json", "progress.json", "AGENTS.md", "FINAL.md"]:
    show(ROOT / "results/deep_learning/latent_fold5_v16_20260928" / name, 18000 if name == "progress.json" else 12000)
print("\nPROJECT_TAIL")
print("\n".join((ROOT / "PROJECTS.md").read_text(encoding="utf-8-sig").splitlines()[-28:]))
print("\nRELEVANT_SCRIPTS")
for p in (ROOT / "experiments/pure_dl").glob("*.py"):
    if any(t in p.name for t in ["v16", "architecture_zoo", "latent", "gpu_job_guard"]):
        print(p.name, p.stat().st_size)
print("\nRECENT_OUTPUTS")
for p in sorted((ROOT / "outputs").iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:10]:
    print(p.name)
print("\nRELEVANT_PROCESSES")
for proc in psutil.process_iter(["pid", "name", "cmdline"]):
    try:
        row = proc.info
        args = row.get("cmdline") or []
        if "python" in (row.get("name") or "").lower() and any("pure_dl" in x for x in args) and not any("inspect_latent_gate_v17.py" in x for x in args):
            print(json.dumps(row, ensure_ascii=False))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
print("FREE_GIB", psutil.disk_usage(str(ROOT.anchor)).free / 2**30)
for rel in ["results/AGENTS.md", "results/deep_learning/AGENTS.md", "outputs/AGENTS.md"]:
    if (ROOT / rel).exists():
        show(ROOT / rel)
