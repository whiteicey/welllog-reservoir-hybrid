"""Build the structured submission package (competition template layout) and verify it reproduces result.json.

Run on the server from the v3 project root:
  .venv/bin/python build_submission_v3.py --final-dir outputs/final_v3 --out-dir outputs/submission_v3_structured
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE / "package"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-dir", type=Path, default=Path("outputs/final_v3"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/submission_v3_structured"))
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--skip-repro", action="store_true")
    args = ap.parse_args()
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    stage = out / "code"; shutil.rmtree(stage, ignore_errors=True); stage.mkdir()
    # --- template layout ---
    for f in ("README.md", "predict.py", "requirements.txt", "config.yaml"):
        shutil.copy(PKG / f, stage / f)
    shutil.copytree(PKG / "src", stage / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(args.final_dir / "models", stage / "models")
    shutil.copy(args.final_dir / "model_meta.json", stage / "models" / "model_meta.json")
    result_src = args.final_dir / "result.json"
    full = json.loads(result_src.read_text(encoding="utf-8"))
    first = full["resultData"][0]
    example = {"modelId": full["modelId"], "modelName": full["modelName"], "version": full["version"],
               "resultData": [{"logId": first["logId"], "predictions": first["predictions"][:3]}]}
    (stage / "examples").mkdir()
    (stage / "examples" / "result_example.json").write_text(json.dumps(example, ensure_ascii=False, indent=2), encoding="utf-8")
    # --- result files (unchanged predictions) ---
    shutil.copy(result_src, out / "result.json")
    with zipfile.ZipFile(out / "result.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out / "result.json", "result.json")
    with zipfile.ZipFile(out / "submission_code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(stage).as_posix())
    for name in ("result.zip", "submission_code.zip"):
        assert zipfile.ZipFile(out / name).testzip() is None
    listing = [n for n in zipfile.ZipFile(out / "submission_code.zip").namelist() if not n.startswith("models/")]
    print("zip layout (models/ omitted):", *listing, sep="\n  ")
    sums = {name: {"bytes": (out / name).stat().st_size, "sha256": sha256(out / name)} for name in ("result.json", "result.zip", "submission_code.zip")}
    # --- reproduction from the zip in a clean directory ---
    if not args.skip_repro:
        repro = out / "repro"; shutil.rmtree(repro, ignore_errors=True); repro.mkdir()
        with zipfile.ZipFile(out / "submission_code.zip") as z:
            z.extractall(repro)
        cmd = [args.python, "predict.py", "--data_dir", str(args.data_dir.resolve()), "--output", str((repro / "result.json").resolve())]
        print("running", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=repro, check=True)
        subprocess.run([args.python, "-m", "src.validate", "--data_dir", str(args.data_dir.resolve()), "--result", str((repro / "result.json").resolve()),
                        "--reference", str(result_src.resolve())], cwd=repro, check=True)
        sums["reproduced_result.json"] = {"bytes": (repro / "result.json").stat().st_size, "sha256": sha256(repro / "result.json")}
        sums["reproduction_identical"] = sums["reproduced_result.json"]["sha256"] == sums["result.json"]["sha256"]
        print("REPRODUCTION IDENTICAL:", sums["reproduction_identical"])
        shutil.rmtree(repro / "models", ignore_errors=True)
    (out / "checksums.json").write_text(json.dumps(sums, indent=1))
    print(json.dumps(sums, indent=1))


if __name__ == "__main__":
    main()
