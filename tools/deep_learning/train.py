"""Portable fixed-fifth-fold launcher for existing, unchanged pure-DL trainers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import platform
import runpy
import sys


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments" / "pure_dl"
METHODS = {
    "calibrated75": ("train_paired_calibrated_view_sequence.py", 75, 80.2),
    "calibrated80": ("train_paired_calibrated_view80_sequence.py", 80, 80.2),
    "router": ("train_paired_detached_router80_sequence.py", 80, 80.2),
    "local-router": ("train_paired_local_context_router80_sequence.py", 80, 80.4),
    "posterior-router": ("train_paired_posterior_router80_sequence.py", 80, 80.4),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the original fixed FIFTH-fold screen with an unchanged pure-DL trainer.",
        epilog=(
            "This launcher does not enable five-fold continuation or create a gate-verification proof. "
            "See tools/deep_learning/README.md. --help needs no data, PyTorch, or GPU."
        ),
    )
    parser.add_argument("--method", required=True, choices=tuple(METHODS),
                        help="calibrated75 reproduces the historical 75-epoch budget; all other methods use 80")
    parser.add_argument("--data_dir", "--data-dir", dest="data_dir", required=True, type=Path,
                        help="directory containing the original train/*.txt files (not the train directory itself)")
    parser.add_argument("--out", required=True, type=Path,
                        help="new or empty output directory; existing results are never resumed or overwritten")
    parser.add_argument("--fold-count", type=int, choices=(1,), default=1,
                        help="only 1 is supported: original fold 5 (zero-based index 4), not fold 1")
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sidecar(path: Path, value: dict, *, exclusive: bool = False) -> None:
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    data_dir = args.data_dir.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if not (data_dir / "train").is_dir():
        parser.error(f"--data_dir must contain a train directory: {data_dir}")
    if not any((data_dir / "train").glob("*.txt")):
        parser.error(f"No original *.txt well files found in {data_dir / 'train'}")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error("--out must be a new or empty directory; use a new path to preserve existing evidence.")
    filename, maximum_epochs, native_threshold = METHODS[args.method]
    trainer = EXPERIMENTS / filename
    if not trainer.is_file():
        parser.error(f"Frozen trainer is missing from this checkout: {trainer}")

    # This file is __main__, not the 'train' module expected by train_invariant.
    # Put the experiment directory first before importing or executing trainers.
    original_path = list(sys.path)
    original_argv = list(sys.argv)
    sys.path.insert(0, str(EXPERIMENTS))
    base = None
    native_guard = None
    original_loader = None
    original_check = None
    sidecar_path = out / "launcher_adjustments.json"
    sidecar_owned = False
    manifest = None
    try:
        base = importlib.import_module("train_invariant")
        original_train = sys.modules.get("train")
        if original_train is None or Path(original_train.__file__).resolve() != EXPERIMENTS / "train.py":
            raise RuntimeError("The 'train' import resolved outside experiments/pure_dl; run this CLI in a fresh Python process.")
        if Path(base.__file__).resolve() != EXPERIMENTS / "train_invariant.py":
            raise RuntimeError("The 'train_invariant' import resolved outside this checkout.")
        native_guard = importlib.import_module("gpu_job_guard_v2")
        if Path(native_guard.__file__).resolve() != EXPERIMENTS / "gpu_job_guard_v2.py":
            raise RuntimeError("The GPU guard import resolved outside this checkout.")
        # This works both for a script invocation and python -m tools.deep_learning.train.
        portable_guard_path = Path(__file__).with_name("gpu_guard.py")
        portable_guard = runpy.run_path(str(portable_guard_path))
        original_loader = base.load_split
        original_check = native_guard.check_slot

        def redirected_load_split(_native_data_dir: Path, split: str):
            if split != "train":
                raise ValueError("This launcher only permits the original train split.")
            return original_loader(data_dir, split)

        manifest = {
            "schema_version": 1,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "status": "starting",
            "method": args.method,
            "trainer": trainer.relative_to(ROOT).as_posix(),
            "data_dir": str(data_dir),
            "output_dir": str(out),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "fold_count": 1,
            "screening_fold": 5,
            "full_cv_enabled": False,
            "creates_gate_verification": False,
            "native_screening_threshold": native_threshold,
            "current_promotion_threshold": 80.4,
            "screening_note": "Native historical screening_passed is not a current promotion proof; no remaining folds are launched.",
            "frozen_training": {
                "seed": 20261006, "fold_seed": 20261010,
                "maximum_epochs": maximum_epochs, "patience": 18, "minimum_epochs": 10,
                "windows_per_step": 64, "steps_per_epoch": 64,
                "learning_rate": 0.0006, "schedule": "cosine",
                "model_loss_preprocessing_scoring": "unchanged original trainer and dependencies",
            },
            "runtime_overrides": {
                "train_invariant.load_split": "same original loader; replace only data directory and restrict split to train",
                "gpu_job_guard_v2.check_slot": "tools/deep_learning/gpu_guard.py; process snapshot, max 2 including self, at least 6 GiB free",
            },
            "sha256": {
                "original_trainer": _sha256(trainer),
                "original_loader_module": _sha256(Path(base.__file__)),
                "original_gpu_guard": _sha256(Path(native_guard.__file__)),
                "launcher": _sha256(Path(__file__)),
                "portable_gpu_guard": _sha256(portable_guard_path),
            },
            "algorithm_dependency_hashes": "Recorded by the original trainer in config.json; launcher overrides are recorded only in this sidecar.",
        }
        out.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also prevents two launches from sharing the same output.
        _write_sidecar(sidecar_path, manifest, exclusive=True)
        sidecar_owned = True
        base.load_split = redirected_load_split
        native_guard.check_slot = portable_guard["check_slot"]
        sys.path.insert(0, str(EXPERIMENTS))
        sys.argv = [str(trainer), "--out", str(out), "--learning-rate", "0.0006",
                    "--schedule", "cosine", "--fold-count", "1"]
        print(f"LAUNCH method={args.method} fixed_fold=5 budget={maximum_epochs}/18 out={out}", flush=True)
        runpy.run_path(str(trainer), run_name="__main__")
        manifest["status"] = "trainer_returned"
    except BaseException as exc:
        if manifest is not None:
            manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            manifest["error_type"] = type(exc).__name__
        raise
    finally:
        if original_loader is not None:
            base.load_split = original_loader
        if original_check is not None:
            native_guard.check_slot = original_check
        sys.argv = original_argv
        sys.path[:] = original_path
        if manifest is not None and sidecar_owned:
            manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
            _write_sidecar(sidecar_path, manifest)


if __name__ == "__main__":
    main()
