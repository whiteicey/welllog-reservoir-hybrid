"""Portable, advisory admission check for this repository's GPU jobs.

The process snapshot is not an atomic reservation. Coordinate concurrent starts
externally; this module must not be described as a race-free GPU scheduler.
"""
from __future__ import annotations

import os
import re


MAXIMUM_JOBS = 2
MINIMUM_FREE_GIB = 6.0
_PYTHON_NAME = re.compile(r"^(?:pythonw?|pypy)(?:\d+(?:\.\d+)*)?(?:\.exe)?$")


def _normalize(value: str) -> str:
    return value.replace("\\", "/").lower().rstrip("/")


def is_gpu_script(name: str) -> bool:
    """Recognize native trainers/benchmarks, including older warm-start jobs."""
    name = _normalize(name).rsplit("/", 1)[-1]
    return name.endswith(".py") and (
        name == "train.py"
        or name.startswith(("train_", "benchmark_", "finetune_"))
        or (
            name.startswith("gen_")
            and any(tag in name for tag in ("train", "benchmark", "checks"))
        )
    )


def _training_entrypoint(cmdline: list[str], cwd: str) -> str | None:
    """Match script paths or -m invocations, without counting wait-only parents."""
    if "--help" in cmdline or "-h" in cmdline:
        return None
    for index, argument in enumerate(cmdline[1:], 1):
        if argument == "-c":
            # Arbitrary inline Python cannot be classified from its command line.
            return None
        if argument == "-m" and index + 1 < len(cmdline):
            module = cmdline[index + 1].lower()
            if module == "tools.deep_learning.train":
                return module
            if module.startswith("experiments.pure_dl."):
                name = module.rsplit(".", 1)[-1] + ".py"
                return module if is_gpu_script(name) else None
            return None
        normalized = _normalize(argument)
        if not normalized.endswith(".py"):
            continue
        location = _normalize(cwd) + "/" + normalized
        if location.endswith("/tools/deep_learning/train.py"):
            return argument
        if "/pure_dl/" in location and is_gpu_script(normalized):
            return argument
        # Only the executed script counts; later arguments may be file paths.
        return None
    return None


def check_slot(min_free_gib: float = MINIMUM_FREE_GIB) -> dict:
    """Require at most two recognized Python jobs including self, and 6 GiB free.

    Imports and CUDA access are deferred so importing the launcher or requesting
    --help works without psutil, PyTorch, data, or a GPU.
    """
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("GPU admission requires psutil; install it in the training environment.") from exc

    current_pid = os.getpid()
    other_jobs: dict[int, dict] = {}
    unknown_python_pids: set[int] = set()
    for process in psutil.process_iter(["pid", "name", "exe", "cmdline", "cwd"], ad_value=None):
        try:
            info = process.info
            pid = int(info["pid"])
            if pid == current_pid:
                continue
            names = (info.get("name") or "", (info.get("exe") or "").replace("\\", "/").rsplit("/", 1)[-1])
            if not any(_PYTHON_NAME.fullmatch(name.lower()) for name in names):
                continue
            command = info.get("cmdline")
            if not command:
                unknown_python_pids.add(pid)
                continue
            entrypoint = _training_entrypoint(command, info.get("cwd") or "")
            if entrypoint:
                other_jobs[pid] = {"pid": pid, "script": entrypoint}
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except psutil.AccessDenied:
            unknown_python_pids.add(process.pid)
    if unknown_python_pids:
        raise RuntimeError(
            "Cannot inspect Python processes for GPU admission: "
            f"{sorted(unknown_python_pids)}. Check these processes before launching."
        )
    active_count = len(other_jobs) + 1
    if active_count > MAXIMUM_JOBS:
        raise RuntimeError(
            f"GPU job limit is {MAXIMUM_JOBS}, including this process; "
            f"other jobs: {list(other_jobs.values())}. Wait for a slot."
        )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Training requires CUDA-enabled PyTorch in the selected environment.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use the normal environment with GPU PyTorch.")
    free, total = torch.cuda.mem_get_info()
    if free < min_free_gib * 1024**3:
        raise RuntimeError(f"Only {free / 1024**3:.2f} GiB free; require {min_free_gib:.1f} GiB.")
    return {
        "maximum_concurrent_jobs": MAXIMUM_JOBS,
        "current_pid": current_pid,
        "active_jobs_including_self": active_count,
        "other_jobs": [other_jobs[pid] for pid in sorted(other_jobs)],
        "free_gib": free / 1024**3,
        "total_gib": total / 1024**3,
        "admission_kind": "non_atomic_process_snapshot",
    }
