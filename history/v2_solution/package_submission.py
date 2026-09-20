#!/usr/bin/env python3
"""Package v2 result and reproducible inference code."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def add_file(archive: zipfile.ZipFile, path: Path, arcname: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    archive.write(path, arcname=arcname)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution-dir", type=Path, required=True)
    parser.add_argument("--outputs-dir", type=Path, required=True)
    args = parser.parse_args()

    result_path = args.outputs_dir / "result.json"
    result_zip = args.outputs_dir / "result.zip"
    with zipfile.ZipFile(result_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        add_file(archive, result_path, "result.json")

    code_zip = args.outputs_dir / "submission_code.zip"
    with zipfile.ZipFile(code_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        add_file(archive, args.solution_dir / "README.md", "README.md")
        add_file(archive, args.solution_dir / "requirements.txt", "requirements.txt")
        add_file(archive, args.solution_dir / "train_v2.py", "train.py")
        add_file(archive, args.solution_dir / "predict_v2.py", "predict.py")
        add_file(archive, args.solution_dir / "validate_result.py", "validate_result.py")
        for path in sorted((args.solution_dir / "src").glob("*.py")):
            add_file(archive, path, f"src/{path.name}")
        add_file(archive, args.outputs_dir / "model_meta.json", "models/model_meta.json")
        for path in sorted((args.outputs_dir / "models").glob("*.cbm")):
            add_file(archive, path, f"models/{path.name}")

    for archive_path in (result_zip, code_zip):
        with zipfile.ZipFile(archive_path, "r") as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise RuntimeError(f"Corrupt archive member: {bad_member}")
        print(f"{archive_path}: {archive_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
