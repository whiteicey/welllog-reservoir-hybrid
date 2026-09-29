"""Audit archived v3.5 artifacts locally, without importing or training models.

Run from the repository root with Python 3.10+ (standard library only):
    uv run --no-project --offline --python 3.14 python tools/audit_v35.py \
        --data-zip ../data.zip
Add --check to verify the committed reports without rewriting them.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

TARGETS = ("POR", "PERM", "SW")
# Submission sanity ranges, matching src/validate_result.py; not accuracy tests.
BOUNDS = {"POR": (0.0, 50.0), "PERM": (0.01, 10000.0), "SW": (8.3, 99.9)}
DEFAULTS = {"POR": 0.1, "PERM": 0.01, "SW": 99.9}
ROOT_KEYS = {"modelId", "modelName", "version", "resultData"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def parse_json(content: bytes | str) -> Any:
    return json.loads(content, object_pairs_hook=unique_object,
                      parse_constant=reject_constant)


def result_from_zip(path: Path) -> tuple[dict[str, Any], bytes]:
    with zipfile.ZipFile(path) as archive:
        files = [info for info in archive.infolist() if not info.is_dir()]
        require(len(files) == 1 and files[0].filename == "result.json",
                f"{path.name}: expected exactly one root result.json")
        require(files[0].file_size <= 64 * 1024 * 1024,
                f"{path.name}: result exceeds the audit size limit")
        content = archive.read(files[0])
    result = parse_json(content)
    require(isinstance(result, dict), "Result must be a JSON object")
    return result, content


def log_rows(archive: zipfile.ZipFile, name: str) -> Iterator[dict[str, str]]:
    with archive.open(name) as raw, io.TextIOWrapper(
        raw, encoding="utf-8-sig", newline=""
    ) as text:
        reader = csv.DictReader(text)
        require(reader.fieldnames is not None, f"{name}: missing header")
        next(reader, None)  # The second row contains units, not observations.
        yield from reader


def split_names(archive: zipfile.ZipFile, split: str) -> list[str]:
    names = sorted(name for name in archive.namelist()
                   if name.startswith(f"data/{split}/") and name.endswith(".txt"))
    require(bool(names), f"Dataset has no data/{split}/*.txt")
    require(len(names) == len(set(names)), f"Duplicate {split} archive members")
    ids = [PurePosixPath(name).stem for name in names]
    require(len(ids) == len(set(ids)), f"Duplicate {split} well IDs")
    return names


def test_depths(data_zip: Path) -> dict[str, list[float]]:
    with zipfile.ZipFile(data_zip) as archive:
        template = parse_json(archive.read("data/测试数据返回结果格式.json"))
        require(set(template) == ROOT_KEYS, "Dataset template fields changed")
        require(set(template["resultData"][0]["predictions"][0]) ==
                {"depth", *TARGETS}, "Dataset prediction template fields changed")
        result = {
            PurePosixPath(name).stem: [float(row["DEPTH"])
                                     for row in log_rows(archive, name)]
            for name in split_names(archive, "test")
        }
    require(all(depths and all(math.isfinite(d) for d in depths)
                for depths in result.values()), "Invalid test depth column")
    return result


def is_default(target: str, value: float) -> bool:
    return value >= DEFAULTS[target] if target == "SW" else value <= DEFAULTS[target]


def numeric(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def validate_result(result: dict[str, Any], depths: dict[str, list[float]]) -> dict[str, Any]:
    require(set(result) == ROOT_KEYS, "Unexpected top-level result fields")
    require(all(isinstance(result[key], str) for key in ROOT_KEYS - {"resultData"}),
            "Model metadata must contain strings")
    entries = result["resultData"]
    require(isinstance(entries, list), "resultData must be a list")
    require(all(isinstance(entry, dict) and set(entry) == {"logId", "predictions"}
                for entry in entries), "Unexpected well fields")
    ids = [entry["logId"] for entry in entries]
    require(all(isinstance(wid, str) for wid in ids), "Well IDs must be strings")
    require(len(ids) == len(set(ids)), "Duplicate well IDs")
    require(set(ids) == set(depths), "Missing or extra well IDs")
    values: dict[str, list[float]] = {target: [] for target in TARGETS}
    for entry in entries:
        wid, predictions = entry["logId"], entry["predictions"]
        require(isinstance(predictions, list), f"{wid}: predictions must be a list")
        require(len(predictions) == len(depths[wid]), f"{wid}: row count mismatch")
        for i, (row, depth) in enumerate(zip(predictions, depths[wid])):
            require(isinstance(row, dict) and set(row) == {"depth", *TARGETS},
                    f"{wid}:{i}: unexpected prediction fields")
            require(numeric(row["depth"]) and abs(row["depth"] - depth) < 1e-6,
                    f"{wid}:{i}: depth mismatch")
            for target, (low, high) in BOUNDS.items():
                value = row[target]
                require(numeric(value) and low - 1e-9 <= value <= high + 1e-9,
                        f"{wid}:{i}: invalid {target} value {value!r}")
                values[target].append(value)
    require(all(values.values()), "No prediction rows")
    return {
        "passed": True, "wells": len(entries), "rows": len(values["POR"]),
        "targets": {
            target: {
                "min": min(series), "max": max(series),
                "mean": math.fsum(series) / len(series),
                "default_count": sum(is_default(target, v) for v in series),
                "default_fraction": sum(is_default(target, v) for v in series) / len(series),
            } for target, series in values.items()
        },
    }


def compare_predictions(left: dict[str, Any], right: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Compare aligned predictions, never infer which prediction is correct."""
    a = {entry["logId"]: entry["predictions"] for entry in left["resultData"]}
    b = {entry["logId"]: entry["predictions"] for entry in right["resultData"]}
    require(set(a) == set(b), "Comparison well IDs do not match")
    rows: list[dict[str, Any]] = []
    for wid in sorted(a):
        require(len(a[wid]) == len(b[wid]), f"{wid}: comparison row counts differ")
        require(all(abs(x["depth"] - y["depth"]) < 1e-6
                    for x, y in zip(a[wid], b[wid])), f"{wid}: comparison depths differ")
        for target in TARGETS:
            pairs = [(x[target], y[target]) for x, y in zip(a[wid], b[wid])]
            require(bool(pairs), f"{wid}: no rows to compare")
            left_default = [is_default(target, x) for x, _ in pairs]
            right_default = [is_default(target, y) for _, y in pairs]
            flags = list(zip(left_default, right_default))
            rows.append({
                "well_id": wid, "target": target, "rows": len(pairs),
                "changed_count": sum(abs(x - y) > 1e-9 for x, y in pairs),
                "default_disagreement_count": sum(x != y for x, y in flags),
                "left_default_count": sum(left_default),
                "right_default_count": sum(right_default),
                "left_default_right_continuous_count": sum(x and not y for x, y in flags),
                "left_continuous_right_default_count": sum(not x and y for x, y in flags),
                "mean_absolute_prediction_difference": math.fsum(abs(x - y) for x, y in pairs) / len(pairs),
            })
    summary: dict[str, Any] = {}
    count_keys = [key for key in rows[0] if key.endswith("_count")]
    for target in TARGETS:
        selected = [row for row in rows if row["target"] == target]
        n = sum(row["rows"] for row in selected)
        summary[target] = {"rows": n, **{
            key: sum(row[key] for row in selected) for key in count_keys
        }}
        for key in ("changed", "default_disagreement", "left_default", "right_default"):
            summary[target][key + "_fraction"] = summary[target][key + "_count"] / n
        summary[target]["mean_absolute_prediction_difference"] = math.fsum(
            row["mean_absolute_prediction_difference"] * row["rows"] for row in selected
        ) / n
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return summary, stream.getvalue()


def literal_setting(path: Path, name: str) -> Any:
    """Read a literal constant without importing potentially side-effectful code."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing literal setting {name} in {path.name}")


def training_label_audit(data_zip: Path, ranges: dict[str, tuple[float, float]],
                         missing: float) -> dict[str, Any]:
    targets = {
        target: {"baseline_valid": 0, "v35_has": 0, "v35_accepted_out_of_range": 0,
                 "accepted_invalid_values": Counter()} for target in TARGETS
    }
    per_well: list[dict[str, Any]] = []
    invalid_complete = 0
    with zipfile.ZipFile(data_zip) as archive:
        for name in split_names(archive, "train"):
            total = complete = legacy_complete = 0
            for row in log_rows(archive, name):
                total += 1
                values = {target: float(row[target]) for target in TARGETS}
                valid = {target: math.isfinite(value) and ranges[target][0] <= value <= ranges[target][1]
                         for target, value in values.items()}
                legacy = {target: value > missing / 2 for target, value in values.items()}
                complete += all(valid.values())
                legacy_complete += all(legacy.values())
                invalid_complete += all(legacy.values()) and not all(valid.values())
                for target, value in values.items():
                    stats = targets[target]
                    stats["baseline_valid"] += valid[target]
                    stats["v35_has"] += legacy[target]
                    if legacy[target] and not valid[target]:
                        stats["v35_accepted_out_of_range"] += 1
                        stats["accepted_invalid_values"][str(value)] += 1
            per_well.append({"well_id": PurePosixPath(name).stem, "all_rows": total,
                             "baseline_complete_rows": complete, "v35_has_all_rows": legacy_complete})
    return {
        "scope": "Source-policy comparison, not model training or CV recomputation",
        "train_wells": len(per_well), "train_rows": sum(row["all_rows"] for row in per_well),
        "baseline_complete_target_rows": sum(row["baseline_complete_rows"] for row in per_well),
        "v35_has_all_target_rows": sum(row["v35_has_all_rows"] for row in per_well),
        "v35_accepts_invalid_complete_rows": invalid_complete,
        "target_labels": targets, "well_rows": per_well,
    }


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def source_integrity(archive_path: Path, source_dir: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    ignored = 0
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            path = PurePosixPath(info.filename)
            require(not path.is_absolute() and ".." not in path.parts and
                    all(":" not in part and "\\" not in part for part in path.parts) and
                    len(path.parts) >= 2 and path.parts[0] == "v3_solution",
                    "Unsafe or unexpected source archive path")
            if path.suffix == ".pyc":
                ignored += 1
                continue
            relative = PurePosixPath(*path.parts[1:]).as_posix()
            require(relative not in files, f"Duplicate source archive member: {relative}")
            content = archive.read(info)
            require(source_dir.joinpath(relative).read_bytes() == content,
                    f"Archived source changed: {relative}")
            if path.suffix == ".py":
                ast.parse(content, filename=relative)
            files[relative] = {"bytes": len(content), "sha256": sha256(content)}
    actual = {path.relative_to(source_dir).as_posix()
              for path in source_dir.rglob("*") if path.is_file() and path.suffix != ".pyc"}
    require(actual == set(files), "Extracted source file set differs from original archive")
    return {"passed": True, "preserved_files": len(files), "excluded_bytecode_files": ignored,
            "python_syntax_checked": True, "files": files}


def build_report(repo: Path, data_zip: Path) -> tuple[dict[str, Any], str]:
    result_dir = repo / "submissions/v3.5"
    source_dir = repo / "experiments/v3.5/source"
    legacy, legacy_bytes = result_from_zip(repo / "submissions/v3/result.zip")
    current, current_bytes = result_from_zip(result_dir / "result.zip")
    depths = test_depths(data_zip)
    validations = {"v3": validate_result(legacy, depths), "v3.5": validate_result(current, depths)}
    difference, csv_text = compare_predictions(legacy, current)
    return {
        "schema_version": 1,
        "scope": "Local static/data audit only; no training, inference, test scoring, SSH, or platform verification",
        "inputs": {
            "data.zip": fingerprint(data_zip),
            "submissions/v3/result.zip": fingerprint(repo / "submissions/v3/result.zip"),
            "submissions/v3.5/result.zip": fingerprint(result_dir / "result.zip"),
            "submissions/v3.5/solution.original.zip": fingerprint(result_dir / "solution.original.zip"),
        },
        "result_json": {
            "v3": {"bytes": len(legacy_bytes), "sha256": sha256(legacy_bytes)},
            "v3.5": {"bytes": len(current_bytes), "sha256": sha256(current_bytes),
                     "metadata": {key: current[key] for key in sorted(ROOT_KEYS - {"resultData"})}},
        },
        "source_integrity": source_integrity(result_dir / "solution.original.zip", source_dir),
        "submission_validation": validations,
        "prediction_comparison": {"left": "v3", "right": "v3.5", "targets": difference,
                                  "accuracy_or_complementarity_inferred": False},
        "training_label_audit": training_label_audit(
            data_zip, literal_setting(repo / "package/src/data.py", "VALID_RANGES"),
            literal_setting(source_dir / "src/config.py", "MISSING")),
        "generated_files": {"prediction_diff_by_well.csv": {
            "bytes": len(csv_text.encode("utf-8")), "sha256": sha256(csv_text.encode("utf-8"))}},
        "independent_model_inference": {"performed": False, "reason": "Matching v3.5 weights and decision artifacts were not supplied"},
    }, csv_text


def validate_output_dir(repo: Path, output: Path) -> None:
    destination = output.resolve()
    for protected in (repo / "experiments/v3.5/source", repo / "submissions", repo / "package", repo / ".git"):
        require(not destination.is_relative_to(protected.resolve()),
                f"Output directory overlaps protected input: {protected}")


def write_reports(output: Path, outputs: dict[str, str]) -> None:
    """Stage all bytes first; publish JSON last with the companion CSV hash.

    Each replacement is atomic, not the pair. An interrupted second replacement
    leaves the previous complete JSON; its CSV hash and --check expose a mismatch.
    """
    output.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {}
    try:
        for name, content in outputs.items():
            with tempfile.NamedTemporaryFile(dir=output, prefix=".audit-", suffix=".tmp",
                                             delete=False) as temporary:
                staged[name] = Path(temporary.name)
                temporary.write(content.encode("utf-8"))
                temporary.flush()
                os.fsync(temporary.fileno())
        order = [name for name in outputs if name != "local_audit.json"] + ["local_audit.json"]
        for name in order:
            os.replace(staged[name], output / name)
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check", action="store_true", help="Compare existing reports without writing")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output_dir or repo / "results/v3.5"
    try:
        validate_output_dir(repo, output)
        report, csv_text = build_report(repo, args.data_zip)
        outputs = {
            "local_audit.json": json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            "prediction_diff_by_well.csv": csv_text,
        }
        if args.check:
            for name, content in outputs.items():
                path = output / name
                require(path.read_bytes() == content.encode("utf-8"), f"Report differs: {path}")
        else:
            write_reports(output, outputs)
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile, SyntaxError) as error:
        parser.exit(1, f"Audit failed: {error}\n")
    validation = report["submission_validation"]["v3.5"]
    invalid = report["training_label_audit"]["v35_accepts_invalid_complete_rows"]
    print(f"PASS: {validation['wells']} wells, {validation['rows']} aligned prediction rows")
    print(f"Source preserved; legacy label policy accepts {invalid} out-of-range complete rows")
    print("Reports match" if args.check else "Reports written")


if __name__ == "__main__":
    main()
