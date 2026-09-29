"""Standard-library regression tests for the local artifact audit."""
from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import audit_v35
from audit_v35 import (compare_predictions, parse_json, result_from_zip,
                       source_integrity, training_label_audit, validate_output_dir,
                       validate_result, write_reports)


class AuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.depths = {"well": [1000.0, 1000.1]}
        self.result = {
            "modelId": "", "modelName": "fixture", "version": "3.0",
            "resultData": [{"logId": "well", "predictions": [
                {"depth": 1000.0, "POR": 0.1, "PERM": 0.01, "SW": 99.9},
                {"depth": 1000.1, "POR": 10.0, "PERM": 1.0, "SW": 50.0},
            ]}],
        }

    def test_valid_result_and_no_accuracy_claim(self) -> None:
        report = validate_result(self.result, self.depths)
        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["targets"]["SW"]["default_fraction"], 0.5)
        self.assertNotIn("score", report)

    def test_duplicate_well_is_rejected(self) -> None:
        self.result["resultData"].append(copy.deepcopy(self.result["resultData"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate well"):
            validate_result(self.result, self.depths)

    def test_row_count_and_depth_order_are_checked(self) -> None:
        for rows in ([1000.0], [1000.1, 1000.0]):
            with self.subTest(depths=rows), self.assertRaises(ValueError):
                validate_result(self.result, {"well": rows})

    def test_booleans_nan_and_range_errors_are_rejected(self) -> None:
        for value in (True, float("nan"), float("inf"), -9999.0, 51.0):
            with self.subTest(value=value):
                result = copy.deepcopy(self.result)
                result["resultData"][0]["predictions"][0]["POR"] = value
                with self.assertRaises(ValueError):
                    validate_result(result, self.depths)

    def test_extra_fields_and_missing_wells_are_rejected(self) -> None:
        extra = copy.deepcopy(self.result)
        extra["resultData"][0]["predictions"][0]["extra"] = 1
        with self.assertRaises(ValueError):
            validate_result(extra, self.depths)
        with self.assertRaises(ValueError):
            validate_result(self.result, {"another-well": [1000.0, 1000.1]})

    def test_strict_json(self) -> None:
        for text in ('{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_json(text)

    def test_zip_requires_one_root_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("result.json", json.dumps(self.result))
            self.assertEqual(result_from_zip(path)[0], self.result)
            with zipfile.ZipFile(path, "a") as archive:
                archive.writestr("extra.txt", "extra")
            with self.assertRaises(ValueError):
                result_from_zip(path)

    def test_comparison_reports_direction_not_accuracy(self) -> None:
        changed = copy.deepcopy(self.result)
        changed["resultData"][0]["predictions"][1]["POR"] = 0.1
        summary, csv_text = compare_predictions(self.result, changed)
        self.assertEqual(summary["POR"]["default_disagreement_count"], 1)
        self.assertEqual(summary["POR"]["left_continuous_right_default_count"], 1)
        self.assertEqual(summary["SW"]["changed_count"], 0)
        self.assertIn("well_id,target,rows", csv_text)
        changed["resultData"][0]["predictions"][1]["depth"] = 1001.0
        with self.assertRaises(ValueError):
            compare_predictions(self.result, changed)

    def test_legacy_label_policy_accepts_bad_negative_labels(self) -> None:
        content = io.StringIO()
        content.write("DEPTH,POR,PERM,SW\nm,percent,mD,percent\n")
        content.write("1000,0.1,0.01,99.9\n")
        content.write("1001,-99999,-99999,-99999\n")
        content.write("1002,-9999,-9999,-9999\n")
        content.write("1003,-2001.673,-2001.745,-1921.853\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("data/train/well.txt", content.getvalue())
            report = training_label_audit(
                path, {"POR": (0, 50), "PERM": (0, 10000), "SW": (0, 100)}, -99999)
        self.assertEqual(report["baseline_complete_target_rows"], 1)
        self.assertEqual(report["v35_has_all_target_rows"], 3)
        self.assertEqual(report["v35_accepts_invalid_complete_rows"], 2)

    def test_source_identity_omits_only_bytecode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            code = b'"""Fixture."""\nx = 1\n'
            (source / "main.py").write_bytes(code)
            archive_path = root / "source.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("v3_solution/main.py", code)
                archive.writestr("v3_solution/__pycache__/main.pyc", b"bytecode")
            report = source_integrity(archive_path, source)
            self.assertEqual(report["preserved_files"], 1)
            self.assertEqual(report["excluded_bytecode_files"], 1)
            (source / "main.py").write_bytes(b"x = 2\n")
            with self.assertRaisesRegex(ValueError, "source changed"):
                source_integrity(archive_path, source)

    def test_non_bytecode_cache_file_is_still_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            cache = source / "__pycache__"
            cache.mkdir(parents=True)
            (cache / "metadata.txt").write_text("original", encoding="utf-8")
            path = root / "source.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("v3_solution/__pycache__/metadata.txt", "original")
            self.assertEqual(source_integrity(path, source)["preserved_files"], 1)
            (cache / "metadata.txt").write_text("modified", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source changed"):
                source_integrity(path, source)

    def test_protected_output_directories_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            for relative in ("experiments/v3.5/source", "experiments/v3.5/source/__pycache__",
                             "submissions/v3.5", "package", ".git"):
                with self.subTest(path=relative), self.assertRaises(ValueError):
                    validate_output_dir(repo, repo / relative)
                self.assertFalse((repo / relative).exists())
            validate_output_dir(repo, repo / "results/v3.5")

    def test_staging_failure_preserves_existing_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "local_audit.json").write_bytes(b"old-json")
            (output / "prediction_diff_by_well.csv").write_bytes(b"old-csv")
            outputs = {"local_audit.json": "new-json", "prediction_diff_by_well.csv": "new-csv"}
            with mock.patch.object(audit_v35.os, "fsync", side_effect=[None, OSError("disk full")]):
                with self.assertRaises(OSError):
                    write_reports(output, outputs)
            self.assertEqual((output / "local_audit.json").read_bytes(), b"old-json")
            self.assertEqual((output / "prediction_diff_by_well.csv").read_bytes(), b"old-csv")
            self.assertFalse(list(output.glob(".audit-*")))

    def test_cli_check_does_not_write_and_rejects_changed_reports(self) -> None:
        report = {"submission_validation": {"v3.5": {"wells": 10, "rows": 95948}},
                  "training_label_audit": {"v35_accepts_invalid_complete_rows": 128}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit"
            argv = ["audit_v35.py", "--data-zip", "fixture.zip", "--output-dir", str(output)]
            with mock.patch.object(audit_v35, "build_report", return_value=(report, "csv\n")):
                with mock.patch("sys.argv", argv), mock.patch("sys.stdout", new_callable=io.StringIO):
                    audit_v35.main()
                before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                          for path in output.iterdir()}
                with mock.patch("sys.argv", argv + ["--check"]), mock.patch("sys.stdout", new_callable=io.StringIO):
                    audit_v35.main()
                after = {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                         for path in output.iterdir()}
                self.assertEqual(before, after)
                (output / "prediction_diff_by_well.csv").write_bytes(b"changed")
                with mock.patch("sys.argv", argv + ["--check"]), mock.patch("sys.stderr", new_callable=io.StringIO):
                    with self.assertRaises(SystemExit) as error:
                        audit_v35.main()
                self.assertEqual(error.exception.code, 1)
                self.assertEqual((output / "prediction_diff_by_well.csv").read_bytes(), b"changed")

    def test_archive_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("v3_solution/../escape.py", "x = 1")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                source_integrity(path, Path(directory))


if __name__ == "__main__":
    unittest.main()
