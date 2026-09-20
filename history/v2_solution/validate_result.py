#!/usr/bin/env python3
"""Validate the generated result.json against the official test data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from src.data import load_split


TOP_LEVEL_KEYS = {"modelId", "modelName", "version", "resultData"}
PREDICTION_KEYS = {"depth", "POR", "PERM", "SW"}
BOUNDS = {
    "POR": (0.0, 50.0),
    "PERM": (0.01, 10000.0),
    "SW": (8.305, 99.9),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.result.open("r", encoding="utf-8") as handle:
        result = json.load(handle)

    if set(result) != TOP_LEVEL_KEYS:
        raise ValueError(f"Unexpected top-level keys: {sorted(result)}")
    if not isinstance(result["resultData"], list):
        raise ValueError("resultData must be a list")

    test_records = load_split(args.data_dir, "test")
    if len(result["resultData"]) != len(test_records):
        raise ValueError(
            f"Expected {len(test_records)} wells, got {len(result['resultData'])}"
        )

    total_rows = 0
    for item, (well_id, frame) in zip(result["resultData"], test_records):
        if set(item) != {"logId", "predictions"}:
            raise ValueError(f"Unexpected result item keys for {well_id}: {sorted(item)}")
        if item["logId"] != well_id:
            raise ValueError(f"Log ID mismatch: {item['logId']} != {well_id}")
        predictions = item["predictions"]
        if len(predictions) != len(frame):
            raise ValueError(
                f"Prediction count mismatch for {well_id}: "
                f"{len(predictions)} != {len(frame)}"
            )

        depths = frame["DEPTH"].to_numpy(dtype=float)
        for index, prediction in enumerate(predictions):
            if set(prediction) != PREDICTION_KEYS:
                raise ValueError(
                    f"Unexpected prediction keys for {well_id}, row {index}: "
                    f"{sorted(prediction)}"
                )
            if not math.isclose(prediction["depth"], depths[index], rel_tol=0, abs_tol=1e-5):
                raise ValueError(
                    f"Depth mismatch for {well_id}, row {index}: "
                    f"{prediction['depth']} != {depths[index]}"
                )
            for target, (low, high) in BOUNDS.items():
                value = prediction[target]
                if not math.isfinite(value) or value < low or value > high:
                    raise ValueError(
                        f"Out-of-range {target} for {well_id}, row {index}: {value}"
                    )
        total_rows += len(predictions)

    print(f"Validated {len(test_records)} wells and {total_rows} prediction rows.")
    print("Top-level keys, log IDs, row counts, depths, fields, and bounds are valid.")


if __name__ == "__main__":
    main()
