#!/usr/bin/env python3
"""Profile the well-log dataset without modifying source files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_COLUMNS = [
    "DEPTH", "GR", "PE", "SP", "CAL", "AC", "DEN", "CNL",
    "RXO", "RT", "DEVI", "AZIM", "BIT", "CASE",
]
TARGET_COLUMNS = ["POR", "PERM", "SW"]
VALID_RANGES = {
    "DEPTH": (500.0, 5000.0),
    "GR": (0.0, 2000.0),
    "PE": (0.0, 20.0),
    "SP": (-500.0, 500.0),
    "CAL": (5.0, 100.0),
    "AC": (100.0, 1000.0),
    "DEN": (1.0, 4.0),
    "CNL": (-30.0, 100.0),
    "RXO": (0.0, 10000.0),
    "RT": (0.0, 10000.0),
    "DEVI": (0.0, 100.0),
    "AZIM": (0.0, 360.0),
    "BIT": (0.0, 50.0),
    "CASE": (0.0, 100.0),
    "POR": (0.0, 50.0),
    "PERM": (0.0, 10000.0),
    "SW": (0.0, 100.0),
}


def read_curve(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, skiprows=[1])
    frame = frame.replace(-99999.0, np.nan)
    for column, (low, high) in VALID_RANGES.items():
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame.loc[(frame[column] < low) | (frame[column] > high), column] = np.nan
    return frame


def summarize(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        summary[column] = {
            "count": int(series.count()),
            "missing": int(series.isna().sum()),
            "missing_rate": float(series.isna().mean()),
            "unique": int(series.nunique(dropna=True)),
            "min": float(series.min()),
            "p01": float(series.quantile(0.01)),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "p99": float(series.quantile(0.99)),
            "max": float(series.max()),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    train_dir = args.data_dir / "train"
    test_dir = args.data_dir / "test"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_frames: list[pd.DataFrame] = []
    for path in sorted(train_dir.glob("*.txt")):
        frame = read_curve(path)
        frame["_well"] = path.stem
        train_frames.append(frame)
    train = pd.concat(train_frames, ignore_index=True)

    test_frames: list[pd.DataFrame] = []
    for path in sorted(test_dir.glob("*.txt")):
        frame = read_curve(path)
        frame["_well"] = path.stem
        test_frames.append(frame)
    test = pd.concat(test_frames, ignore_index=True)

    valid = train.dropna(subset=TARGET_COLUMNS).copy()
    correlation = valid[INPUT_COLUMNS + TARGET_COLUMNS].corr(method="spearman")
    correlation.to_csv(args.output_dir / "spearman_correlation.csv", encoding="utf-8-sig")

    profile = {
        "train": {
            "wells": int(train["_well"].nunique()),
            "rows": int(len(train)),
            "columns": summarize(train, INPUT_COLUMNS + TARGET_COLUMNS),
            "valid_label_rows": int(len(valid)),
            "valid_label_rate": float(len(valid) / len(train)),
            "rows_per_well": {
                "min": int(train.groupby("_well").size().min()),
                "median": float(train.groupby("_well").size().median()),
                "max": int(train.groupby("_well").size().max()),
            },
        },
        "test": {
            "wells": int(test["_well"].nunique()),
            "rows": int(len(test)),
            "columns": summarize(test, INPUT_COLUMNS),
            "rows_per_well": {
                "min": int(test.groupby("_well").size().min()),
                "median": float(test.groupby("_well").size().median()),
                "max": int(test.groupby("_well").size().max()),
            },
        },
        "target_joint_missing": {
            target: int(train[target].isna().sum()) for target in TARGET_COLUMNS
        },
    }
    with (args.output_dir / "data_profile.json").open("w", encoding="utf-8") as handle:
        json.dump(profile, handle, ensure_ascii=False, indent=2)

    print(json.dumps(profile, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
