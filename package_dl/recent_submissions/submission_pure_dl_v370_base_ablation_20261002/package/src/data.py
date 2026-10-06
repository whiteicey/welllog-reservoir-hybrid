from __future__ import annotations

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
    """Read one official curve file (two header rows) and remove known non-physical sentinels."""
    frame = pd.read_csv(path, skiprows=[1])
    frame = frame.replace(-99999.0, np.nan)
    for column, (low, high) in VALID_RANGES.items():
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame.loc[(frame[column] < low) | (frame[column] > high), column] = np.nan
    return frame


def load_split(data_dir: Path, split: str) -> list[tuple[str, pd.DataFrame]]:
    """Return [(well_id, frame), ...] for data_dir/<split>/*.txt, sorted by file name."""
    split_dir = data_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")
    records: list[tuple[str, pd.DataFrame]] = []
    for path in sorted(split_dir.glob("*.txt")):
        records.append((path.stem, read_curve(path)))
    if not records:
        raise ValueError(f"No curve files found in {split_dir}")
    return records
