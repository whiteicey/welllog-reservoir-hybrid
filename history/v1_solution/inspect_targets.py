#!/usr/bin/env python3
"""Inspect target distributions and their local curve relationships."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data import INPUT_COLUMNS, TARGET_COLUMNS, load_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    records = load_split(args.data_dir, "train")
    frame = pd.concat([item for _, item in records], ignore_index=True)
    valid = frame.dropna(subset=TARGET_COLUMNS).copy()

    for target in TARGET_COLUMNS:
        print(f"\n=== {target} value counts ===")
        print(valid[target].value_counts().head(30).to_string())
        print(f"unique={valid[target].nunique()}")

    print("\n=== POR bins and median curves ===")
    bins = [-0.01, 0.001, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    valid["POR_BIN"] = pd.cut(valid["POR"], bins=bins, include_lowest=True)
    grouped = valid.groupby("POR_BIN", observed=True)[
        ["GR", "PE", "SP", "CAL", "AC", "DEN", "CNL", "RXO", "RT", "PERM", "SW"]
    ].median()
    grouped["COUNT"] = valid.groupby("POR_BIN", observed=True).size()
    print(grouped.to_string())

    print("\n=== Default-value crosstab ===")
    flags = pd.DataFrame({
        "POR_DEFAULT": valid["POR"] <= 0.2,
        "PERM_DEFAULT": valid["PERM"] <= 0.05,
        "SW_DEFAULT": valid["SW"] >= 99.89,
    })
    print(pd.crosstab(flags["POR_DEFAULT"], [flags["PERM_DEFAULT"], flags["SW_DEFAULT"]]).to_string())

    print("\n=== Feature rank correlation within common POR ranges ===")
    for low, high in [(0.0, 0.2), (0.2, 1.0), (1.0, 5.0), (5.0, 50.0)]:
        subset = valid[(valid["POR"] >= low) & (valid["POR"] < high)]
        correlations = subset[INPUT_COLUMNS + ["PERM", "SW"]].corrwith(subset["POR"], method="pearson")
        print(f"POR [{low}, {high}) n={len(subset)}")
        print(correlations.sort_values(key=abs, ascending=False).head(12).to_string())


if __name__ == "__main__":
    main()
