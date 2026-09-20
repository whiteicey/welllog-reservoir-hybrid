#!/usr/bin/env python3
"""Train and cross-validate the reservoir-parameter regression model."""

from __future__ import annotations

import argparse
import json
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.data import TARGET_COLUMNS, load_split
from src.features import FeatureBuilder
from src.metrics import official_score
from src.model import fit_ridge


DEFAULT_ALPHAS = [100.0, 1000.0, 10000.0, 100000.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, default=Path("models/ridge.npz"))
    parser.add_argument("--report-output", type=Path, default=Path("reports/cv_report.json"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--alphas", type=str, default=",".join(str(x) for x in DEFAULT_ALPHAS))
    parser.add_argument("--target-transforms", type=str, default="log1p,log10,logit")
    parser.add_argument("--no-well-context", action="store_true")
    return parser.parse_args()


def build_matrices(
    builder: FeatureBuilder,
    records: list[tuple[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    feature_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    for _, frame in records:
        valid_mask = frame[TARGET_COLUMNS].notna().all(axis=1).to_numpy()
        if not valid_mask.any():
            continue
        feature_chunks.append(builder.transform(frame)[valid_mask])
        target_chunks.append(frame.loc[valid_mask, TARGET_COLUMNS].to_numpy(dtype=np.float64))
    return np.concatenate(feature_chunks, axis=0), np.concatenate(target_chunks, axis=0)


def assign_folds(well_ids: list[str], folds: int) -> np.ndarray:
    hashed = sorted(range(len(well_ids)), key=lambda index: zlib.crc32(well_ids[index].encode("utf-8")))
    assignment = np.empty(len(well_ids), dtype=np.int32)
    for position, index in enumerate(hashed):
        assignment[index] = position % folds
    return assignment


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2 for cross-validation")

    alphas = [float(value) for value in args.alphas.split(",") if value.strip()]
    target_transforms = [value.strip() for value in args.target_transforms.split(",")]
    if len(target_transforms) != len(TARGET_COLUMNS):
        raise ValueError("--target-transforms must contain three comma-separated values")

    records = load_split(args.data_dir, "train")
    well_ids = [well_id for well_id, _ in records]
    fold_assignment = assign_folds(well_ids, args.folds)

    alpha_scores: dict[float, list[dict[str, float]]] = defaultdict(list)
    fold_report: list[dict[str, object]] = []
    for fold in range(args.folds):
        train_records = [record for record, assignment in zip(records, fold_assignment) if assignment != fold]
        valid_records = [record for record, assignment in zip(records, fold_assignment) if assignment == fold]
        builder = FeatureBuilder(use_well_context=not args.no_well_context).fit(
            [frame for _, frame in train_records]
        )
        x_train, y_train = build_matrices(builder, train_records)
        x_valid, y_valid = build_matrices(builder, valid_records)
        candidates = fit_ridge(
            features=x_train,
            targets=y_train,
            feature_names=builder.feature_names_,
            target_names=TARGET_COLUMNS,
            target_transforms=target_transforms,
            alphas=alphas,
        )

        fold_alpha_scores: dict[float, dict[str, float]] = {}
        for alpha, model in candidates:
            predictions = model.predict(x_valid)
            score = official_score(y_valid, predictions)
            alpha_scores[alpha].append(score)
            fold_alpha_scores[alpha] = score

        best_alpha = max(fold_alpha_scores, key=lambda alpha: fold_alpha_scores[alpha]["TOTAL"])
        best_score = fold_alpha_scores[best_alpha]
        fold_report.append({
            "fold": fold,
            "valid_wells": [well_id for well_id, assignment in zip(well_ids, fold_assignment) if assignment == fold],
            "best_alpha": best_alpha,
            "best_score": best_score,
            "alpha_scores": {
                str(alpha): score
                for alpha, score in sorted(fold_alpha_scores.items())
            },
        })
        print(f"Fold {fold + 1}/{args.folds}: best alpha={best_alpha:g}, total={best_score['TOTAL']:.4f}")

    mean_scores = {
        str(alpha): {
            metric: float(np.mean([scores[metric] for scores in scores_list]))
            for metric in ["POR", "PERM", "SW", "TOTAL"]
        }
        for alpha, scores_list in alpha_scores.items()
    }
    selected_alpha = max(mean_scores, key=lambda alpha: mean_scores[alpha]["TOTAL"])
    selected_alpha_value = float(selected_alpha)

    builder = FeatureBuilder(use_well_context=not args.no_well_context).fit(
        [frame for _, frame in records]
    )
    x_train, y_train = build_matrices(builder, records)
    final_model = fit_ridge(
        features=x_train,
        targets=y_train,
        feature_names=builder.feature_names_,
        target_names=TARGET_COLUMNS,
        target_transforms=target_transforms,
        alphas=[selected_alpha_value],
    )[0][1]
    final_model.save(args.model_output)

    report = {
        "selected_alpha": selected_alpha_value,
        "target_transforms": dict(zip(TARGET_COLUMNS, target_transforms)),
        "well_context": not args.no_well_context,
        "feature_count": len(builder.feature_names_),
        "training_rows": int(len(y_train)),
        "training_wells": len(records),
        "cross_validation": {
            "folds": args.folds,
            "mean_scores_by_alpha": mean_scores,
            "fold_report": fold_report,
        },
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    with args.report_output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"Selected alpha: {selected_alpha_value:g}")
    print(f"CV total score: {mean_scores[selected_alpha]['TOTAL']:.4f}")
    print(f"Model saved: {args.model_output}")
    print(f"Report saved: {args.report_output}")


if __name__ == "__main__":
    main()
