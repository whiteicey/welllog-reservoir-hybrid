#!/usr/bin/env python3
"""Train two-stage CatBoost models on the H20 server."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import GroupKFold

from src.data import TARGET_COLUMNS, load_split
from src.features import FeatureBuilder
from src.metrics import official_score


DOMINANT = {"POR": 0.1, "PERM": 0.01, "SW": 99.9}
BOUNDS = {
    "POR": (0.0, 50.0),
    "PERM": (0.01, 10000.0),
    "SW": (8.305, 99.9),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--iterations", type=int, default=1500)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--no-well-context", action="store_true")
    p.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    return p.parse_args()


def dominant_mask(values: np.ndarray, target: str) -> np.ndarray:
    if target == "SW":
        return values >= 99.9
    return values <= DOMINANT[target]


def build_matrix(
    builder: FeatureBuilder,
    records: list[tuple[str, "object"]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, groups = [], [], []
    for well, frame in records:
        mask = frame[TARGET_COLUMNS].notna().all(axis=1).to_numpy()
        if not mask.any():
            continue
        xs.append(builder.transform(frame)[mask])
        ys.append(frame.loc[mask, TARGET_COLUMNS].to_numpy(np.float64))
        groups.append(np.full(int(mask.sum()), well, dtype=object))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(groups)


def fit_classifier(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    folds: int,
    iterations: int,
    depth: int,
    lr: float,
    use_gpu: bool,
    seed: int = 42,
) -> tuple[CatBoostClassifier, list[dict]]:
    oof = []
    splitter = GroupKFold(n_splits=folds)
    final = None
    for fold, (tr, va) in enumerate(splitter.split(x, y, groups)):
        model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=lr,
            loss_function="Logloss",
            random_seed=seed + fold,
            verbose=0,
            allow_writing_files=False,
            **({"task_type": "GPU", "devices": "0"} if use_gpu else {}),
        )
        model.fit(x[tr], y[tr], eval_set=(x[va], y[va]), early_stopping_rounds=100)
        proba = model.predict_proba(x[va])[:, 1]
        oof.append(
            {
                "fold": fold,
                "indices": va,
                "probability": proba,
                "best_iteration": model.get_best_iteration(),
            }
        )
        final = model
    return final, oof


def fit_regressor(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    folds: int,
    iterations: int,
    depth: int,
    lr: float,
    use_gpu: bool,
    target: str,
    seed: int = 42,
) -> tuple[CatBoostRegressor, list[dict]]:
    y_t = np.log10(y) if target == "PERM" else y
    oof = []
    splitter = GroupKFold(n_splits=folds)
    final = None
    for fold, (tr, va) in enumerate(splitter.split(x, y_t, groups)):
        model = CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=lr,
            loss_function="RMSE",
            random_seed=seed + fold + 100,
            verbose=0,
            allow_writing_files=False,
            **({"task_type": "GPU", "devices": "0"} if use_gpu else {}),
        )
        model.fit(x[tr], y_t[tr], eval_set=(x[va], y_t[va]), early_stopping_rounds=100)
        pred = model.predict(x[va])
        if target == "PERM":
            pred = 10**pred
        oof.append(
            {
                "fold": fold,
                "indices": va,
                "prediction": pred,
                "best_iteration": model.get_best_iteration(),
            }
        )
        final = model
    return final, oof


def combine_two_stage(
    y: np.ndarray,
    dominant_prob: np.ndarray,
    regression_pred: np.ndarray,
    target: str,
) -> tuple[np.ndarray, float]:
    best_score = -1.0
    best_threshold = 0.5
    best_pred = None
    low, high = BOUNDS[target]
    for threshold in np.arange(0.10, 0.91, 0.02):
        is_dominant = dominant_prob >= threshold
        pred = np.where(is_dominant, DOMINANT[target], regression_pred)
        pred = np.clip(pred, low, high)
        score = official_score(
            np.column_stack([y, y, y]),
            np.column_stack([pred, pred, pred]),
        )[target]
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_pred = pred
    if best_pred is None:
        best_pred = np.full(len(y), DOMINANT[target])
    return best_pred, best_threshold


def main() -> None:
    args = parse_args()
    use_gpu = not args.cpu
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_split(args.data_dir, "train")
    builder = FeatureBuilder(use_well_context=not args.no_well_context).fit(
        [f for _, f in records]
    )
    x, y, groups = build_matrix(builder, records)
    print(f"Features: {x.shape}, rows: {len(y)}")

    report = {"feature_count": x.shape[1], "rows": len(y), "targets": {}}
    final_predictions = {}

    for target in TARGET_COLUMNS:
        print(f"\n=== {target} ===")
        values = y[:, TARGET_COLUMNS.index(target)]
        mask = dominant_mask(values, target).astype(np.int8)
        print(f"Dominant ratio: {mask.mean():.4f}")

        print("Training classifier...")
        _, clf_oof = fit_classifier(
            x, mask, groups, args.folds, args.iterations, args.depth,
            args.learning_rate, use_gpu,
        )

        non_dom = mask == 0
        print(f"Training regressor on {non_dom.sum()} non-dominant rows...")
        _, reg_oof = fit_regressor(
            x[non_dom], values[non_dom], groups[non_dom],
            args.folds, args.iterations, args.depth,
            args.learning_rate, use_gpu, target,
        )

        prob = np.zeros(len(y))
        reg_pred = np.full(len(y), DOMINANT[target])
        for entry in clf_oof:
            prob[entry["indices"]] = entry["probability"]
        for entry in reg_oof:
            reg_pred[entry["indices"]] = entry["prediction"]

        combined, threshold = combine_two_stage(values, prob, reg_pred, target)
        score = official_score(
            np.column_stack([values, values, values]),
            np.column_stack([combined, combined, combined]),
        )[target]
        print(f"Best threshold: {threshold:.2f}, score: {score:.4f}")

        report["targets"][target] = {
            "threshold": threshold,
            "score": score,
            "dominant_ratio": float(mask.mean()),
            "classifier_best_iterations": [e["best_iteration"] for e in clf_oof],
            "regressor_best_iterations": [e["best_iteration"] for e in reg_oof],
        }
        final_predictions[target] = {"probability": prob, "regression": reg_pred}

    pred_matrix = np.column_stack([
        final_predictions[t]["regression"] for t in TARGET_COLUMNS
    ])
    # Recompute with per-target dominant override
    for i, target in enumerate(TARGET_COLUMNS):
        t = final_predictions[target]
        mask = t["probability"] >= report["targets"][target]["threshold"]
        pred_matrix[:, i] = np.where(mask, DOMINANT[target], t["regression"])
        pred_matrix[:, i] = np.clip(pred_matrix[:, i], *BOUNDS[target])

    total = official_score(y, pred_matrix)
    report["total"] = total
    print(f"\nTotal: {total['TOTAL']:.4f}")
    print(f"POR: {total['POR']:.4f}, PERM: {total['PERM']:.4f}, SW: {total['SW']:.4f}")

    with (args.output_dir / "cv_report.json").open("w") as f:
        json.dump(report, f, indent=2, default=float)
    np.savez_compressed(
        args.output_dir / "oof_predictions.npz",
        y_true=y,
        y_pred=pred_matrix,
        **{f"prob_{t}": final_predictions[t]["probability"] for t in TARGET_COLUMNS},
        **{f"reg_{t}": final_predictions[t]["regression"] for t in TARGET_COLUMNS},
    )
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
