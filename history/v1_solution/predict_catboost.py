#!/usr/bin/env python3
"""Generate official-format predictions using trained CatBoost models."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor

from src.data import TARGET_COLUMNS, load_split
from src.features import FeatureBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/models"))
    parser.add_argument("--metadata", type=Path, default=Path("outputs/model_meta.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/result.json"))
    return parser.parse_args()


def inverse_transform_target(
    values: np.ndarray,
    target: str,
) -> np.ndarray:
    if target == "POR":
        return np.expm1(values)
    if target == "PERM":
        return np.power(10.0, values)
    return values.astype(np.float64)


def main() -> None:
    args = parse_args()
    with args.metadata.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    train_records = load_split(args.data_dir, "train")
    builder = FeatureBuilder(
        use_well_context=bool(metadata.get("well_context", True))
    ).fit([frame for _, frame in train_records])

    test_records = load_split(args.data_dir, "test")
    models: dict[str, tuple[CatBoostClassifier, CatBoostRegressor]] = {}
    for target in TARGET_COLUMNS:
        classifier = CatBoostClassifier()
        classifier.load_model(args.model_dir / f"{target}_classifier.cbm")
        regressor = CatBoostRegressor()
        regressor.load_model(args.model_dir / f"{target}_regressor.cbm")
        models[target] = (classifier, regressor)

    result_data: list[dict[str, object]] = []
    total_rows = 0
    for log_id, frame in test_records:
        features = builder.transform(frame)
        depth = frame["DEPTH"].to_numpy(dtype=np.float64)
        if len(depth) != len(features):
            raise RuntimeError(f"Feature/depth length mismatch for {log_id}")

        predictions: list[dict[str, float]] = []
        target_predictions: dict[str, np.ndarray] = {}
        for target in TARGET_COLUMNS:
            classifier, regressor = models[target]
            default_probability = classifier.predict_proba(features)[:, 1]
            regression_prediction = regressor.predict(features)
            parameters = metadata["global_params"][target]
            calibrated = inverse_transform_target(
                regression_prediction * parameters["scale"] + parameters["offset"],
                target,
            )
            low, high = metadata["target_bounds"][target]
            calibrated = np.clip(calibrated, low, high)
            prediction = np.where(
                default_probability >= parameters["threshold"],
                metadata["default_values"][target],
                calibrated,
            )
            target_predictions[target] = prediction

        for index in range(len(depth)):
            prediction = {
                "depth": round(float(depth[index]), 6),
            }
            for target in TARGET_COLUMNS:
                value = float(target_predictions[target][index])
                if not math.isfinite(value):
                    raise RuntimeError(f"Non-finite prediction for {log_id}, {target}, row {index}")
                prediction[target] = round(value, 6)
            predictions.append(prediction)

        result_data.append({
            "logId": log_id,
            "predictions": predictions,
        })
        total_rows += len(predictions)

    result = {
        "modelId": "",
        "modelName": metadata.get("model_name", "catboost_two_stage"),
        "version": metadata.get("version", "1.0"),
        "resultData": result_data,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {args.output}")
    print(f"Wells: {len(result_data)}")
    print(f"Rows: {total_rows}")


if __name__ == "__main__":
    main()
