#!/usr/bin/env python3
"""Infer v3 candidates, with explicit compatibility for v2 checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor

from src.data import TARGET_COLUMNS, load_split
from src.features import FeatureBuilder
from src.decision import por_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--por-decision", choices=["argmax", "utility", "joint"], default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_classifiers(model_dir: Path, names: list[str]) -> list[CatBoostClassifier]:
    models: list[CatBoostClassifier] = []
    for name in names:
        model = CatBoostClassifier()
        model.load_model(model_dir / name)
        models.append(model)
    return models


def load_regressors(model_dir: Path, names: list[str]) -> list[CatBoostRegressor]:
    models: list[CatBoostRegressor] = []
    for name in names:
        model = CatBoostRegressor()
        model.load_model(model_dir / name)
        models.append(model)
    return models


def mean_probability(models: list[CatBoostClassifier], features: np.ndarray) -> np.ndarray:
    return np.mean([model.predict_proba(features)[:, 1] for model in models], axis=0)


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))

    if "preprocessing" in metadata:
        builder = FeatureBuilder()
        builder.medians_ = metadata["preprocessing"]["medians"]
        builder.base_feature_names_ = metadata["base_feature_names"]
        builder.enhanced_feature_names_ = metadata["enhanced_feature_names"]
    else:
        # Compatibility with existing v2 checkpoints only.
        train_records = load_split(args.data_dir, "train")
        builder = FeatureBuilder().fit([frame for _, frame in train_records])
    if builder.base_feature_names_ != metadata["base_feature_names"]:
        raise RuntimeError("Base feature schema differs from the training metadata")
    if builder.enhanced_feature_names_ != metadata["enhanced_feature_names"]:
        raise RuntimeError("Enhanced feature schema differs from the training metadata")

    files = metadata["model_files"]
    por_default_models = load_classifiers(args.model_dir, files["POR_default"])
    por_ordinal_models = load_classifiers(args.model_dir, files["POR_ordinal"])
    perm_default_models = load_classifiers(args.model_dir, files["PERM_default"])
    perm_regression_models = load_regressors(args.model_dir, files["PERM_regressor"])
    sw_default_models = load_classifiers(args.model_dir, files["SW_default"])
    sw_regression_models = load_regressors(args.model_dir, files["SW_regressor"])

    por_centers = np.asarray(metadata["por_ordinal"]["centers_log1p"], dtype=np.float64)
    defaults = metadata["default_values"]
    bounds = metadata["target_bounds"]
    calibration = metadata["calibration"]

    result_data: list[dict[str, object]] = []
    total_rows = 0
    for log_id, frame in load_split(args.data_dir, "test"):
        matrices = builder.transform(frame)
        base = matrices.base
        enhanced = matrices.enhanced
        depth = frame["DEPTH"].to_numpy(dtype=np.float64)

        por_default_probability = mean_probability(por_default_models, base)
        por_ordinal_probability = np.zeros((len(base), len(por_centers)), dtype=np.float64)
        for model in por_ordinal_models:
            classes = model.classes_.astype(int)
            if np.any(classes < 0) or np.any(classes >= len(por_centers)):
                raise ValueError("POR model classes differ from metadata")
            por_ordinal_probability[:, classes] += model.predict_proba(base) / len(por_ordinal_models)
        por = por_prediction(
            por_ordinal_probability, por_default_probability, por_centers,
            calibration["POR"], bounds["POR"], defaults["POR"],
            mode=args.por_decision or metadata.get("por_decision", "argmax"),
        )

        perm_default_probability = mean_probability(perm_default_models, base)
        perm_log_prediction = np.mean(
            [model.predict(base) for model in perm_regression_models], axis=0
        )
        perm_cfg = calibration["PERM"]
        perm_continuous = np.power(
            10.0,
            perm_log_prediction * perm_cfg["scale"] + perm_cfg["offset"],
        )
        perm_continuous = np.clip(perm_continuous, *bounds["PERM"])
        perm = np.where(
            perm_default_probability >= perm_cfg["threshold"],
            defaults["PERM"],
            perm_continuous,
        )

        sw_default_probability = mean_probability(sw_default_models, enhanced)
        sw_prediction = np.mean(
            [model.predict(enhanced) for model in sw_regression_models], axis=0
        )
        sw_cfg = calibration["SW"]
        sw_continuous = np.clip(
            sw_prediction * sw_cfg["scale"] + sw_cfg["offset"],
            *bounds["SW"],
        )
        sw = np.where(
            sw_default_probability >= sw_cfg["threshold"],
            defaults["SW"],
            sw_continuous,
        )

        predictions: list[dict[str, float]] = []
        for index in range(len(depth)):
            row = {
                "depth": round(float(depth[index]), 6),
                "POR": round(float(por[index]), 6),
                "PERM": round(float(perm[index]), 6),
                "SW": round(float(sw[index]), 6),
            }
            if not all(math.isfinite(row[key]) for key in TARGET_COLUMNS):
                raise RuntimeError(f"Non-finite prediction for {log_id}, row {index}")
            predictions.append(row)

        result_data.append({"logId": log_id, "predictions": predictions})
        total_rows += len(predictions)

    mode = args.por_decision or metadata.get("por_decision", "argmax")
    result = {
        "modelId": "",
        "modelName": metadata.get("model_name", "target_specialized_catboost_v2") + "_por_" + mode,
        "version": metadata.get("version", "2.0"),
        "resultData": result_data,
    }
    if args.output.exists():
        raise FileExistsError("Use a new output filename")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wells: {len(result_data)}")
    print(f"Rows: {total_rows}")


if __name__ == "__main__":
    main()
