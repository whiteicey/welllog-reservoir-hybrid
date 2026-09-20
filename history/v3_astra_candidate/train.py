#!/usr/bin/env python3
"""Train an independently versioned v3 candidate; validation is a separate command."""

from __future__ import annotations

import argparse
import hashlib
import time
import json
from pathlib import Path
from typing import Any

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from src.data import TARGET_COLUMNS, load_split
from src.features import FeatureBuilder


DEFAULT_VALUES = {"POR": 0.1, "PERM": 0.01, "SW": 99.9}
TARGET_BOUNDS = {
    "POR": (0.0, 50.0),
    "PERM": (0.01, 10000.0),
    "SW": (8.305, 99.9),
}
CALIBRATION = {
    "POR": {"threshold": 0.5, "scale": 1.0, "offset": 0.0},
    "PERM": {"threshold": 0.37384488065262944, "scale": 1.03, "offset": 0.0},
    "SW": {"threshold": 0.08702066266867681, "scale": 1.05, "offset": -3.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="42,2026")
    parser.add_argument("--por-decision", choices=["argmax", "utility", "joint"], default="utility")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--verbose", type=int, default=100)
    return parser.parse_args()


def model_runtime(cpu: bool) -> dict[str, Any]:
    if cpu:
        return {"task_type": "CPU", "thread_count": -1}
    return {"task_type": "GPU", "devices": "0", "max_bin": 128}


def default_mask(values: np.ndarray, target: str) -> np.ndarray:
    if target == "SW":
        return values >= DEFAULT_VALUES[target]
    return values <= DEFAULT_VALUES[target]


def build_training_data(
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], FeatureBuilder, list[str]]:
    records = load_split(data_dir, "train")
    builder = FeatureBuilder().fit([frame for _, frame in records])
    base_chunks: list[np.ndarray] = []
    enhanced_chunks: list[np.ndarray] = []
    target_chunks: dict[str, list[np.ndarray]] = {target: [] for target in TARGET_COLUMNS}

    for _, frame in records:
        valid = frame[TARGET_COLUMNS].notna().all(axis=1).to_numpy()
        if not valid.any():
            continue
        matrices = builder.transform(frame)
        base_chunks.append(matrices.base[valid])
        enhanced_chunks.append(matrices.enhanced[valid])
        for target in TARGET_COLUMNS:
            target_chunks[target].append(frame.loc[valid, target].to_numpy(np.float64))

    base = np.concatenate(base_chunks)
    enhanced = np.concatenate(enhanced_chunks)
    targets = {
        target: np.concatenate(chunks)
        for target, chunks in target_chunks.items()
    }
    return base, enhanced, targets, builder, [well_id for well_id, _ in records]


def train_binary_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    iterations: int,
    seed: int,
    cpu: bool,
    verbose: int,
) -> CatBoostClassifier:
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.04,
        depth=8,
        loss_function="Logloss",
        random_seed=seed,
        bootstrap_type="Bayesian",
        allow_writing_files=False,
        verbose=verbose,
        **model_runtime(cpu),
    )
    model.fit(Pool(features, label=labels.astype(np.int8)))
    return model


def train_regressor(
    features: np.ndarray,
    labels: np.ndarray,
    iterations: int,
    loss_function: str,
    seed: int,
    cpu: bool,
    verbose: int,
) -> CatBoostRegressor:
    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=0.04,
        depth=8,
        loss_function=loss_function,
        random_seed=seed,
        bootstrap_type="Bayesian",
        l2_leaf_reg=4.0,
        allow_writing_files=False,
        verbose=verbose,
        **model_runtime(cpu),
    )
    model.fit(Pool(features, label=labels))
    return model


def make_por_bins(values: np.ndarray, bins: int = 80) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transformed = np.log1p(values)
    non_default = values > DEFAULT_VALUES["POR"]
    edges = np.unique(
        np.quantile(transformed[non_default], np.linspace(0.0, 1.0, bins + 1))
    )
    class_count = len(edges) - 1
    if class_count < 2:
        raise ValueError("POR needs at least two distinct non-default bins")
    labels = np.clip(
        np.searchsorted(edges, transformed, side="right") - 1,
        0,
        class_count - 1,
    ).astype(np.int32)
    centers = np.empty(class_count, dtype=np.float64)
    for class_index in range(class_count):
        selected = transformed[non_default & (labels == class_index)]
        centers[class_index] = (float(np.median(selected)) if len(selected) else
                                float((edges[class_index] + edges[class_index + 1]) / 2))
    return non_default, labels, centers


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct")
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new empty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    base, enhanced, targets, builder, well_ids = build_training_data(args.data_dir)
    print(f"Base features: {base.shape}")
    print(f"Enhanced features: {enhanced.shape}")

    por_non_default, por_classes, por_centers = make_por_bins(targets["POR"], bins=80)
    model_files: dict[str, list[str]] = {
        "POR_default": [],
        "POR_ordinal": [],
        "PERM_default": [],
        "PERM_regressor": [],
        "SW_default": [],
        "SW_regressor": [],
    }

    for ensemble_index, seed in enumerate(seeds):
        print(f"\n=== Ensemble member {ensemble_index + 1}/{len(seeds)} (seed={seed}) ===")

        por_default = train_binary_classifier(
            base,
            default_mask(targets["POR"], "POR"),
            iterations=500,
            seed=seed,
            cpu=args.cpu,
            verbose=args.verbose,
        )
        path = model_dir / f"POR_default_seed{seed}.cbm"
        por_default.save_model(path)
        model_files["POR_default"].append(path.name)

        por_ordinal = CatBoostClassifier(
            iterations=1200,
            learning_rate=0.04,
            depth=8,
            loss_function="MultiClass",
            random_seed=seed + 100,
            bootstrap_type="Bayesian",
            allow_writing_files=False,
            verbose=args.verbose,
            **model_runtime(args.cpu),
        )
        por_ordinal.fit(
            Pool(base[por_non_default], label=por_classes[por_non_default])
        )
        path = model_dir / f"POR_ordinal_seed{seed}.cbm"
        por_ordinal.save_model(path)
        model_files["POR_ordinal"].append(path.name)

        perm_default_mask = default_mask(targets["PERM"], "PERM")
        perm_default = train_binary_classifier(
            base,
            perm_default_mask,
            iterations=800,
            seed=seed + 200,
            cpu=args.cpu,
            verbose=args.verbose,
        )
        path = model_dir / f"PERM_default_seed{seed}.cbm"
        perm_default.save_model(path)
        model_files["PERM_default"].append(path.name)

        perm_regressor = train_regressor(
            base[~perm_default_mask],
            np.log10(targets["PERM"][~perm_default_mask]),
            iterations=370,
            loss_function="MAE",
            seed=seed + 300,
            cpu=args.cpu,
            verbose=args.verbose,
        )
        path = model_dir / f"PERM_regressor_seed{seed}.cbm"
        perm_regressor.save_model(path)
        model_files["PERM_regressor"].append(path.name)

        sw_default_mask = default_mask(targets["SW"], "SW")
        sw_default = train_binary_classifier(
            enhanced,
            sw_default_mask,
            iterations=950,
            seed=seed + 400,
            cpu=args.cpu,
            verbose=args.verbose,
        )
        path = model_dir / f"SW_default_seed{seed}.cbm"
        sw_default.save_model(path)
        model_files["SW_default"].append(path.name)

        sw_regressor = train_regressor(
            enhanced[~sw_default_mask],
            targets["SW"][~sw_default_mask],
            iterations=500,
            loss_function="RMSE",
            seed=seed + 500,
            cpu=args.cpu,
            verbose=args.verbose,
        )
        path = model_dir / f"SW_regressor_seed{seed}.cbm"
        sw_regressor.save_model(path)
        model_files["SW_regressor"].append(path.name)

    metadata = {
        "model_name": "target_specialized_catboost_v3_candidate",
        "version": "3.0",
        "seeds": seeds,
        "elapsed_seconds": time.perf_counter() - started,
        "source_sha256": {str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(Path(__file__).parent.rglob("*.py"))},
        "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted((args.data_dir / "train").glob("*.txt"))},
        "training_rows": int(len(base)),
        "training_wells": len(well_ids),
        "base_feature_names": builder.base_feature_names_,
        "enhanced_feature_names": builder.enhanced_feature_names_,
        "model_files": model_files,
        "por_ordinal": {
            "bins": int(len(por_centers)),
            "centers_log1p": por_centers.tolist(),
        },
        "default_values": DEFAULT_VALUES,
        "target_bounds": TARGET_BOUNDS,
        "calibration": CALIBRATION,
        "preprocessing": {"medians": builder.medians_},
        "por_decision": args.por_decision,
        "validation": {"status": "not_evaluated_this_run",
                       "calibration_source": {"POR": "fixed_gate_0.5_no_OOF_search",
                                              "PERM_SW": "historical_v2_OOF_not_nested"}},
    }
    metadata_path = args.output_dir / "model_meta.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved models to {model_dir}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
