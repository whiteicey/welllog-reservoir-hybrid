from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


TARGET_BOUNDS = {
    "POR": (0.0, 50.0),
    "PERM": (0.01, 10000.0),
    "SW": (8.305, 99.9),
}


@dataclass
class RidgeModel:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    coefficients: np.ndarray
    alpha: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    target_transforms: list[str]

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        x = (x - self.feature_mean) / self.feature_std
        x = np.column_stack([np.ones(len(x), dtype=np.float64), x])
        y_std = x @ self.coefficients
        y = y_std * self.target_std + self.target_mean
        return self._inverse_transform(y)

    def _inverse_transform(self, values: np.ndarray) -> np.ndarray:
        result = np.empty_like(values, dtype=np.float64)
        for index, transform in enumerate(self.target_transforms):
            column = values[:, index]
            if transform == "identity":
                result[:, index] = column
            elif transform == "log1p":
                result[:, index] = np.expm1(column)
            elif transform == "log10":
                result[:, index] = np.power(10.0, column)
            elif transform == "logit":
                result[:, index] = 100.0 / (1.0 + np.exp(-column))
            else:
                raise ValueError(f"Unknown target transform: {transform}")
            low, high = TARGET_BOUNDS[self.target_names[index]]
            result[:, index] = np.clip(result[:, index], low, high)
        return result

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
            target_mean=self.target_mean,
            target_std=self.target_std,
            coefficients=self.coefficients,
            alpha=self.alpha,
            feature_names=np.array(self.feature_names),
            target_names=np.array(self.target_names),
            target_transforms=np.array(self.target_transforms),
        )

    @classmethod
    def load(cls, path: Path) -> "RidgeModel":
        with np.load(path, allow_pickle=False) as archive:
            return cls(
                feature_mean=archive["feature_mean"],
                feature_std=archive["feature_std"],
                target_mean=archive["target_mean"],
                target_std=archive["target_std"],
                coefficients=archive["coefficients"],
                alpha=archive["alpha"],
                feature_names=archive["feature_names"].tolist(),
                target_names=archive["target_names"].tolist(),
                target_transforms=archive["target_transforms"].tolist(),
            )


def transform_targets(
    y: np.ndarray,
    transforms: list[str],
) -> np.ndarray:
    result = np.empty_like(y, dtype=np.float64)
    for index, transform in enumerate(transforms):
        column = y[:, index]
        if transform == "identity":
            result[:, index] = column
        elif transform == "log1p":
            result[:, index] = np.log1p(column)
        elif transform == "log10":
            result[:, index] = np.log10(column)
        elif transform == "logit":
            result[:, index] = np.log(np.clip(column, 1e-6, 100.0 - 1e-6) / (100.0 - np.clip(column, 1e-6, 100.0 - 1e-6)))
        else:
            raise ValueError(f"Unknown target transform: {transform}")
    return result


def fit_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: list[str],
    target_names: list[str],
    target_transforms: list[str],
    alphas: list[float],
) -> list[tuple[float, RidgeModel]]:
    """Fit one ridge candidate per alpha using a joint normal-equation system."""
    x = np.asarray(features, dtype=np.float32)
    y = transform_targets(np.asarray(targets, dtype=np.float64), target_transforms)

    feature_mean = x.mean(axis=0)
    feature_std = x.std(axis=0)
    feature_std[feature_std < 1e-8] = 1.0
    target_mean = y.mean(axis=0)
    target_std = y.std(axis=0)
    target_std[target_std < 1e-8] = 1.0

    feature_mean32 = feature_mean.astype(np.float32)
    feature_std32 = feature_std.astype(np.float32)
    y = (y - target_mean) / target_std
    gram = np.zeros((x.shape[1] + 1, x.shape[1] + 1), dtype=np.float64)
    moment = np.zeros((x.shape[1] + 1, y.shape[1]), dtype=np.float64)
    chunk_size = 65536
    for start in range(0, len(x), chunk_size):
        stop = min(start + chunk_size, len(x))
        x_chunk = (x[start:stop] - feature_mean32) / feature_std32
        design = np.column_stack([np.ones(stop - start, dtype=np.float32), x_chunk])
        gram += design.T.astype(np.float64) @ design.astype(np.float64)
        moment += design.T.astype(np.float64) @ y[start:stop]

    models: list[tuple[float, RidgeModel]] = []
    for alpha in alphas:
        penalty = np.eye(gram.shape[0], dtype=np.float64) * float(alpha)
        penalty[0, 0] = 0.0
        try:
            coefficients = np.linalg.solve(gram + penalty, moment)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(gram + penalty, moment, rcond=None)[0]
        models.append((
            float(alpha),
            RidgeModel(
                feature_mean=feature_mean,
                feature_std=feature_std,
                target_mean=target_mean,
                target_std=target_std,
                coefficients=coefficients,
                alpha=np.array([float(alpha)] * len(target_names)),
                feature_names=feature_names,
                target_names=target_names,
                target_transforms=target_transforms,
            ),
        ))
    return models
