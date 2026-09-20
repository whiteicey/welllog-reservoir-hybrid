from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np


@dataclass
class TreeNode:
    value: float
    is_leaf: bool
    feature: int = -1
    split_bin: int = -1
    left: int = -1
    right: int = -1


class HistTree:
    """A compact histogram decision tree for classification or regression."""

    def __init__(
        self,
        task: str,
        max_depth: int,
        min_samples_leaf: int,
        max_features: int | None,
        bin_count: int,
        random_state: int,
    ) -> None:
        if task not in {"classification", "regression"}:
            raise ValueError("task must be classification or regression")
        self.task = task
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bin_count = bin_count
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.nodes: list[TreeNode] = []

    def fit(
        self,
        binned: np.ndarray,
        y: np.ndarray,
    ) -> "HistTree":
        if self.task == "classification":
            targets = np.asarray(y, dtype=np.int8)
            if not np.isin(targets, [0, 1]).all():
                raise ValueError("classification targets must be 0 or 1")
        else:
            targets = np.asarray(y, dtype=np.float32)
        self.nodes = []
        self._build(np.arange(len(targets), dtype=np.int64), binned, targets, 0)
        return self

    def _build(
        self,
        indices: np.ndarray,
        binned: np.ndarray,
        y: np.ndarray,
        depth: int,
    ) -> int:
        if self.task == "classification":
            value = float(np.mean(y[indices]))
        else:
            value = float(np.mean(y[indices]))
        node_index = len(self.nodes)
        self.nodes.append(TreeNode(value=value, is_leaf=True))

        if depth >= self.max_depth or len(indices) < 2 * self.min_samples_leaf:
            return node_index

        split = self._best_split(indices, binned, y)
        if split is None:
            return node_index

        feature, split_bin = split
        mask = binned[indices, feature] <= split_bin
        left_indices = indices[mask]
        right_indices = indices[~mask]
        if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
            return node_index

        self.nodes[node_index].is_leaf = False
        self.nodes[node_index].feature = int(feature)
        self.nodes[node_index].split_bin = int(split_bin)
        self.nodes[node_index].left = self._build(left_indices, binned, y, depth + 1)
        self.nodes[node_index].right = self._build(right_indices, binned, y, depth + 1)
        return node_index

    def _best_split(
        self,
        indices: np.ndarray,
        binned: np.ndarray,
        y: np.ndarray,
    ) -> tuple[int, int] | None:
        feature_count = binned.shape[1]
        if self.max_features is None or self.max_features >= feature_count:
            features = np.arange(feature_count)
        else:
            features = self.rng.choice(feature_count, size=self.max_features, replace=False)

        best_gain = 1e-7
        best_feature = -1
        best_bin = -1
        bins_count = self.bin_count

        if self.task == "classification":
            node_y = y[indices]
            positive = node_y == 1
            total = len(indices)
            total_positive = int(np.count_nonzero(positive))
            total_negative = total - total_positive
            if total_positive == 0 or total_negative == 0:
                return None
            parent_impurity = 1.0 - (
                (total_positive / total) ** 2 + (total_negative / total) ** 2
            )
            parent_score = total * parent_impurity
            for feature in features:
                bins = binned[indices, feature]
                hist_positive = np.bincount(bins[positive], minlength=bins_count)
                hist_negative = np.bincount(bins[~positive], minlength=bins_count)
                left_positive = np.cumsum(hist_positive)
                left_negative = np.cumsum(hist_negative)
                left_total = left_positive + left_negative
                right_total = total - left_total
                valid = (
                    (left_total >= self.min_samples_leaf)
                    & (right_total >= self.min_samples_leaf)
                    & (left_total > 0)
                    & (right_total > 0)
                )
                if not valid.any():
                    continue
                left_impurity = 1.0 - (
                    (left_positive / left_total) ** 2
                    + (left_negative / left_total) ** 2
                )
                right_positive = total_positive - left_positive
                right_negative = total_negative - left_negative
                right_impurity = 1.0 - (
                    (right_positive / right_total) ** 2
                    + (right_negative / right_total) ** 2
                )
                score = left_total * left_impurity + right_total * right_impurity
                gain = np.where(valid, parent_score - score, -np.inf)
                local_index = int(np.argmax(gain))
                if gain[local_index] > best_gain:
                    best_gain = float(gain[local_index])
                    best_feature = int(feature)
                    best_bin = local_index
        else:
            node_y = y[indices].astype(np.float64, copy=False)
            total = len(indices)
            total_sum = float(np.sum(node_y))
            total_square = float(np.sum(node_y * node_y))
            parent_sse = total_square - total_sum * total_sum / total
            if parent_sse <= 1e-7:
                return None
            for feature in features:
                bins = binned[indices, feature]
                hist_sum = np.bincount(bins, weights=node_y, minlength=bins_count)
                hist_square = np.bincount(
                    bins,
                    weights=node_y * node_y,
                    minlength=bins_count,
                )
                left_sum = np.cumsum(hist_sum)
                left_square = np.cumsum(hist_square)
                left_total = np.cumsum(np.bincount(bins, minlength=bins_count))
                right_total = total - left_total
                right_sum = total_sum - left_sum
                right_square = total_square - left_square
                valid = (
                    (left_total >= self.min_samples_leaf)
                    & (right_total >= self.min_samples_leaf)
                    & (left_total > 0)
                    & (right_total > 0)
                )
                if not valid.any():
                    continue
                left_sse = left_square - left_sum * left_sum / left_total
                right_sse = right_square - right_sum * right_sum / right_total
                gain = np.where(valid, parent_sse - left_sse - right_sse, -np.inf)
                local_index = int(np.argmax(gain))
                if gain[local_index] > best_gain:
                    best_gain = float(gain[local_index])
                    best_feature = int(feature)
                    best_bin = local_index

        if best_feature < 0:
            return None
        return best_feature, best_bin

    def predict(self, binned: np.ndarray) -> np.ndarray:
        result = np.empty(len(binned), dtype=np.float64)
        self._predict_node(np.arange(len(binned), dtype=np.int64), binned, 0, result)
        return result

    def _predict_node(
        self,
        indices: np.ndarray,
        binned: np.ndarray,
        node_index: int,
        output: np.ndarray,
    ) -> None:
        node = self.nodes[node_index]
        if node.is_leaf:
            output[indices] = node.value
            return
        mask = binned[indices, node.feature] <= node.split_bin
        self._predict_node(indices[mask], binned, node.left, output)
        self._predict_node(indices[~mask], binned, node.right, output)


class HistForest:
    """Random forest using shared histogram thresholds."""

    def __init__(
        self,
        task: str,
        n_trees: int = 3,
        max_depth: int = 10,
        min_samples_leaf: int = 100,
        max_features: int | None = 24,
        max_bins: int = 64,
        row_subsample: float = 0.8,
        random_state: int = 42,
    ) -> None:
        self.task = task
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_bins = max_bins
        self.row_subsample = row_subsample
        self.random_state = random_state
        self.thresholds_: list[np.ndarray] = []
        self.trees_: list[HistTree] = []

    def fit(self, features: np.ndarray, target: np.ndarray) -> "HistForest":
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(target)
        if len(x) != len(y):
            raise ValueError("features and target must have the same length")
        self.thresholds_ = []
        binned = np.empty(x.shape, dtype=np.int16)
        quantiles = np.linspace(0.0, 1.0, self.max_bins + 1)[1:-1]
        for feature in range(x.shape[1]):
            values = x[:, feature]
            thresholds = np.unique(np.quantile(values, quantiles))
            self.thresholds_.append(thresholds.astype(np.float32))
            binned[:, feature] = np.searchsorted(thresholds, values, side="right")

        rng = np.random.default_rng(self.random_state)
        sample_size = max(self.min_samples_leaf * 2, int(len(x) * self.row_subsample))
        sample_size = min(sample_size, len(x))
        self.trees_ = []
        for tree_index in range(self.n_trees):
            sample_indices = rng.choice(len(x), size=sample_size, replace=False)
            tree = HistTree(
                task=self.task,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                bin_count=self.max_bins,
                random_state=self.random_state + tree_index + 1,
            )
            tree.fit(binned[sample_indices], y[sample_indices])
            self.trees_.append(tree)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self.trees_:
            raise RuntimeError("forest must be fitted before predict")
        x = np.asarray(features, dtype=np.float32)
        binned = np.empty(x.shape, dtype=np.int16)
        for feature, thresholds in enumerate(self.thresholds_):
            binned[:, feature] = np.searchsorted(thresholds, x[:, feature], side="right")
        predictions = np.stack([tree.predict(binned) for tree in self.trees_], axis=0)
        return np.mean(predictions, axis=0)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "HistForest":
        with path.open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError("Model file does not contain a HistForest")
        return model
