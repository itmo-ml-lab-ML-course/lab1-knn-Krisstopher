import numpy as np
import numpy.linalg as linalg

from Kernel import Kernel
from Metric import Metric


class KNearestNeighbors:
    def __init__(
            self, neighbors_count: int, metric: Metric, kernel: Kernel = None,
            window_width: float = None, minkowski_p: int = None
    ):
        self.metric = metric
        self.neighbors_count = neighbors_count
        self.kernel = kernel
        self.window_width = window_width
        self.minkowski_p = minkowski_p

        self.class_weights = None
        self.object_weights = None
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train, class_weights=None, object_weights=None):
        self.x_train = x_train
        self.y_train = y_train
        if class_weights:
            self.class_weights = class_weights
        else:
            self.class_weights = dict.fromkeys(np.unique(y_train), 1.0)

        if object_weights:
            self.object_weights = object_weights
        else:
            self.object_weights = np.ones(len(x_train))

    def predict(self, x_test):
        predictions = []

        for point_idx in range(len(x_test)):
            weighted_x_i = x_test[point_idx] * self.object_weights[point_idx]
            scores = self.compute_scores(weighted_x_i)
            predictions.append(max(scores, key=scores.get))

        return predictions

    def compute_scores(self, x_i):
        distances = self.compute_distances(x_i)
        normalized_distances = self.apply_window(distances)
        nearest_indexes = np.argsort(normalized_distances)[:self.neighbors_count]

        distances_with_kernel = self.apply_kernel(normalized_distances)
        scores = dict.fromkeys(np.unique(self.y_train), 0.0)

        for neighbor_idx in nearest_indexes:
            label = self.y_train[neighbor_idx]
            scores[label] += self.class_weights[label] * distances_with_kernel[neighbor_idx]

        return scores


    def compute_distances(self, x_i):
        if self.metric == Metric.EUCLIDEAN:
            return np.sqrt(np.sum((self.x_train - x_i) ** 2, axis=1))
        if self.metric == Metric.COSINE:
            return 1 - (np.sum(self.x_train * x_i, axis=1)) \
                   / (linalg.norm(self.x_train, axis=1) * linalg.norm(x_i))
        if self.metric == Metric.MINKOWSKI:
            return np.sum(np.abs(self.x_train - x_i) ** self.minkowski_p, axis=1) ** (1 / self.minkowski_p)

    def get_neighbors(self, x_i):
        distances = self.compute_distances(x_i)
        normalized_distances = self.apply_window(distances)

        return np.argsort(normalized_distances)[:self.neighbors_count]

    def apply_window(self, distances):
        if self.window_width is None:  # Dynamic window
            distance_for_k_plus_1 = np.sort(distances)[self.neighbors_count]
            return distances / distance_for_k_plus_1
        else:
            return distances / self.window_width

    def apply_kernel(self, distances):
        if self.kernel is None:
            return distances
        return np.vectorize(self.kernel)(distances)
