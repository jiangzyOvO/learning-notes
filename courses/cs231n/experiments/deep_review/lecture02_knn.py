"""CS231n lecture 2 supplement: distances, voting and cross-validation.

Run: python3 experiments/deep_review/lecture02_knn.py
Requires NumPy. The constructed points are not image benchmark results.
"""
import numpy as np


class KNearestNeighbor:
    def train(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(X) == 0:
            raise ValueError("Expected nonempty X (N,D) and aligned y (N,).")
        if not np.issubdtype(y.dtype, np.integer) or np.any(y < 0):
            raise ValueError("This teaching implementation uses nonnegative integer labels.")
        self.X_train, self.y_train = X, y

    def _query(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.X_train.shape[1]:
            raise ValueError("Queries must have the same feature dimension as training data.")
        return X

    def compute_distances_two_loops(self, X):
        X = self._query(X)
        dists = np.zeros((len(X), len(self.X_train)))
        for i in range(len(X)):
            for j in range(len(self.X_train)):
                difference = X[i] - self.X_train[j]
                dists[i, j] = np.sqrt(np.sum(difference * difference))
        return dists

    def compute_distances_no_loops(self, X):
        X = self._query(X)
        test_sq = np.sum(X * X, axis=1, keepdims=True)  # (num_test,1)
        train_sq = np.sum(self.X_train * self.X_train, axis=1)  # (num_train,)
        cross = X @ self.X_train.T  # (num_test,num_train)
        squared = test_sq + train_sq - 2 * cross
        return np.sqrt(np.maximum(squared, 0.0))

    def predict_labels(self, dists, k=1):
        if not isinstance(k, (int, np.integer)) or not 1 <= k <= len(self.X_train):
            raise ValueError("k must be an integer between 1 and the training set size.")
        predictions = np.empty(len(dists), dtype=self.y_train.dtype)
        for i in range(len(dists)):
            # At equal distance, preserve training-array order.
            nearest_indices = np.argsort(dists[i], kind="stable")[:k]
            nearest_labels = self.y_train[nearest_indices]
            # unique sorts label IDs; argmax returns the first maximum.
            labels, counts = np.unique(nearest_labels, return_counts=True)
            predictions[i] = labels[np.argmax(counts)]
        return predictions

    def predict(self, X, k=1):
        return self.predict_labels(self.compute_distances_no_loops(X), k)


def cross_validate(X, y, k_choices, num_folds=5, seed=231):
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("Expected aligned X (N,D) and y (N,).")
    if not 2 <= num_folds <= len(X) or not k_choices:
        raise ValueError("Need at least two nonempty folds and a nonempty k list.")
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(X))
    folds = np.array_split(shuffled_indices, num_folds)
    k_to_accuracies = {k: [] for k in k_choices}
    for validation_fold in range(num_folds):
        validation_indices = folds[validation_fold]
        training_indices = np.concatenate([
            fold for i, fold in enumerate(folds) if i != validation_fold
        ])
        model = KNearestNeighbor()
        model.train(X[training_indices], y[training_indices])
        # For a fixed fold, all candidate k values share the same distance table.
        dists = model.compute_distances_no_loops(X[validation_indices])
        for k in k_choices:
            prediction = model.predict_labels(dists, k)
            accuracy = np.mean(prediction == y[validation_indices])
            k_to_accuracies[k].append(float(accuracy))
    return k_to_accuracies


def main():
    X_train = np.array([[1., 0.], [0., 2.], [2., 2.], [4., 0.], [-3., 0.]])
    y_train = np.array([0, 1, 1, 0, 1])  # A=0, B=1
    X_test = np.array([[0., 0.], [4., 1.]])
    model = KNearestNeighbor()
    model.train(X_train, y_train)
    direct = model.compute_distances_two_loops(X_test)
    vectorized = model.compute_distances_no_loops(X_test)
    np.testing.assert_allclose(vectorized, direct, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(vectorized**2, [[1, 4, 8, 16, 9], [10, 17, 5, 1, 50]])
    assert model.predict(X_test, k=1).tolist() == [0, 0]
    assert model.predict(X_test, k=3).tolist() == [1, 0]
    print("Squared distance matrix:")
    print(vectorized**2)
    print("Nearest indices for query 0:", np.argsort(vectorized[0], kind="stable").tolist())
    print("k=1 predictions:", model.predict(X_test, k=1).tolist())
    print("k=3 predictions:", model.predict(X_test, k=3).tolist())

    # Verify the documented voting tie rule independently from distance ties.
    model.train([[0., 0.], [1., 0.]], [1, 0])
    assert model.predict([[.5, 0.]], k=2).tolist() == [0]
    # Casting before subtraction prevents uint8 wraparound.
    model.train(np.array([[255]], dtype=np.uint8), [1])
    assert model.compute_distances_no_loops(np.array([[0]], dtype=np.uint8))[0, 0] == 255

    rng = np.random.default_rng(231)
    X = np.concatenate([rng.normal([0., 0.], .8, size=(20, 2)),
                        rng.normal([1., .5], .8, size=(20, 2))])
    y = np.repeat([0, 1], 20)
    results = cross_validate(X, y, k_choices=[1, 3, 5], num_folds=5)
    print("Constructed 40-point data, 5-fold validation:")
    for k, accuracies in results.items():
        assert len(accuracies) == 5
        print(f"k={k}: folds={accuracies}, mean={np.mean(accuracies):.3f}")
    best_k = max(results, key=lambda k: (np.mean(results[k]), -k))
    print("Selected k among the specified candidates:", best_k)
    print("No independent test-set score is claimed by this demonstration.")
    print("All small-distance, vote and representation checks passed.")


if __name__ == "__main__":
    main()
