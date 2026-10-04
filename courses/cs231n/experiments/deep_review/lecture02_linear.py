"""Lecture 2 teaching examples: linear scores, batch shapes and boundaries.

Run: python3 experiments/deep_review/lecture02_linear.py
Requires NumPy. Parameters are specified by hand, not trained.
"""
import numpy as np


def main():
    # C rows of class weights, D columns of input features.
    W = np.array([[1., 0.], [-1., 2.], [0., -1.]])
    b = np.array([0., 0., 1.])
    x = np.array([2., 1.])
    scores = W @ x + b
    np.testing.assert_allclose(scores, [2., 0., 0.])
    print("Single example scores:", scores)
    print("Predicted class ID:", np.argmax(scores))

    X = np.array([[2., 1.], [0., 2.], [1., -1.]])
    S = X @ W.T + b
    expected = np.array([[2., 0., 0.], [0., 4., -1.], [1., -3., 2.]])
    np.testing.assert_allclose(S, expected)
    # Independently expand each class score, rather than repeat matmul.
    direct = np.array([
        [sum(W[c, j] * row[j] for j in range(2)) + b[c]
         for c in range(3)] for row in X
    ])
    np.testing.assert_allclose(S, direct)
    np.testing.assert_array_equal(np.argmax(S, axis=1), [0, 1, 2])
    print("Batch scores:\n", S)
    print("Batch predictions:", np.argmax(S, axis=1))

    changed_b = b.copy()
    changed_b[1] = 3.
    np.testing.assert_allclose(W @ x + changed_b, [2., 3., 0.])
    print("Scores after changing B bias:", W @ x + changed_b)

    # A/B scores tie when 2*x1 - 2*x2 = 0 for the original parameters.
    for point in (np.array([1., 1.]), np.array([2., 2.])):
        point_scores = W @ point + b
        np.testing.assert_allclose(point_scores[0], point_scores[1])
    print("A/B score-equality examples verified.")
    print("All examples passed; this script performs forward calculations only.")


if __name__ == "__main__":
    main()
