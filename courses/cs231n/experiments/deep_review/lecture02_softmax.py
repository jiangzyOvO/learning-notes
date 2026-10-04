"""Stable batch Softmax and mean cross-entropy, with teaching examples.

Run: python3 experiments/deep_review/lecture02_softmax.py
Requires NumPy. Scores are specified, not trained.
"""
import numpy as np


def softmax_cross_entropy(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("Expected nonempty scores (N,C).")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Scores must be finite.")
    if labels.shape != (len(scores),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Expected integer class IDs (N,).")
    if np.any(labels < 0) or np.any(labels >= scores.shape[1]):
        raise ValueError("Class IDs must lie in [0,C).")
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    sum_exp = exp_scores.sum(axis=1, keepdims=True)
    probabilities = exp_scores / sum_exp
    losses = -shifted[np.arange(len(scores)), labels] + np.log(sum_exp[:, 0])
    return probabilities, losses, losses.mean()


def main():
    S = np.array([[2., 0., 0.], [0., 4., -1.], [1., -3., 2.]])
    y = np.array([0, 1, 2])
    p, losses, mean_loss = softmax_cross_entropy(S, y)
    np.testing.assert_allclose(p.sum(axis=1), 1.)
    np.testing.assert_allclose(losses, -np.log(p[np.arange(3), y]))
    np.testing.assert_array_equal(p.argmax(axis=1), S.argmax(axis=1))
    print("Probabilities:\n", p)
    print("Per-example losses:", losses)
    print("Mean loss:", mean_loss)

    shifted_p, shifted_losses, _ = softmax_cross_entropy(
        S + np.array([[1000.], [-300.], [20.]]), y
    )
    np.testing.assert_allclose(shifted_p, p)
    np.testing.assert_allclose(shifted_losses, losses)
    print("Common shift within each row leaves probabilities and losses unchanged.")

    _, wrong_label_loss, _ = softmax_cross_entropy(S[:1], np.array([1]))
    print("First example, true label B instead of A:", wrong_label_loss[0])
    _, uniform_losses, _ = softmax_cross_entropy(np.zeros((1, 3)), np.array([0]))
    np.testing.assert_allclose(uniform_losses, np.log(3.))
    print("Three equal scores give loss log(3):", uniform_losses[0])

    # The probability of class 1 underflows; direct log-sum-exp loss stays finite.
    extreme_p, extreme_losses, _ = softmax_cross_entropy(
        np.array([[0., -1000.]]), np.array([1])
    )
    np.testing.assert_allclose(extreme_losses, [1000.])
    print("Extreme probabilities:", extreme_p)
    print("Stable extreme loss:", extreme_losses[0])
    print("All examples passed; no model training or benchmark evaluation.")


if __name__ == "__main__":
    main()
