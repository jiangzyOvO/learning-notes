"""Affine backprop and a two-layer ReLU/Softmax classifier gradient check.

Run: python3 experiments/deep_review/lecture04_matrix_backprop.py
Requires NumPy. Constructed samples, not a benchmark training result.
"""
import numpy as np


def loss_and_grad(params, X, labels, reg=0.1):
    W1, b1, W2, b2 = (params[k] for k in ("W1", "b1", "W2", "b2"))
    a = X @ W1 + b1
    h = np.maximum(a, 0.)
    scores = h @ W2 + b2
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    sum_exp = exp_scores.sum(axis=1, keepdims=True)
    p = exp_scores / sum_exp
    N = len(X)
    data_loss = np.mean(-shifted[np.arange(N), labels] + np.log(sum_exp[:, 0]))
    penalty = 0.5 * reg * (np.sum(W1*W1) + np.sum(W2*W2))
    ds = p.copy()
    ds[np.arange(N), labels] -= 1
    ds /= N
    grad_W2 = h.T @ ds + reg * W2
    grad_b2 = ds.sum(axis=0)
    dh = ds @ W2.T
    da = dh * (a > 0.)
    grad_W1 = X.T @ da + reg * W1
    grad_b1 = da.sum(axis=0)
    grad_X = da @ W1.T
    grads = {"W1": grad_W1, "b1": grad_b1, "W2": grad_W2, "b2": grad_b2}
    return data_loss + penalty, grads, grad_X, {
        "a": a, "h": h, "scores": scores, "p": p,
        "data_loss": data_loss, "reg_loss": penalty,
    }


def main():
    X_affine = np.array([[1., 2.], [3., 4.]])
    W_affine = np.array([[1., 2., 3.], [4., 5., 6.]])
    b_affine = np.array([0., 0.5, -0.5])
    G = np.array([[1., 2., 3.], [4., 5., 6.]])
    Z = X_affine @ W_affine + b_affine
    grad_W = X_affine.T @ G
    grad_X = G @ W_affine.T
    grad_b = G.sum(axis=0)
    np.testing.assert_allclose(Z, [[9., 12.5, 14.5], [19., 26.5, 32.5]])
    np.testing.assert_allclose(grad_W, [[13., 17., 21.], [18., 24., 30.]])
    np.testing.assert_allclose(grad_X, [[14., 32.], [32., 77.]])
    np.testing.assert_allclose(grad_b, [5., 7., 9.])
    # Independent elementwise accumulation, without the gradient matmul.
    direct = np.zeros_like(W_affine)
    for d in range(2):
        for m in range(3):
            for i in range(2):
                direct[d, m] += X_affine[i, d] * G[i, m]
    np.testing.assert_allclose(grad_W, direct)
    # G is realized by the scalar objective sum(G*Z), not guessed from Z.
    delta = 1e-5
    affine_error = 0.
    for parameter, analytical in ((X_affine, grad_X), (W_affine, grad_W), (b_affine, grad_b)):
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + delta
            plus = np.sum(G * (X_affine @ W_affine + b_affine))
            parameter[index] = original - delta
            minus = np.sum(G * (X_affine @ W_affine + b_affine))
            parameter[index] = original
            affine_error = max(affine_error, abs((plus-minus)/(2*delta)-analytical[index]))
    assert affine_error < 1e-7
    print("Affine forward:\n", Z)
    print("Affine grad_W:\n", grad_W)
    print("Affine grad_X:\n", grad_X)
    print("Affine grad_b:", grad_b, "max numerical error:", affine_error)

    X = np.array([[1., 2.], [-1., 1.]])
    labels = np.array([0, 1])
    params = {
        "W1": np.array([[1., -1.], [0.5, 0.5]]),
        "b1": np.array([0.1, -0.2]),
        "W2": np.array([[0.2, -0.3], [0.4, 0.1]]),
        "b2": np.array([0., 0.05]),
    }
    loss, grads, input_grad, cache = loss_and_grad(params, X, labels)
    assert np.min(np.abs(cache["a"])) > 0.1  # Avoid ReLU kinks in checks.
    max_error = 0.
    variables = [(params[k], grads[k]) for k in params] + [(X, input_grad)]
    for parameter, analytical in variables:
        numerical = np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + delta
            plus = loss_and_grad(params, X, labels)[0]
            parameter[index] = original - delta
            minus = loss_and_grad(params, X, labels)[0]
            parameter[index] = original
            numerical[index] = (plus-minus)/(2*delta)
        np.testing.assert_allclose(numerical, analytical, atol=1e-9, rtol=1e-7)
        max_error = max(max_error, float(np.max(np.abs(numerical-analytical))))
    updated = {k: params[k] - 0.1*grads[k] for k in params}
    new_loss, _, _, new_cache = loss_and_grad(updated, X, labels)
    assert new_loss < loss
    print("Two-layer preactivation a:\n", cache["a"])
    print("Hidden h:\n", cache["h"])
    print("Scores:\n", cache["scores"])
    print("Probabilities:\n", cache["p"])
    print("Data/regularization/total:", cache["data_loss"], cache["reg_loss"], loss)
    for k, gradient in grads.items():
        print("Gradient", k, ":", gradient)
    print("Two-layer max numerical error:", max_error)
    print("Total before/after one update:", loss, new_loss)
    print("Data loss after:", new_cache["data_loss"])
    print("No validation or test set; no generalization conclusion.")


if __name__ == "__main__":
    main()
