"""Residual gradients and a frozen constructed feature extractor.

Run: python3 experiments/deep_review/lecture06_residual_transfer.py
Requires NumPy. The feature extractor is specified, not actually pretrained.
"""
import numpy as np
from lecture04_matrix_backprop import loss_and_grad


def residual_forward_backward(X, W, b, upstream):
    a = X @ W + b
    branch = np.maximum(a, 0.)
    Y = X + branch
    da = upstream * (a > 0)
    dX = upstream + da @ W.T
    dW = X.T @ da
    db = da.sum(axis=0)
    return Y, dX, dW, db


def main():
    X = np.array([[1., 2.], [-1., 1.]])
    W = np.array([[0.2, -0.1], [0.3, 0.4]])
    b = np.array([0.1, -0.55])
    G = np.array([[1., 2.], [3., 4.]])
    Y, dx, dw, db = residual_forward_backward(X, W, b, G)
    max_error, delta = 0., 1e-5
    for parameter, analytical in ((X, dx), (W, dw), (b, db)):
        numerical = np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            old = parameter[index]
            parameter[index] = old+delta
            plus = np.sum(residual_forward_backward(X, W, b, G)[0]*G)
            parameter[index] = old-delta
            minus = np.sum(residual_forward_backward(X, W, b, G)[0]*G)
            parameter[index] = old
            numerical[index] = (plus-minus)/(2*delta)
        np.testing.assert_allclose(numerical, analytical, atol=1e-8, rtol=1e-6)
        max_error = max(max_error, float(np.max(np.abs(numerical-analytical))))
    np.testing.assert_allclose(Y, [[1.9, 2.15], [-0.8, 1.]])
    zeroY, zeroDX, _, _ = residual_forward_backward(X, np.zeros_like(W), np.zeros_like(b), G)
    np.testing.assert_array_equal(zeroY, X)
    np.testing.assert_array_equal(zeroDX, G)
    print("Residual output:\n", Y)
    print("Residual input gradient (shortcut + branch):\n", dx)
    print("Residual numerical-gradient max error:", max_error)
    print("Zero residual branch preserves both input and upstream gradient in this example.")

    X_toy = np.array([[-2., 0.], [-1., 1.], [-1., -1.], [2., 0.], [1., 1.], [1., -1.]])
    labels = np.array([0, 0, 0, 1, 1, 1])
    rng = np.random.default_rng(231)
    params = {
        "W1": np.array([[1., -1., 0., 0.], [0., 0., 1., -1.]]),
        "b1": np.zeros(4),
        "W2": rng.normal(0., 0.01, size=(4, 2)),
        "b2": np.zeros(2),
    }
    before_backbone = params["W1"].copy(), params["b1"].copy()
    before_head = params["W2"].copy()
    initial = loss_and_grad(params, X_toy, labels, reg=0.)[0]
    for step in range(200):
        _, grads, _, _ = loss_and_grad(params, X_toy, labels, reg=0.)
        # Only the head is updated. Computing unused backbone gradients is
        # intentionally retained to reuse the verified teaching implementation.
        for name in ("W2", "b2"):
            params[name] -= 0.1 * grads[name]
    final, _, _, cache = loss_and_grad(params, X_toy, labels, reg=0.)
    np.testing.assert_array_equal(params["W1"], before_backbone[0])
    np.testing.assert_array_equal(params["b1"], before_backbone[1])
    assert not np.array_equal(params["W2"], before_head) and final < initial
    prediction = cache["scores"].argmax(axis=1)
    np.testing.assert_array_equal(prediction, labels)
    print("Head-only training loss before/after:", initial, final)
    print("Backbone unchanged; training predictions:", prediction)
    # One fine-tuning step uses separate learning rates; new parameters are
    # computed from the same old gradients.
    _, grads, _, _ = loss_and_grad(params, X_toy, labels, reg=0.)
    tuned = {name: value - (0.001 if name in ("W1", "b1") else 0.01)*grads[name]
             for name, value in params.items()}
    assert np.max(np.abs(tuned["W1"]-params["W1"])) > 0.
    print("Fine-tuning max backbone change:", np.max(np.abs(tuned["W1"]-params["W1"])))
    print("No pretrained checkpoint, validation set or transfer-performance claim.")


if __name__ == "__main__":
    main()
