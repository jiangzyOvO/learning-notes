"""The screenshot's two-layer sigmoid/SSE network, plus a small gradient check.

Run: python3 experiments/sigmoid_network.py
Requires NumPy. All data are synthetic; this is not a classification benchmark.
"""
import numpy as np


def sigmoid(a):
    """Stable, elementwise sigmoid without evaluating an overflowing branch."""
    h = np.empty_like(a)
    positive = a >= 0
    h[positive] = 1 / (1 + np.exp(-a[positive]))
    exp_a = np.exp(a[~positive])
    h[~positive] = exp_a / (1 + exp_a)
    return h


def loss_and_grads(x, y, w1, w2):
    h = sigmoid(x @ w1)
    y_pred = h @ w2
    error = y_pred - y
    loss = np.square(error).sum()
    grad_y_pred = 2 * error
    grad_w2 = h.T @ grad_y_pred
    grad_h = grad_y_pred @ w2.T
    grad_a = grad_h * h * (1 - h)
    grad_w1 = x.T @ grad_a
    return float(loss), grad_w1, grad_w2


def check_gradients():
    """Compare chain-rule derivatives to independent central differences."""
    rng = np.random.default_rng(231)
    x = rng.standard_normal((3, 2))
    y = rng.standard_normal((3, 2))
    w1 = rng.standard_normal((2, 3)) * .2
    w2 = rng.standard_normal((3, 2)) * .2
    _, g1, g2 = loss_and_grads(x, y, w1, w2)
    eps = 1e-5
    max_error = 0.0
    for weight, analytic in ((w1, g1), (w2, g2)):
        numeric = np.zeros_like(weight)
        for index in np.ndindex(weight.shape):
            old = weight[index]
            weight[index] = old + eps
            plus = loss_and_grads(x, y, w1, w2)[0]
            weight[index] = old - eps
            minus = loss_and_grads(x, y, w1, w2)[0]
            weight[index] = old
            numeric[index] = (plus - minus) / (2 * eps)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)
        max_error = max(max_error, float(np.abs(analytic - numeric).max()))
    print(f'Gradient check passed: maximum absolute error = {max_error:.3e}')


def train_demo():
    rng = np.random.default_rng(231)
    n, d_in, hidden, d_out = 64, 1000, 100, 10
    x = rng.standard_normal((n, d_in))
    y = rng.standard_normal((n, d_out))
    w1 = rng.standard_normal((d_in, hidden))
    w2 = rng.standard_normal((hidden, d_out))
    initial = loss_and_grads(x, y, w1, w2)[0]
    learning_rate = 1e-4
    for step in range(2000):
        loss, g1, g2 = loss_and_grads(x, y, w1, w2)
        # Both gradients refer to the same old weights.
        w1 -= learning_rate * g1
        w2 -= learning_rate * g2
        if step % 200 == 0:
            print(f'step {step:4d}, loss before update = {loss:.6f}')
    final = loss_and_grads(x, y, w1, w2)[0]
    assert np.isfinite(final) and final < initial
    print(f'After 2000 updates: {initial:.6f} -> {final:.6g}')
    print('Synthetic continuous targets; no test-set accuracy is being reported.')


if __name__ == '__main__':
    check_gradients()
    train_demo()
