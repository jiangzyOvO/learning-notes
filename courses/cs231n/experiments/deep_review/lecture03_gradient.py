"""Scalar gradient descent and one-example Softmax classifier update.

Run: python3 experiments/deep_review/lecture03_gradient.py
Requires NumPy. Teaching calculations, not benchmark training.
"""
import numpy as np


def loss_and_grad(W, b, x, y):
    scores = W @ x + b
    shifted = scores - scores.max()
    exp_scores = np.exp(shifted)
    p = exp_scores / exp_scores.sum()
    loss = -shifted[y] + np.log(exp_scores.sum())
    grad_scores = p.copy()
    grad_scores[y] -= 1
    grad_W = grad_scores[:, None] * x[None, :]
    grad_b = grad_scores.copy()
    return loss, p, grad_W, grad_b


def main():
    w = 0.
    for step in range(4):
        loss = (w - 3.) ** 2
        print(f"scalar step={step}: w={w:.6f}, loss={loss:.6f}")
        if step < 3:
            w -= 0.1 * 2 * (w - 3.)
    np.testing.assert_allclose(w, 1.464)
    for lr in (0.1, 0.5, 1., 1.1):
        next_w = 0. - lr * (-6.)
        print(f"lr={lr}: next_w={next_w:.3f}, next_loss={(next_w-3.)**2:.3f}")

    W = np.array([[1., 0.], [-1., 2.], [0., -1.]])
    b = np.array([0., 0., 1.])
    x = np.array([2., 1.])
    y = 1  # True class B, initially predicted A.
    loss, p, grad_W, grad_b = loss_and_grad(W, b, x, y)
    h = 1e-5
    max_error = 0.
    for parameter, gradient in ((W, grad_W), (b, grad_b)):
        numerical = np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + h
            plus = loss_and_grad(W, b, x, y)[0]
            parameter[index] = original - h
            minus = loss_and_grad(W, b, x, y)[0]
            parameter[index] = original
            numerical[index] = (plus - minus) / (2 * h)
        np.testing.assert_allclose(numerical, gradient, atol=1e-9, rtol=1e-7)
        max_error = max(max_error, float(np.max(np.abs(numerical-gradient))))
    new_W = W - 0.1 * grad_W
    new_b = b - 0.1 * grad_b
    new_loss, new_p, _, _ = loss_and_grad(new_W, new_b, x, y)
    assert new_loss < loss
    print("Initial probabilities:", p)
    print("grad_W:\n", grad_W)
    print("grad_b:", grad_b)
    print("Updated scores:", new_W @ x + new_b)
    print("Updated probabilities:", new_p)
    print("Loss before/after:", loss, new_loss)
    print("Central-difference maximum absolute error:", max_error)
    print("One example, one update; no validation or test score.")


if __name__ == "__main__":
    main()
