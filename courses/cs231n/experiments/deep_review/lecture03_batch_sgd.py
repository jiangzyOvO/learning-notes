"""Batch-mean cross-entropy, L2 gradients and shuffled mini-batch SGD.

Run: python3 experiments/deep_review/lecture03_batch_sgd.py
Requires NumPy. Uses small constructed data, not an image benchmark.
"""
import numpy as np


def loss_and_grad(W, b, X, y, reg=0.):
    # Shapes: X (B,D), W (C,D), b (C,), y (B,).
    B = len(X)
    if B == 0:
        raise ValueError("A batch must contain at least one sample.")
    scores = X @ W.T + b
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    sum_exp = exp_scores.sum(axis=1, keepdims=True)
    probabilities = exp_scores / sum_exp
    data_loss = np.mean(-shifted[np.arange(B), y] + np.log(sum_exp[:, 0]))
    reg_loss = 0.5 * reg * np.sum(W * W)
    grad_scores = probabilities.copy()
    grad_scores[np.arange(B), y] -= 1
    grad_scores /= B
    grad_W = grad_scores.T @ X + reg * W
    grad_b = grad_scores.sum(axis=0)
    return data_loss + reg_loss, data_loss, reg_loss, grad_W, grad_b


def main():
    W = np.array([[1., 0.], [-1., 2.], [0., -1.]])
    b = np.array([0., 0., 1.])
    X = np.array([[2., 1.], [0., 2.], [1., -1.]])
    y = np.array([1, 1, 2])
    reg = 0.1
    total, data, penalty, gW, gb = loss_and_grad(W, b, X, y, reg)
    single_grads = [loss_and_grad(W, b, X[i:i+1], y[i:i+1], 0.)
                    for i in range(len(X))]
    np.testing.assert_allclose(gW, np.mean([g[3] for g in single_grads], axis=0)+reg*W)
    np.testing.assert_allclose(gb, np.mean([g[4] for g in single_grads], axis=0))
    repeated = loss_and_grad(W, b, np.repeat(X, 2, axis=0), np.repeat(y, 2), reg)
    for original, duplicated in zip((total, data, penalty, gW, gb), repeated):
        np.testing.assert_allclose(original, duplicated)
    max_error = 0.
    h = 1e-5
    for parameter, gradient in ((W, gW), (b, gb)):
        numerical = np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + h
            plus = loss_and_grad(W, b, X, y, reg)[0]
            parameter[index] = original - h
            minus = loss_and_grad(W, b, X, y, reg)[0]
            parameter[index] = original
            numerical[index] = (plus-minus)/(2*h)
        np.testing.assert_allclose(numerical, gradient, atol=1e-9, rtol=1e-7)
        max_error = max(max_error, float(np.max(np.abs(numerical-gradient))))
    print("Batch example: total/data/L2:", total, data, penalty)
    print("Batch grad_W:\n", gW)
    print("Batch grad_b:", gb)
    print("Duplicating the entire batch leaves mean loss and gradient unchanged.")
    print("Central-difference maximum absolute error:", max_error)

    # Six samples; batch size four leaves a final batch of size two.
    X_toy = np.array([[-2., 0.], [-1., 1.], [-1., -1.],
                      [2., 0.], [1., 1.], [1., -1.]])
    y_toy = np.array([0, 0, 0, 1, 1, 1])
    rng = np.random.default_rng(231)
    weights = rng.normal(0., 0.01, size=(2, 2))
    bias = np.zeros(2)
    initial = loss_and_grad(weights, bias, X_toy, y_toy, reg)[0]
    step_count = 0
    for epoch in range(5):
        order = rng.permutation(len(X_toy))
        batch_sizes = []
        for start in range(0, len(order), 4):
            indices = order[start:start+4]
            _, _, _, grad_W, grad_b = loss_and_grad(
                weights, bias, X_toy[indices], y_toy[indices], reg
            )
            weights -= 0.1 * grad_W
            bias -= 0.1 * grad_b
            step_count += 1
            batch_sizes.append(len(indices))
        print(f"epoch={epoch+1}, batch sizes={batch_sizes}, "
              f"full toy training objective={loss_and_grad(weights, bias, X_toy, y_toy, reg)[0]:.6f}")
    final = loss_and_grad(weights, bias, X_toy, y_toy, reg)[0]
    assert step_count == 10 and final < initial
    print("Toy training objective before/after:", initial, final)
    print("Updates:", step_count)
    print("No validation or test set; decreasing training objective is not generalization evidence.")


if __name__ == "__main__":
    main()
