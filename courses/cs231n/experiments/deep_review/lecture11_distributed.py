"""Single-process NumPy simulation of gradient aggregation and tensor splits.

No GPUs, communication backend, DDP, or FSDP are executed.
"""
import numpy as np


def loss_and_grad(X, y, w):
    residual = X @ w - y
    return float(0.5 * np.mean(residual**2)), X.T @ residual / len(X)


def aggregate(parts, X, y, w):
    counts = np.array([len(indices) for indices in parts])
    local_grads = np.stack([loss_and_grad(X[idx], y[idx], w)[1] for idx in parts])
    weighted = (local_grads * counts[:, None]).sum(axis=0) / counts.sum()
    return weighted, local_grads.mean(axis=0)


def normalize_batch(x):
    return (x - x.mean(axis=0)) / np.sqrt(x.var(axis=0) + 1e-5)


def main():
    X = np.array([[1., 0.], [0., 1.], [1., 1.], [2., 1.], [1., 3.], [-1., 2.]])
    y = np.array([1., -1., 2., 0., 3., -2.])
    w = np.array([.2, -.3])
    loss, full_grad = loss_and_grad(X, y, w)
    equal_parts = [np.arange(3), np.arange(3, 6)]
    aggregated, simple = aggregate(equal_parts, X, y, w)
    np.testing.assert_allclose(aggregated, full_grad)
    np.testing.assert_allclose(simple, full_grad)
    unequal_parts = [np.arange(2), np.arange(2, 6)]
    weighted, wrong = aggregate(unequal_parts, X, y, w)
    np.testing.assert_allclose(weighted, full_grad)
    assert not np.allclose(wrong, full_grad)
    # Sequential microbatches use the same parameter version until a single update.
    accumulated = np.zeros_like(w)
    for idx in [np.array([0]), np.array([1,2]), np.array([3,4,5])]:
        accumulated += len(idx) * loss_and_grad(X[idx], y[idx], w)[1]
    accumulated /= len(X)
    np.testing.assert_allclose(accumulated, full_grad)
    lr = .1
    np.testing.assert_allclose(w-lr*weighted, w-lr*full_grad)
    sequential_w = w.copy()
    for idx in equal_parts:
        sequential_w -= lr * loss_and_grad(X[idx], y[idx], sequential_w)[1]
    assert not np.allclose(sequential_w, w-lr*full_grad)
    print('Global loss / gradient:', loss, full_grad)
    print('Unequal split weighted / wrong unweighted:', weighted, wrong)
    print('Accumulated one update / two sequential updates:', w-lr*accumulated, sequential_w)
    # Local batch normalization breaks a generally exact big-batch equivalence.
    values = np.array([[1.], [3.], [10.], [20.]])
    big_bn = normalize_batch(values)
    local_bn = np.concatenate([normalize_batch(values[:2]), normalize_batch(values[2:])])
    assert not np.allclose(big_bn, local_bn)
    print('Global/local BN differ:', big_bn[:,0], local_bn[:,0])
    rng = np.random.default_rng(11)
    inputs = rng.normal(size=(3, 4))
    W = rng.normal(size=(4, 6))
    left, right = inputs @ W[:, :3], inputs @ W[:, 3:]
    hidden = np.concatenate([left, right], axis=1)
    np.testing.assert_allclose(hidden, inputs @ W)
    U = rng.normal(size=(6, 2))
    result = left @ U[:3] + right @ U[3:]
    np.testing.assert_allclose(result, (inputs @ W) @ U)
    print('Tensor column-concatenate / row-sum checks passed')
    print('FP32 Adam model-state accounting for 1e9 parameters:', 1_000_000_000*4*4/1e9, 'decimal GB')
    print('All checks passed; no actual distributed execution or speedup measurement')


if __name__ == '__main__':
    main()
