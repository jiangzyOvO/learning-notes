"""CPU-only arithmetic checks for Lecture 2; no third-party dependencies.

These verify the hand-worked examples, not PyTorch/CUDA performance.
"""
from math import prod, isclose


def tensor_bytes(shape, bytes_per_element):
    return prod(shape) * bytes_per_element


def matmul(a, b):
    assert a and b and len(a[0]) == len(b)
    return [[sum(x * y for x, y in zip(row, col))
             for col in zip(*b)] for row in a]


def loss_and_grad(w):
    x = [1.0, 2.0, 3.0]
    prediction = sum(a * b for a, b in zip(x, w))
    residual = prediction - 5.0
    return 0.5 * residual ** 2, [residual * a for a in x]


def main():
    shape = (2, 3, 4)
    print('Shape:', shape, 'rank:', len(shape), 'elements:', prod(shape))
    print('float32 bytes:', tensor_bytes(shape, 4))
    print('bfloat16 bytes:', tensor_bytes(shape, 2))
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[1, 0], [0, 1], [1, 1]]
    result = matmul(a, b)
    assert result == [[4, 5], [10, 11]]
    print('Matrix product:', result)
    print('Exact FLOPs:', 2 * 2 * (2 * 3 - 1), 'approximate:', 2 * 2 * 3 * 2)
    w = [1.0, 1.0, 1.0]
    loss, grad = loss_and_grad(w)
    assert loss == 0.5 and grad == [1.0, 2.0, 3.0]
    step = 1e-5
    errors = []
    for i in range(len(w)):
        plus, minus = w.copy(), w.copy()
        plus[i] += step
        minus[i] -= step
        numeric = (loss_and_grad(plus)[0] - loss_and_grad(minus)[0]) / (2 * step)
        errors.append(abs(numeric - grad[i]))
        assert isclose(numeric, grad[i], abs_tol=1e-8)
    new_w = [p - 0.1 * g for p, g in zip(w, grad)]
    new_loss, _ = loss_and_grad(new_w)
    assert isclose(new_loss, 0.08, abs_tol=1e-12)
    print('Gradient:', grad, 'finite-difference max error:', max(errors))
    print('SGD update:', new_w, 'loss:', new_loss)
    n = 1024
    dot_ai = (2 * n - 1) / (4 * n + 2)
    mv_ai = n * (2 * n - 1) / (2 * n * n + 4 * n)
    mm_ai = n * n * (2 * n - 1) / (6 * n * n)
    print('bf16 arithmetic intensity: dot:', dot_ai, 'matvec:', mv_ai, 'matmul:', mm_ai)
    peak, bandwidth = 1979e12 / 2, 3.35e12
    threshold = peak / bandwidth
    assert dot_ai < threshold and mv_ai < threshold < mm_ai
    print('Lecture hardware balance:', threshold, 'FLOPs/byte')
    days = 6 * 70e9 * 15e12 / (peak * 0.5 * 1024) / 86400
    params_upper_bound = 8 * 80e9 / 12
    print('Lecture training-time estimate:', days, 'days')
    print('Lecture AdamW parameter-only upper bound:', params_upper_bound / 1e9, 'B')
    # Mean loss gradients for equal microbatches need equal averaging weights.
    samples = [(1.0, 2.0), (2.0, 1.0), (3.0, 0.0), (4.0, 3.0)]
    scalar_w = 0.5
    sample_grads = [(scalar_w * x - y) * x for x, y in samples]
    full_grad = sum(sample_grads) / len(sample_grads)
    micro_grads = [sum(sample_grads[:2]) / 2, sum(sample_grads[2:]) / 2]
    accumulated = sum(g / 2 for g in micro_grads)
    assert isclose(accumulated, full_grad)
    print('Full-batch gradient:', full_grad, 'weighted accumulated gradient:', accumulated)
    print('All arithmetic checks passed. No PyTorch or GPU benchmark was run.')


if __name__ == '__main__':
    main()
