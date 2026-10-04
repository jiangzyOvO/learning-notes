"""Convolution gradients, max-pooling, GAP and a CNN forward shape trace.

Run: python3 experiments/deep_review/lecture05_backward_pooling.py
Requires NumPy. Teaching implementations; no groups or dilation.
"""
import numpy as np
from lecture05_convolution import conv2d_forward


def conv2d_backward(X, weights, upstream, stride=1, padding=0):
    # Assumes valid shapes matching the preceding forward pass.
    N, _, H, W = X.shape
    C_out, _, K_h, K_w = weights.shape
    padded = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    d_padded = np.zeros_like(padded, dtype=np.float64)
    d_weights = np.zeros_like(weights, dtype=np.float64)
    d_bias = upstream.sum(axis=(0, 2, 3))
    for n in range(N):
        for c in range(C_out):
            for row in range(upstream.shape[2]):
                for col in range(upstream.shape[3]):
                    top, left = row*stride, col*stride
                    patch = padded[n, :, top:top+K_h, left:left+K_w]
                    g = upstream[n, c, row, col]
                    d_weights[c] += g * patch
                    d_padded[n, :, top:top+K_h, left:left+K_w] += g * weights[c]
    return d_padded[:, :, padding:padding+H, padding:padding+W], d_weights, d_bias


def maxpool_forward(X, kernel=2, stride=2):
    N, C, H, W = X.shape
    H_out, W_out = (H-kernel)//stride+1, (W-kernel)//stride+1
    out = np.empty((N, C, H_out, W_out))
    switches = np.empty(out.shape, dtype=np.int64)
    for n in range(N):
        for c in range(C):
            for row in range(H_out):
                for col in range(W_out):
                    patch = X[n, c, row*stride:row*stride+kernel, col*stride:col*stride+kernel]
                    index = np.argmax(patch)
                    switches[n, c, row, col] = index
                    out[n, c, row, col] = patch.flat[index]
    return out, switches


def maxpool_backward(upstream, switches, input_shape, kernel=2, stride=2):
    dx = np.zeros(input_shape)
    for n, c, row, col in np.ndindex(upstream.shape):
        inner_row, inner_col = divmod(int(switches[n, c, row, col]), kernel)
        dx[n, c, row*stride+inner_row, col*stride+inner_col] += upstream[n, c, row, col]
    return dx


def main():
    X = np.arange(1., 10.).reshape(1, 1, 3, 3)
    weights = np.array([[[[1., 0.], [0., -1.]]]])
    dx, dw, db = conv2d_backward(X, weights, np.ones((1, 1, 2, 2)))
    np.testing.assert_allclose(dw[0, 0], [[12., 16.], [24., 28.]])
    np.testing.assert_allclose(dx[0, 0], [[1., 1., 0.], [1., 0., -1.], [0., -1., -1.]])
    np.testing.assert_allclose(db, [4.])
    print("Shared kernel gradient:\n", dw[0, 0], "\nInput gradient:\n", dx[0, 0], "bias:", db)

    rng = np.random.default_rng(231)
    X = rng.normal(size=(2, 2, 3, 4))
    weights = rng.normal(size=(2, 2, 2, 2))
    bias = rng.normal(size=2)
    out = conv2d_forward(X, weights, bias, stride=2, padding=1)
    G = rng.normal(size=out.shape)
    analytical = conv2d_backward(X, weights, G, stride=2, padding=1)
    delta, max_error = 1e-5, 0.
    for parameter, grad in zip((X, weights, bias), analytical):
        numerical = np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + delta
            plus = np.sum(conv2d_forward(X, weights, bias, stride=2, padding=1)*G)
            parameter[index] = original - delta
            minus = np.sum(conv2d_forward(X, weights, bias, stride=2, padding=1)*G)
            parameter[index] = original
            numerical[index] = (plus-minus)/(2*delta)
        np.testing.assert_allclose(numerical, grad, atol=1e-8, rtol=1e-6)
        max_error = max(max_error, float(np.max(np.abs(numerical-grad))))
    print("Convolution numerical-gradient max error:", max_error)

    pool_X = np.array([[[[1., 2., 3.], [4., 9., 6.], [7., 8., 0.]]]])
    pooled, switches = maxpool_forward(pool_X, stride=1)
    pool_dx = maxpool_backward(np.ones_like(pooled), switches, pool_X.shape, stride=1)
    np.testing.assert_allclose(pooled, 9.)
    expected = np.zeros_like(pool_X)
    expected[0, 0, 1, 1] = 4.
    np.testing.assert_array_equal(pool_dx, expected)
    pool_error = 0.
    for index in np.ndindex(pool_X.shape):
        old = pool_X[index]
        pool_X[index] = old+delta
        plus = maxpool_forward(pool_X, stride=1)[0].sum()
        pool_X[index] = old-delta
        minus = maxpool_forward(pool_X, stride=1)[0].sum()
        pool_X[index] = old
        pool_error = max(pool_error, abs((plus-minus)/(2*delta)-pool_dx[index]))
    assert pool_error < 1e-8
    print("Overlapping max-pool input gradient:\n", pool_dx[0, 0])
    print("Max-pool numerical-gradient max error:", pool_error)
    # Ties select the first flattened position in this implementation.
    tie_X = np.ones((1, 1, 2, 2))
    _, tie_switches = maxpool_forward(tie_X)
    tie_dx = maxpool_backward(np.ones((1, 1, 1, 1)), tie_switches, tie_X.shape)
    np.testing.assert_array_equal(tie_dx, [[[[1., 0.], [0., 0.]]]])
    gap_input = np.arange(8.).reshape(1, 2, 2, 2)
    gap = gap_input.mean(axis=(2, 3))
    np.testing.assert_allclose(gap, [[1.5, 5.5]])
    gap_upstream = np.array([[2., -4.]])
    gap_dx = np.broadcast_to(gap_upstream[:, :, None, None]/4, gap_input.shape)
    np.testing.assert_allclose(gap_dx[0, 0], 0.5)
    np.testing.assert_allclose(gap_dx[0, 1], -1.)
    print("GAP values:", gap, "per-position gradients:", gap_dx)

    # Forward shape trace only; zero arrays do not demonstrate learned features.
    rgb = np.zeros((1, 3, 32, 32))
    h1 = np.maximum(conv2d_forward(rgb, np.zeros((16, 3, 3, 3)), np.zeros(16), padding=1), 0.)
    p1, _ = maxpool_forward(h1)
    h2 = np.maximum(conv2d_forward(p1, np.zeros((32, 16, 3, 3)), np.zeros(32), padding=1), 0.)
    p2, _ = maxpool_forward(h2)
    features = p2.mean(axis=(2, 3))
    scores = features @ np.zeros((32, 10)) + np.zeros(10)
    assert scores.shape == (1, 10)
    print("CNN shapes:", rgb.shape, h1.shape, p1.shape, h2.shape, p2.shape, features.shape, scores.shape)
    print("CNN parameter count:", 16*(3*3*3+1)+32*(16*3*3+1)+(32*10+10))
    receptive_field, jump = 1, 1
    for kernel, stride in ((3, 1), (2, 2), (3, 1), (2, 2)):
        receptive_field += (kernel-1)*jump
        jump *= stride
    assert (receptive_field, jump) == (10, 4)
    print("Before GAP: receptive field/jump:", receptive_field, jump)
    print("No complete CNN training or test result is claimed.")


if __name__ == "__main__":
    main()
