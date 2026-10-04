"""Plain NCHW convolution forward calculation (cross-correlation).

Run: python3 experiments/deep_review/lecture05_convolution.py
Requires NumPy. No groups or dilation; teaching inputs, not a trained CNN.
"""
import numpy as np


def conv2d_forward(X, weights, bias, stride=1, padding=0):
    X = np.asarray(X, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    if X.ndim != 4 or weights.ndim != 4:
        raise ValueError("Expected X (N,C_in,H,W), weights (C_out,C_in,K_h,K_w).")
    if not isinstance(stride, (int, np.integer)) or stride < 1:
        raise ValueError("stride must be a positive integer.")
    if not isinstance(padding, (int, np.integer)) or padding < 0:
        raise ValueError("padding must be a nonnegative integer.")
    N, C_in, H, W = X.shape
    C_out, C_weights, K_h, K_w = weights.shape
    if C_in != C_weights or bias.shape != (C_out,) or min(K_h, K_w) < 1:
        raise ValueError("Input channels, bias or kernel shape mismatch.")
    H_out = (H + 2*padding - K_h)//stride + 1
    W_out = (W + 2*padding - K_w)//stride + 1
    if min(H_out, W_out) < 1:
        raise ValueError("Kernel does not fit the padded input.")
    padded = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    out = np.empty((N, C_out, H_out, W_out))
    for n in range(N):
        for c in range(C_out):
            for row in range(H_out):
                for col in range(W_out):
                    top, left = row*stride, col*stride
                    patch = padded[n, :, top:top+K_h, left:left+K_w]
                    out[n, c, row, col] = np.sum(patch * weights[c]) + bias[c]
    return out


def main():
    X = np.arange(1., 10.).reshape(1, 1, 3, 3)
    weights = np.array([[[[1., 0.], [0., -1.]]]])
    out = conv2d_forward(X, weights, np.zeros(1))
    np.testing.assert_allclose(out, np.full((1, 1, 2, 2), -4.))
    print("Single-channel 2x2-kernel output:\n", out[0, 0])

    two_channels = np.array([[[[1., 2.], [3., 4.]], [[10., 20.], [30., 40.]]]])
    mixing_weights = np.array([[[[2.]], [[-1.]]], [[[0.5]], [[0.1]]]])
    mixed = conv2d_forward(two_channels, mixing_weights, np.array([1., -2.]))
    np.testing.assert_allclose(mixed[0, 0], [[-7., -15.], [-23., -31.]])
    np.testing.assert_allclose(mixed[0, 1], [[-0.5, 1.], [2.5, 4.]])
    print("Two input channels, two 1x1 filters:\n", mixed[0])

    # Check independently against windows and tensor contraction for a batch,
    # multiple channels, rectangular kernel, padding and stride.
    rng = np.random.default_rng(231)
    random_X = rng.normal(size=(2, 3, 4, 5))
    random_W = rng.normal(size=(2, 3, 2, 3))
    random_b = rng.normal(size=2)
    actual = conv2d_forward(random_X, random_W, random_b, stride=2, padding=1)
    padded = np.pad(random_X, ((0, 0), (0, 0), (1, 1), (1, 1)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (2, 3), axis=(2, 3))
    windows = windows[:, :, ::2, ::2, :, :]
    expected = np.einsum("nchwij,ocij->nohw", windows, random_W) + random_b[None, :, None, None]
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    print("Independent window-contraction check passed; output shape:", actual.shape)

    rgb = np.zeros((1, 3, 32, 32))
    rgb_W = np.zeros((16, 3, 3, 3))
    rgb_b = np.zeros(16)
    rgb_out = conv2d_forward(rgb, rgb_W, rgb_b, padding=1)
    downsampled = conv2d_forward(rgb, rgb_W, rgb_b, stride=2, padding=1)
    assert rgb_out.shape == (1, 16, 32, 32)
    assert downsampled.shape == (1, 16, 16, 16)
    assert rgb_W.size + rgb_b.size == 448
    print("RGB example: outputs:", rgb_out.shape, downsampled.shape, "parameters:", 448)
    print("Zero-valued RGB arrays check shapes only; no learned image features.")


if __name__ == "__main__":
    main()
