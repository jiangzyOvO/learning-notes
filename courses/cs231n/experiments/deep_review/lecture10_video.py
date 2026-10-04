"""Video shape, sampling and valid stride-1 3D cross-correlation examples.

NCTHW input; no padding, training or complete video model.
"""
import numpy as np


def conv3d_valid(x, weights, bias):
    N, C, T, H, W = x.shape
    O, Cin, Kt, Kh, Kw = weights.shape
    if C != Cin or T < Kt or H < Kh or W < Kw:
        raise ValueError('Incompatible input/kernel')
    output = np.empty((N, O, T-Kt+1, H-Kh+1, W-Kw+1), dtype=float)
    for n in range(N):
        for o in range(O):
            for t in range(T-Kt+1):
                for h in range(H-Kh+1):
                    for w in range(W-Kw+1):
                        window = x[n, :, t:t+Kt, h:h+Kh, w:w+Kw]
                        output[n, o, t, h, w] = np.sum(window * weights[o]) + bias[o]
    return output


def temporal_iou(a, b):
    intersection = max(0., min(a[1], b[1])-max(a[0], b[0]))
    union = (a[1]-a[0]) + (b[1]-b[0]) - intersection
    return intersection/union if union > 0 else 0.


def main():
    fps, interval, count = 30, 3, 16
    indices = np.arange(count) * interval
    times = indices / fps
    np.testing.assert_allclose(times[-1]-times[0], 1.5)
    print('Sampling indices:', indices, 'Endpoint time span:', times[-1]-times[0])
    video = np.array([1., 3.]).reshape(1, 1, 2, 1, 1)
    reversed_video = video[:, :, ::-1]
    np.testing.assert_allclose(video.mean(axis=2), reversed_video.mean(axis=2))
    difference_kernel = np.array([-1., 1.]).reshape(1, 1, 2, 1, 1)
    forward = conv3d_valid(video, difference_kernel, np.zeros(1))
    backward = conv3d_valid(reversed_video, difference_kernel, np.zeros(1))
    np.testing.assert_allclose(forward, 2)
    np.testing.assert_allclose(backward, -2)
    print('Mean before/after reversing:', video.mean(), reversed_video.mean())
    print('Temporal difference before/after:', forward.item(), backward.item())
    rng = np.random.default_rng(10)
    x = rng.normal(size=(2, 2, 4, 3, 4))
    weights = rng.normal(size=(3, 2, 2, 2, 3))
    bias = rng.normal(size=3)
    result = conv3d_valid(x, weights, bias)
    windows = np.lib.stride_tricks.sliding_window_view(x, (2,2,3), axis=(2,3,4))
    reference = np.einsum('ncthwijk,ocijk->nothw', windows, weights) + bias[None,:,None,None,None]
    np.testing.assert_allclose(result, reference, atol=1e-12)
    print('3D convolution input/weight/output:', x.shape, weights.shape, result.shape)
    # Inflated kernel preserves response for repeated static frames, valid temporal windows.
    image = rng.normal(size=(1, 2, 1, 4, 4))
    spatial_weights = rng.normal(size=(3, 2, 1, 2, 2))
    spatial_bias = rng.normal(size=3)
    static_clip = np.repeat(image, 5, axis=2)
    inflated_weights = np.repeat(spatial_weights, 3, axis=2) / 3
    spatial_result = conv3d_valid(image, spatial_weights, spatial_bias)
    inflated_result = conv3d_valid(static_clip, inflated_weights, spatial_bias)
    np.testing.assert_allclose(inflated_result, np.repeat(spatial_result, 3, axis=2), atol=1e-12)
    print('Static-frame inflation equality passed (valid convolution, no temporal padding)')
    np.testing.assert_allclose(temporal_iou([2,6], [4,8]), 1/3)
    print('Interval IoU:', temporal_iou([2,6], [4,8]))
    print('3x3x3 Conv, 3 input/16 output channels parameter count:', 16*(3*27+1))
    print('All constructed checks passed; no optical-flow estimator or trained video classifier')


if __name__ == '__main__':
    main()
