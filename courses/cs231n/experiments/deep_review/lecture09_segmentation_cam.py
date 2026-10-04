"""Pixel CE, ignore labels, segmentation metrics and CAM teaching checks.

Uses NHWC scores; skip-connection demo is not a full U-Net. No trained model.
"""
import numpy as np


def pixel_ce(scores, labels, ignore=-1):
    valid = labels != ignore
    if scores.shape[:-1] != labels.shape or not valid.any():
        raise ValueError('Labels must align and contain at least one valid pixel')
    logits = scores[valid]
    targets = labels[valid]
    if np.any(targets < 0) or np.any(targets >= scores.shape[-1]):
        raise ValueError('Invalid class id')
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    count = len(targets)
    loss = (np.log(exp.sum(axis=-1)) - shifted[np.arange(count), targets]).mean()
    grad_valid = probs.copy()
    grad_valid[np.arange(count), targets] -= 1
    grad = np.zeros_like(scores)
    grad[valid] = grad_valid / count
    return float(loss), grad


def confusion(pred, labels, classes, ignore=-1):
    valid = labels != ignore
    return np.bincount(classes * labels[valid] + pred[valid],
                       minlength=classes**2).reshape(classes, classes)


def numerical_gradient(fn, array, eps=1e-5):
    grad = np.zeros_like(array)
    for idx in np.ndindex(array.shape):
        original = array[idx]
        array[idx] = original + eps
        plus = fn()
        array[idx] = original - eps
        minus = fn()
        array[idx] = original
        grad[idx] = (plus - minus) / (2 * eps)
    return grad


def main():
    rng = np.random.default_rng(9)
    scores = rng.normal(size=(1, 2, 2, 2))
    labels = np.array([[[0, 1], [1, -1]]])
    loss, grad = pixel_ce(scores, labels)
    error = np.max(np.abs(numerical_gradient(lambda: pixel_ce(scores, labels)[0], scores) - grad))
    assert error < 1e-7
    assert np.all(grad[labels == -1] == 0)
    changed = scores.copy()
    changed[labels == -1] = [100., -100.]
    np.testing.assert_allclose(pixel_ce(changed, labels)[0], loss)
    print('Pixel CE loss / max gradient error:', loss, error)
    pred = np.array([[[0, 0], [1, 1]]])
    cm = confusion(pred, labels, 2)
    np.testing.assert_array_equal(cm, [[1, 0], [1, 1]])
    tp = cm.diagonal()
    union = cm.sum(axis=1) + cm.sum(axis=0) - tp
    iou = np.divide(tp, union, out=np.full(2, np.nan), where=union > 0)
    accuracy = tp.sum() / cm.sum()
    np.testing.assert_allclose(iou, [.5, .5])
    print('Confusion (GT rows, prediction columns):', cm)
    print('Valid pixel accuracy / per-class IoU / mean:', accuracy, iou, np.nanmean(iou))
    # Upsample and concatenate an encoder skip; demonstrate only this local operation.
    low = np.arange(4.).reshape(1, 2, 2, 1)
    up = np.repeat(np.repeat(low, 2, axis=1), 2, axis=2)
    skip = rng.normal(size=(1, 4, 4, 2))
    fused = np.concatenate([up, skip], axis=-1)
    output_scores = fused @ rng.normal(size=(3, 2))
    assert fused.shape == (1, 4, 4, 3) and output_scores.shape == (1, 4, 4, 2)
    np.testing.assert_allclose(up[0, :2, :2, 0], 0)
    print('Upsample / concatenate / pointwise classifier:', up.shape, fused.shape, output_scores.shape)
    # GAP + linear head permits an exact CAM identity and analytic Grad-CAM gradient.
    features = np.array([[[1., 2.], [3., 4.]], [[2., 0.], [1., 3.]]])
    weights, bias = np.array([2., -1.]), .5
    score_fn = lambda: float(features.mean(axis=(1, 2)) @ weights + bias)
    cam = np.sum(weights[:, None, None] * features, axis=0)
    np.testing.assert_allclose(cam.mean() + bias, score_fn())
    spatial_size = features.shape[1] * features.shape[2]
    ds_dfeatures = np.broadcast_to(weights[:, None, None] / spatial_size, features.shape)
    cam_error = np.max(np.abs(numerical_gradient(score_fn, features) - ds_dfeatures))
    assert cam_error < 1e-7
    alpha = ds_dfeatures.mean(axis=(1, 2))
    grad_cam = np.maximum(np.sum(alpha[:, None, None] * features, axis=0), 0)
    np.testing.assert_allclose(grad_cam, np.maximum(cam / spatial_size, 0))
    print('Class score / CAM / Grad-CAM:', score_fn(), cam, grad_cam)
    print('CAM score gradient error:', cam_error)
    print('All checks passed; no complete U-Net, trained CNN, or real-image heatmap')


if __name__ == '__main__':
    main()
