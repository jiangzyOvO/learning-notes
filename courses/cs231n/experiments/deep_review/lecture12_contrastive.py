"""Symmetric SimCLR-style loss on constructed projection vectors.

Order [view1 batch, view2 batch]; raw vectors are L2-normalized inside loss.
No image augmentation pipeline, encoder training or downstream evaluation.
"""
import numpy as np


def contrastive(raw, temperature=.5):
    M = len(raw)
    if raw.ndim != 2 or M < 4 or M % 2 or temperature <= 0:
        raise ValueError('Need two views of B>=2 images and positive temperature')
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError('This teaching example excludes zero-length vectors')
    unit = raw / norms
    partner = (np.arange(M) + M // 2) % M
    scores = unit @ unit.T / temperature
    np.fill_diagonal(scores, -np.inf)
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    loss = (np.log(exp.sum(axis=1)) - shifted[np.arange(M), partner]).mean()
    ds = probabilities.copy()
    ds[np.arange(M), partner] -= 1
    ds /= M
    # A vector appears in both query rows and candidate columns.
    dunit = (ds + ds.T) @ unit / temperature
    draw = (dunit - unit * (dunit * unit).sum(axis=1, keepdims=True)) / norms
    return float(loss), draw, probabilities, partner


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
    scores = np.log(np.array([4., 2., 1.]))
    shifted = scores - scores.max()
    p = np.exp(shifted) / np.exp(shifted).sum()
    np.testing.assert_allclose(p[0], 4/7)
    print('Hand probability/loss:', p[0], -np.log(p[0]))
    raw = np.array([[1., .2, -.1], [-.1, .9, .3], [.8, .1, .2], [.2, 1., -.2]])
    loss, grad, probabilities, partners = contrastive(raw)
    numeric = numerical_gradient(lambda: contrastive(raw)[0], raw)
    error = np.max(np.abs(numeric-grad))
    assert error < 1e-7
    np.testing.assert_array_equal(partners, [2,3,0,1])
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_allclose(np.diag(probabilities), 0)
    scaled = raw * np.array([[2.], [3.], [.5], [4.]])
    np.testing.assert_allclose(contrastive(scaled)[0], loss)
    collapsed = np.ones_like(raw)
    collapse_loss, collapse_grad, collapse_p, _ = contrastive(collapsed)
    np.testing.assert_allclose(collapse_loss, np.log(3))
    np.testing.assert_allclose(collapse_grad, 0, atol=1e-12)
    assert loss < collapse_loss
    pair_mse_collapsed = np.mean((collapsed[:2] - collapsed[2:])**2)
    np.testing.assert_allclose(pair_mse_collapsed, 0)
    print('Partners / probabilities:', partners, probabilities)
    print('Structured / collapsed contrastive losses:', loss, collapse_loss)
    print('Raw normalized-vector gradient error:', error)
    print('Collapsed positive-pair MSE:', pair_mse_collapsed)
    print('Exact symmetric collapse gradient:', np.max(np.abs(collapse_grad)))
    print('Checks passed; no image encoder was trained')


if __name__ == '__main__':
    main()
