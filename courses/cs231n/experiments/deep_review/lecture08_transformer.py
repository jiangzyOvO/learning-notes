"""Batched Pre-Norm block and tiny ViT forward demo; no training or full block backward.

Uses ReLU MLP, no dropout. NHWC images. Parameters are randomly constructed.
"""
import numpy as np
from lecture08_attention import attention


def layer_norm(x, gamma, beta, eps=1e-5):
    centered = x - x.mean(axis=-1, keepdims=True)
    var = np.mean(centered**2, axis=-1, keepdims=True)
    return centered / np.sqrt(var + eps) * gamma + beta


def multi_head(x, p, heads, allowed=None):
    B, T, D = x.shape
    if D % heads:
        raise ValueError('Demo requires model width divisible by head count')
    d = D // heads
    q, k, v = [x @ p['W' + name] + p['b' + name] for name in ['q', 'k', 'v']]
    q, k, v = [a.reshape(B, T, heads, d).transpose(0, 2, 1, 3) for a in [q, k, v]]
    scores = q @ k.swapaxes(-1, -2) / np.sqrt(d)
    if allowed is not None:
        if allowed.dtype != bool or allowed.shape != (T, T) or not allowed.any(axis=-1).all():
            raise ValueError('Expected a valid T by T allowed mask')
        scores = np.where(allowed, scores, -np.inf)
    shifted = scores - scores.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    A = exp / exp.sum(axis=-1, keepdims=True)
    z = (A @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
    return z @ p['Wo'] + p['bo'], A


def block(x, p, heads, allowed=None):
    normalized = layer_norm(x, p['g1'], p['t1'])
    update, weights = multi_head(normalized, p, heads, allowed)
    u = x + update
    normalized = layer_norm(u, p['g2'], p['t2'])
    hidden = np.maximum(normalized @ p['W1'] + p['b1'], 0)
    return u + hidden @ p['W2'] + p['b2'], weights


def parameters(rng, D, M):
    p = {}
    for name in ['q', 'k', 'v', 'o']:
        p['W' + name] = rng.normal(size=(D, D)) * 0.1
        p['b' + name] = np.zeros(D)
    p.update(W1=rng.normal(size=(D, M)) * 0.1, b1=np.zeros(M),
             W2=rng.normal(size=(M, D)) * 0.1, b2=np.zeros(D),
             g1=np.ones(D), t1=np.zeros(D), g2=np.ones(D), t2=np.zeros(D))
    return p


def patchify(images, patch):
    B, H, W, C = images.shape
    if H % patch or W % patch:
        raise ValueError('Demo requires image dimensions divisible by patch size')
    return images.reshape(B, H // patch, patch, W // patch, patch, C).transpose(
        0, 1, 3, 2, 4, 5).reshape(B, (H // patch) * (W // patch), patch * patch * C)


def main():
    rng = np.random.default_rng(80)
    B, T, D, heads, M = 2, 3, 4, 2, 16
    p = parameters(rng, D, M)
    x = rng.normal(size=(B, T, D))
    output, A = multi_head(x, p, heads)
    # Independent head-by-head reference reuses the previously checked single-head operator.
    for b in range(B):
        pieces = []
        for h in range(heads):
            s = slice(h * (D // heads), (h + 1) * (D // heads))
            q, k, v = [x[b] @ p['W' + n][:, s] + p['b' + n][s] for n in ['q', 'k', 'v']]
            pieces.append(attention(q, k, v)[0])
        expected = np.concatenate(pieces, axis=-1) @ p['Wo'] + p['bo']
        np.testing.assert_allclose(output[b], expected, atol=1e-12)
    np.testing.assert_allclose(A.sum(axis=-1), 1)
    y, _ = block(x, p, heads)
    permutation = [2, 0, 1]
    np.testing.assert_allclose(block(x[:, permutation], p, heads)[0], y[:, permutation], atol=1e-12)
    positions = rng.normal(size=(1, T, D)) * 0.2
    pos_y = block(x + positions, p, heads)[0]
    fixed_pos_y = block(x[:, permutation] + positions, p, heads)[0]
    assert not np.allclose(fixed_pos_y, pos_y[:, permutation])
    causal = np.tril(np.ones((T, T), dtype=bool))
    causal_y, causal_A = block(x, p, heads, causal)
    changed = x.copy()
    changed[:, -1] += 10
    np.testing.assert_allclose(block(changed, p, heads, causal)[0][:, :-1], causal_y[:, :-1], atol=1e-12)
    assert np.all(causal_A[:, :, ~causal] == 0)
    zero = {name: np.zeros_like(value) for name, value in p.items()}
    zero['g1'][:] = zero['g2'][:] = 1
    np.testing.assert_allclose(block(x, zero, heads)[0], x)
    print('MHA shape / weights / block:', output.shape, A.shape, y.shape)
    print('Independent heads, permutation, position, causal and residual checks passed')
    print('Block parameter count:', sum(a.size for a in p.values()))
    # Tiny image demo: 8x8 RGB, 4x4 patches, width 4, two heads, three classes.
    images = rng.normal(size=(2, 8, 8, 3))
    patches = patchify(images, 4)
    np.testing.assert_allclose(patches[:, 0], images[:, :4, :4].reshape(B, -1))
    np.testing.assert_allclose(patches[:, 1], images[:, :4, 4:8].reshape(B, -1))
    Wpatch = rng.normal(size=(48, D)) * 0.1
    tokens = patches @ Wpatch
    cls = np.zeros((B, 1, D))
    tokens = np.concatenate([cls, tokens], axis=1)
    tokens += rng.normal(size=(1, 5, D)) * 0.1
    encoded, _ = block(tokens, p, heads)
    encoded = layer_norm(encoded, np.ones(D), np.zeros(D))
    scores = encoded[:, 0] @ (rng.normal(size=(D, 3)) * 0.1)
    assert scores.shape == (B, 3) and np.isfinite(scores).all()
    print('Tiny ViT patches/tokens/scores:', patches.shape, tokens.shape, scores.shape)
    print('Random scores only; no training or classification accuracy claim')


if __name__ == '__main__':
    main()
