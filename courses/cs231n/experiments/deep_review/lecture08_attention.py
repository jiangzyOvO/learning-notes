"""Single-head 2-D attention and gradients; constructed data, no model training."""
import numpy as np


def attention(Q, K, V, allowed=None):
    if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
        raise ValueError('This teaching implementation expects 2-D arrays')
    if Q.shape[1] != K.shape[1] or K.shape[0] != V.shape[0] or Q.shape[1] == 0:
        raise ValueError('Incompatible Q/K/V shapes')
    scale = np.sqrt(Q.shape[1])
    scores = Q @ K.T / scale
    if allowed is None:
        allowed = np.ones_like(scores, dtype=bool)
    if allowed.dtype != bool or allowed.shape != scores.shape or not allowed.any(axis=1).all():
        raise ValueError('Mask must be boolean, correctly shaped, with a key allowed in every row')
    scores = np.where(allowed, scores, -np.inf)
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    A = exp / exp.sum(axis=1, keepdims=True)
    return A @ V, (Q, K, V, A, scale)


def backward(dZ, cache):
    Q, K, V, A, scale = cache
    dV = A.T @ dZ
    dA = dZ @ V.T
    dS = A * (dA - (dA * A).sum(axis=1, keepdims=True))
    dQ = dS @ K / scale
    dK = dS.T @ Q / scale
    return dQ, dK, dV


def numerical_gradient(fn, array, eps=1e-5):
    gradient = np.zeros_like(array)
    for idx in np.ndindex(array.shape):
        original = array[idx]
        array[idx] = original + eps
        plus = fn()
        array[idx] = original - eps
        minus = fn()
        array[idx] = original
        gradient[idx] = (plus - minus) / (2 * eps)
    return gradient


def main():
    Q = np.array([[1.], [-1.]])
    K = np.log(np.array([[1.], [2.], [3.]]))
    V = np.array([[10.], [20.], [40.]])
    Z, cache = attention(Q, K, V)
    np.testing.assert_allclose(cache[3], [[1/6, 2/6, 3/6], [6/11, 3/11, 2/11]])
    np.testing.assert_allclose(Z[:, 0], [170/6, 200/11])
    print('Hand weights:', cache[3], 'Outputs:', Z[:, 0])

    rng = np.random.default_rng(8)
    errors = []
    for masked in [False, True]:
        Q, K, V = rng.normal(size=(3, 2)), rng.normal(size=(3, 2)), rng.normal(size=(3, 4))
        allowed = np.tril(np.ones((3, 3), dtype=bool)) if masked else None
        Z, cache = attention(Q, K, V, allowed)
        dZ = rng.normal(size=Z.shape)
        fn = lambda: float(np.sum(attention(Q, K, V, allowed)[0] * dZ))
        for name, array, grad in zip(['Q', 'K', 'V'], [Q, K, V], backward(dZ, cache)):
            error = float(np.max(np.abs(numerical_gradient(fn, array) - grad)))
            errors.append(error)
            assert error < 1e-7
            print('masked=', masked, name, 'gradient error:', error)
        np.testing.assert_allclose(cache[3].sum(axis=1), 1)
        if masked:
            assert np.all(cache[3][~allowed] == 0)
            V_changed = V.copy()
            V_changed[-1] += 100
            np.testing.assert_allclose(attention(Q, K, V_changed, allowed)[0][:-1], Z[:-1])
    # Projection gradients: input X feeds three separate branches.
    X = rng.normal(size=(3, 4))
    Wq, Wk, Wv = rng.normal(size=(4, 2)), rng.normal(size=(4, 2)), rng.normal(size=(4, 3))
    Z, cache = attention(X @ Wq, X @ Wk, X @ Wv)
    dZ = rng.normal(size=Z.shape)
    dq, dk, dv = backward(dZ, cache)
    grads = [dq @ Wq.T + dk @ Wk.T + dv @ Wv.T, X.T @ dq, X.T @ dk, X.T @ dv]
    fn = lambda: float(np.sum(attention(X @ Wq, X @ Wk, X @ Wv)[0] * dZ))
    for name, array, grad in zip(['X', 'Wq', 'Wk', 'Wv'], [X, Wq, Wk, Wv], grads):
        error = float(np.max(np.abs(numerical_gradient(fn, array) - grad)))
        errors.append(error)
        assert error < 1e-7
        print(name, 'projection gradient error:', error)
    try:
        attention(Q, K, V, np.zeros((3, 3), dtype=bool))
    except ValueError:
        print('All-masked-row rejection passed')
    else:
        raise AssertionError('All-masked row should fail')
    print('Overall maximum absolute gradient error:', max(errors))


if __name__ == '__main__':
    main()
