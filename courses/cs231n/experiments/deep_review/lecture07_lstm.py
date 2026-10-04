"""Three-step, batched LSTM BPTT; teaching example, not a trained language model.

Row vectors; packed gate order i, f, o, g (our own convention).
Loss = 0.5 * sum(h_final**2 + c_final**2). NumPy only.
"""
import numpy as np


def sigmoid(x):
    return np.exp(-np.logaddexp(0.0, -x))


def step(x, h_prev, c_prev, W, b):
    z = np.concatenate([x, h_prev], axis=1)
    ai, af, ao, ag = np.split(z @ W + b, 4, axis=1)
    i, f, o, g = sigmoid(ai), sigmoid(af), sigmoid(ao), np.tanh(ag)
    c = f * c_prev + i * g
    h = o * np.tanh(c)
    return h, c, (z, c_prev, i, f, o, g, c)


def loss_and_grad(X, W, b, h0, c0):
    h, c = h0.copy(), c0.copy()
    caches = []
    for t in range(X.shape[1]):
        h, c, cache = step(X[:, t, :], h, c, W, b)
        caches.append(cache)
    loss = 0.5 * np.sum(h * h + c * c)
    dh, dc = h.copy(), c.copy()
    dW, db, dX = np.zeros_like(W), np.zeros_like(b), np.zeros_like(X)
    D = X.shape[2]
    for t in reversed(range(X.shape[1])):
        z, c_prev, i, f, o, g, c = caches[t]
        tanh_c = np.tanh(c)
        do = dh * tanh_c
        dc_total = dc + dh * o * (1 - tanh_c**2)
        df, di, dg = dc_total * c_prev, dc_total * g, dc_total * i
        dc = dc_total * f
        da = np.concatenate([
            di * i * (1 - i), df * f * (1 - f),
            do * o * (1 - o), dg * (1 - g**2),
        ], axis=1)
        dW += z.T @ da
        db += da.sum(axis=0)
        dz = da @ W.T
        dX[:, t, :] = dz[:, :D]
        dh = dz[:, D:]
    return float(loss), (dX, dW, db, dh, dc), (h, c)


def numerical_gradient(fn, array, eps=1e-5):
    result = np.zeros_like(array)
    for idx in np.ndindex(array.shape):
        original = array[idx]
        array[idx] = original + eps
        plus = fn()
        array[idx] = original - eps
        minus = fn()
        array[idx] = original
        result[idx] = (plus - minus) / (2 * eps)
    return result


def probabilities(scores, temperature=1.0):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    shifted = scores / temperature
    shifted = shifted - shifted.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def generate(E, W, b, W_y, b_y, start_id, end_id, max_steps=10):
    """Greedy decoding with supplied weights. No training claim."""
    h = np.zeros((1, W_y.shape[0]))
    c = np.zeros_like(h)
    token = start_id
    result = []
    for _ in range(max_steps):
        h, c, _ = step(E[[token]], h, c, W, b)
        token = int(np.argmax(h @ W_y + b_y, axis=1)[0])
        if token == end_id:
            break
        result.append(token)
    return result


def main():
    rng = np.random.default_rng(7)
    N, T, D, K = 2, 3, 2, 2
    X = rng.normal(size=(N, T, D)) * 0.3
    W = rng.normal(size=(D + K, 4 * K)) * 0.2
    b = rng.normal(size=(4 * K,)) * 0.1
    h0, c0 = rng.normal(size=(N, K)) * 0.2, rng.normal(size=(N, K)) * 0.2
    loss, grads, states = loss_and_grad(X, W, b, h0, c0)
    fn = lambda: loss_and_grad(X, W, b, h0, c0)[0]
    errors = []
    for name, array, analytic in zip(['X', 'W', 'b', 'h0', 'c0'],
                                    [X, W, b, h0, c0], grads):
        error = float(np.max(np.abs(numerical_gradient(fn, array) - analytic)))
        errors.append(error)
        print(name, 'max absolute gradient error:', error)
        assert error < 1e-7
    print('Loss:', loss, 'Final h/c:', states)
    c = 0.9 * 2 + 0.2 * (-0.5)
    print('Hand calculation c/h:', c, 0.8 * np.tanh(c))
    p_low, p_high = probabilities(np.array([2., 1., 0.]), .5), probabilities(np.array([2., 1., 0.]), 2.)
    np.testing.assert_allclose(p_low.sum(), 1)
    np.testing.assert_allclose(p_high.sum(), 1)
    assert p_low.max() > p_high.max()
    print('Temperature .5 / 2 probabilities:', p_low, p_high)
    # Three-token toy vocabulary: 0=<START>, 1=a, 2=<END>.
    # Deliberately fixed output bias emits END immediately; checks stopping only.
    E = np.zeros((3, D))
    tokens = generate(E, W, b, np.zeros((K, 3)), np.array([0., 0., 5.]), 0, 2)
    assert tokens == []
    print('Generation END stopping check passed; no language model was trained.')
    print('Overall maximum gradient error:', max(errors))


if __name__ == '__main__':
    main()
