"""2D BatchNorm/LayerNorm and inverted Dropout teaching arithmetic.

Run: python3 experiments/deep_review/lecture06_normalization_dropout.py
Requires NumPy. BN gradient checks use batch statistics without buffer updates.
"""
import numpy as np


def norm_forward(X, gamma, beta, kind="batch", eps=1e-5):
    # X (N,D); gamma/beta (D,). BatchNorm uses axis 0, LayerNorm axis 1.
    axis = 0 if kind == "batch" else 1
    mean = X.mean(axis=axis, keepdims=True)
    var = X.var(axis=axis, keepdims=True)
    inv_std = 1./np.sqrt(var+eps)
    x_hat = (X-mean)*inv_std
    return x_hat*gamma+beta, (x_hat, inv_std, gamma, axis)


def norm_backward(upstream, cache):
    x_hat, inv_std, gamma, axis = cache
    M = upstream.shape[axis]
    dx_hat = upstream*gamma
    dx = inv_std/M*(M*dx_hat-dx_hat.sum(axis=axis, keepdims=True)
                   -x_hat*(dx_hat*x_hat).sum(axis=axis, keepdims=True))
    dgamma = (upstream*x_hat).sum(axis=0)
    dbeta = upstream.sum(axis=0)
    return dx, dgamma, dbeta


def main():
    one_feature = np.array([[1.], [3.]])
    y, _ = norm_forward(one_feature, np.array([2.]), np.array([1.]))
    np.testing.assert_allclose(y[:, 0], [-1., 3.], atol=2e-5)
    print("BN [1,3], gamma=2, beta=1:", y[:, 0])
    X = np.array([[1., 10.], [3., 30.], [5., 50.]])
    gamma, beta = np.ones(2), np.zeros(2)
    batch_y, _ = norm_forward(X, gamma, beta)
    layer_y, _ = norm_forward(X, gamma, beta, kind="layer")
    changed = X.copy()
    changed[2, 0] = 105.
    changed_bn, _ = norm_forward(changed, gamma, beta)
    changed_ln, _ = norm_forward(changed, gamma, beta, kind="layer")
    assert not np.allclose(batch_y[0], changed_bn[0])
    np.testing.assert_array_equal(layer_y[0], changed_ln[0])
    print("BN across rows:\n", batch_y)
    print("LN within each row:\n", layer_y)
    print("Replacing another sample changes first-sample BN:", batch_y[0], changed_bn[0])
    print("First-sample LN unchanged:", layer_y[0], changed_ln[0])

    # Running buffers: new = (1-alpha)*old + alpha*batch statistic.
    alpha = 0.1
    running_mean = np.zeros(2)
    running_var = np.ones(2)
    running_mean = (1-alpha)*running_mean + alpha*X.mean(axis=0)
    running_var = (1-alpha)*running_var + alpha*X.var(axis=0, ddof=1)
    eval_y = (X-running_mean)/np.sqrt(running_var+1e-5)*gamma+beta
    before = running_mean.copy(), running_var.copy()
    # Eval formula reads buffers, does not change them or mix other samples.
    eval_first = (changed[:1]-running_mean)/np.sqrt(running_var+1e-5)*gamma+beta
    np.testing.assert_allclose(eval_y[:1], eval_first)
    np.testing.assert_array_equal(before[0], running_mean)
    np.testing.assert_array_equal(before[1], running_var)
    print("After one running-stat update: mean/var:", running_mean, running_var)
    print("Eval outputs (early statistics, not calibrated):\n", eval_y)
    cnn = np.arange(24.).reshape(2, 3, 2, 2)
    cnn_mean = cnn.mean(axis=(0, 2, 3), keepdims=True)
    assert cnn_mean.shape == (1, 3, 1, 1)
    print("CNN BN statistic shape:", cnn_mean.shape)

    rng = np.random.default_rng(231)
    X_random = rng.normal(size=(4, 3))
    gamma_random, beta_random = rng.normal(size=3), rng.normal(size=3)
    G = rng.normal(size=X_random.shape)
    delta = 1e-5
    for kind in ("batch", "layer"):
        _, cache = norm_forward(X_random, gamma_random, beta_random, kind)
        analytical = norm_backward(G, cache)
        max_error = 0.
        for parameter, grad in zip((X_random, gamma_random, beta_random), analytical):
            numerical = np.zeros_like(parameter)
            for index in np.ndindex(parameter.shape):
                old = parameter[index]
                parameter[index] = old+delta
                plus = np.sum(norm_forward(X_random, gamma_random, beta_random, kind)[0]*G)
                parameter[index] = old-delta
                minus = np.sum(norm_forward(X_random, gamma_random, beta_random, kind)[0]*G)
                parameter[index] = old
                numerical[index] = (plus-minus)/(2*delta)
            np.testing.assert_allclose(numerical, grad, atol=1e-8, rtol=1e-6)
            max_error = max(max_error, float(np.max(np.abs(numerical-grad))))
        print(kind, "input/gamma/beta gradient maximum error:", max_error)

    h = np.array([2., 4.])
    keep_prob = 0.5
    mask = np.array([1., 0.])
    dropped = h*mask/keep_prob
    dh = np.array([3., 5.])*mask/keep_prob
    np.testing.assert_array_equal(dropped, [4., 0.])
    np.testing.assert_array_equal(dh, [6., 0.])
    all_masks = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    np.testing.assert_allclose((all_masks*h/keep_prob).mean(axis=0), h)
    print("Dropout train output/gradient:", dropped, dh, "eval output:", h)
    print("Enumerating all equiprobable masks recovers the input expectation.")
    print("No model training or validation accuracy; normalization checks are NumPy implementations.")


if __name__ == "__main__":
    main()
