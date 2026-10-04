"""Initialization, hidden-unit symmetry and fitting four teaching samples.

Run: python3 experiments/deep_review/lecture06_initialization.py
Requires NumPy. No held-out data; small-set fit checks optimization only.
"""
import numpy as np
from lecture04_matrix_backprop import loss_and_grad


def main():
    X = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    labels = np.array([0, 1, 1, 0])
    zero = {"W1": np.zeros((2, 4)), "b1": np.zeros(4),
            "W2": np.zeros((4, 2)), "b2": np.zeros(2)}
    zero_loss, zero_grads, _, _ = loss_and_grad(zero, X, labels, reg=0.)
    assert all(np.all(gradient == 0.) for gradient in zero_grads.values())
    print("All-zero ReLU network, balanced labels: loss:", zero_loss,
          "all parameter gradients zero:", True)

    symmetric = {"W1": np.full((2, 2), 0.2), "b1": np.full(2, 0.1),
                 "W2": np.tile([0.1, -0.1], (2, 1)), "b2": np.zeros(2)}
    _, grads, _, cache = loss_and_grad(symmetric, X, labels, reg=0.)
    np.testing.assert_allclose(cache["h"][:, 0], cache["h"][:, 1])
    np.testing.assert_allclose(grads["W1"][:, 0], grads["W1"][:, 1])
    np.testing.assert_allclose(grads["W2"][0], grads["W2"][1])
    updated = {k: symmetric[k] - 0.1*grads[k] for k in symmetric}
    np.testing.assert_allclose(updated["W1"][:, 0], updated["W1"][:, 1])
    print("Two identical hidden units have identical gradients and remain identical after one update.")

    rng = np.random.default_rng(231)
    initial = rng.normal(size=(256, 128))
    bases = [rng.normal(size=(128, 128)) for _ in range(6)]
    for name, std in (("std=0.01", 0.01), ("std=1", 1.), ("He std=sqrt(2/128)", np.sqrt(2/128))):
        h = initial.copy()
        moments = []
        for base in bases:
            h = np.maximum(h @ (std*base), 0.)
            moments.append(float(np.mean(h*h)))
        assert np.all(np.isfinite(moments))
        print(name, "activation second moments:", [f"{q:.3e}" for q in moments])

    # ReLU first layer uses He normal. Linear output uses a moderate fan-in scale.
    rng = np.random.default_rng(231)
    width = 16
    params = {
        "W1": rng.normal(size=(2, width))*np.sqrt(2/2),
        "b1": np.zeros(width),
        "W2": rng.normal(size=(width, 2))*np.sqrt(1/width),
        "b2": np.zeros(2),
    }
    initial_loss = loss_and_grad(params, X, labels, reg=0.)[0]
    for step in range(600):
        _, grads, _, _ = loss_and_grad(params, X, labels, reg=0.)
        for name in params:
            params[name] -= 0.1 * grads[name]
    final_loss, _, _, cache = loss_and_grad(params, X, labels, reg=0.)
    prediction = cache["scores"].argmax(axis=1)
    np.testing.assert_array_equal(prediction, labels)
    assert final_loss < initial_loss
    print("Four-sample fit, mean cross-entropy before/after:", initial_loss, final_loss)
    print("Predictions:", prediction, "training accuracy:", np.mean(prediction == labels))
    print("600 full-batch SGD steps, lr=0.1, no regularization, no dropout.")
    print("Fitting four samples does not establish held-out accuracy or initialization superiority.")


if __name__ == "__main__":
    main()
