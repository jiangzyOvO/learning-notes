"""Scalar computational graphs and hand-written backpropagation.

Run: python3 experiments/deep_review/lecture04_scalar_backprop.py
Uses the Python standard library. Teaching regression loss, not classification.
"""


def forward_backward(x, y, params):
    w, b, v, c = (params[name] for name in ("w", "b", "v", "c"))
    a = w * x + b
    h = max(0., a)
    prediction = v * h + c
    residual = prediction - y
    loss = 0.5 * residual * residual
    grad_prediction = residual
    grad_v = grad_prediction * h
    grad_c = grad_prediction
    grad_h = grad_prediction * v
    grad_a = grad_h * (1. if a > 0. else 0.)
    grads = {"w": grad_a * x, "b": grad_a, "v": grad_v, "c": grad_c}
    grad_x = grad_a * w
    return loss, {"a": a, "h": h, "prediction": prediction}, grads, grad_x


def main():
    # The same scalar graph as the lecture-note diagram, f=(x+y)*z.
    x, y, z = 1., 2., 4.
    q = x + y
    f = q * z
    grad_f = 1.
    grad_q = grad_f * z
    grad_z = grad_f * q
    grad_x = grad_q
    grad_y = grad_q
    assert (f, grad_x, grad_y, grad_z) == (12., 4., 4., 3.)
    print("f=(x+y)*z:", f, "gradients x/y/z:", grad_x, grad_y, grad_z)
    # Two uses of x contribute separately, so their gradients must add.
    shared_x = 3.
    grad_shared_x = shared_x + shared_x + 1.
    assert grad_shared_x == 7.
    print("f=x*x+x, x=3: accumulated gradient:", grad_shared_x)

    params = {"w": 1., "b": -1., "v": 2., "c": 0.}
    loss, values, grads, input_grad = forward_backward(2., 1., params)
    assert loss == 0.5 and grads == {"w": 4., "b": 2., "v": 1., "c": 1.}
    assert input_grad == 2.
    delta = 1e-5
    max_error = 0.
    for name in params:
        plus, minus = params.copy(), params.copy()
        plus[name] += delta
        minus[name] -= delta
        numerical = (forward_backward(2., 1., plus)[0]
                     - forward_backward(2., 1., minus)[0]) / (2*delta)
        error = abs(numerical - grads[name])
        assert error < 1e-8
        max_error = max(max_error, error)
    print("Scalar network forward:", values, "loss:", loss)
    print("Parameter gradients:", grads, "input gradient:", input_grad)
    print("Central-difference max error:", max_error)
    for lr in (0.01, 0.1):
        # All gradients come from the same old forward pass.
        updated = {name: params[name]-lr*grads[name] for name in params}
        new_loss, new_values, _, _ = forward_backward(2., 1., updated)
        expected = 0.3049805 if lr == 0.01 else 0.605
        assert abs(new_loss-expected) < 1e-12
        print("lr:", lr, "updated params:", updated,
              "forward:", new_values, "new loss:", new_loss)
    # Negative preactivation shuts off this example's first-layer data gradient.
    neg_loss, neg_values, neg_grads, neg_input_grad = forward_backward(0., 1., params)
    assert neg_grads == {"w": 0., "b": 0., "v": 0., "c": -1.}
    assert neg_input_grad == 0.
    print("Negative ReLU example:", neg_values, "loss:", neg_loss, "grads:", neg_grads)
    print("All checks passed. One-example arithmetic, no generalization evaluation.")


if __name__ == "__main__":
    main()
