"""Three scalar RNN steps and full BPTT with finite-difference checks.

Run: python3 experiments/deep_review/lecture07_rnn_bptt.py
Requires NumPy. Final-state square loss is a teaching objective, not language CE.
"""
import numpy as np


def forward_backward(inputs, params, h0=0., truncate=False):
    u, r, b = params
    states = [float(h0)]
    for x in inputs:
        states.append(float(np.tanh(u*x + r*states[-1] + b)))
    loss = 0.5*states[-1]**2
    dh = states[-1]
    grads = np.zeros(3)
    dx = np.zeros_like(inputs)
    records = []
    for t in range(len(inputs)-1, -1, -1):
        da = dh*(1-states[t+1]**2)
        contributions = np.array([da*inputs[t], da*states[t], da])
        grads += contributions
        dx[t] = da*u
        records.append((t+1, dh, da, contributions.copy()))
        dh = da*r
        if truncate:
            # Stop after the last time step; earlier states treated as fixed.
            dh = 0.
            break
    return loss, np.array(states), grads, dx, dh, records


def main():
    inputs = np.array([1., 0., -1.])
    params = np.array([1., 0.5, 0.])
    loss, states, grads, dx, dh0, records = forward_backward(inputs, params)
    max_error, delta = 0., 1e-5
    for parameter, analytical in ((params, grads), (inputs, dx)):
        for index in np.ndindex(parameter.shape):
            old = parameter[index]
            parameter[index] = old+delta
            plus = forward_backward(inputs, params)[0]
            parameter[index] = old-delta
            minus = forward_backward(inputs, params)[0]
            parameter[index] = old
            numerical = (plus-minus)/(2*delta)
            max_error = max(max_error, abs(numerical-analytical[index]))
            assert abs(numerical-analytical[index]) < 1e-8
    numerical_h0 = (forward_backward(inputs, params, delta)[0]
                    - forward_backward(inputs, params, -delta)[0])/(2*delta)
    assert abs(numerical_h0-dh0) < 1e-8
    print("States h0,h1,h2,h3:", states)
    print("Final-state square loss:", loss)
    for t, dh, da, contribution in records:
        print(f"reverse t={t}: dh={dh:.6f}, da={da:.6f}, u/r/b contributions={contribution}")
    print("Shared parameter gradient u/r/b:", grads)
    print("Input gradients:", dx, "initial-state gradient:", dh0)
    print("Numerical-gradient maximum error:", max_error)
    print("Last-step-only gradient (truncated, different derivative):",
          forward_backward(inputs, params, truncate=True)[2])
    reversed_states = forward_backward(inputs[::-1].copy(), params)[1]
    assert not np.isclose(states[-1], reversed_states[-1])
    print("Reversed-input final state:", reversed_states[-1])
    gradient = np.array([3., 4.])
    threshold = 2.
    clipped = gradient*min(1., threshold/np.linalg.norm(gradient))
    np.testing.assert_allclose(clipped, [1.2, 1.6])
    print("Norm clipping [3,4] at norm 2:", clipped)
    print("Repeated multipliers over 20 steps:", 0.5**20, 1.5**20)
    print("No sequence-model training or language benchmark score.")


if __name__ == "__main__":
    main()
