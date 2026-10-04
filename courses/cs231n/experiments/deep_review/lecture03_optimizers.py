"""Optimizer arithmetic and persistent state, not a model benchmark.

Run: python3 experiments/deep_review/lecture03_optimizers.py
Requires NumPy. Prescribed gradients illustrate the update rules.
"""
import numpy as np


def momentum_step(parameter, gradient, velocity, lr=0.1, mu=0.9):
    velocity = mu * velocity + gradient
    return parameter - lr * velocity, velocity


def adam_step(parameter, gradient, m, v, t, lr=0.1,
              beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.):
    # t starts at one. gradient is the data gradient for decoupled AdamW.
    m = beta1 * m + (1-beta1) * gradient
    v = beta2 * v + (1-beta2) * gradient * gradient
    m_hat = m / (1-beta1**t)
    v_hat = v / (1-beta2**t)
    new_parameter = (1-lr*weight_decay)*parameter - lr*m_hat/(np.sqrt(v_hat)+eps)
    return new_parameter, m, v, m_hat, v_hat


def main():
    parameter, velocity = np.zeros(1), np.zeros(1)
    for step, gradient in enumerate((2., 2., 2.), 1):
        parameter, velocity = momentum_step(parameter, np.array([gradient]), velocity)
        print(f"Momentum constant-gradient step={step}: velocity={velocity[0]:.6f}, "
              f"parameter={parameter[0]:.6f}")
    np.testing.assert_allclose(velocity, [5.42])
    np.testing.assert_allclose(parameter, [-1.122])
    velocity = np.zeros(1)
    history = []
    parameter = np.zeros(1)
    for gradient in (2., -2., 2.):
        parameter, velocity = momentum_step(parameter, np.array([gradient]), velocity)
        history.append(float(velocity[0]))
    np.testing.assert_allclose(history, [2., -0.2, 1.82])
    print("Alternating-gradient Momentum velocities:", history)

    parameter, m, v = np.zeros(1), np.zeros(1), np.zeros(1)
    for t in (1, 2):
        parameter, m, v, mh, vh = adam_step(parameter, np.array([2.]), m, v, t)
        np.testing.assert_allclose(mh, [2.])
        np.testing.assert_allclose(vh, [4.])
        print(f"Adam t={t}: m={m[0]:.6f}, v={v[0]:.6f}, "
              f"m_hat={mh[0]:.6f}, v_hat={vh[0]:.6f}, parameter={parameter[0]:.6f}")
    np.testing.assert_allclose(parameter, [-0.2], atol=1e-8)

    parameter = np.array([2., -3.])
    zero = np.zeros_like(parameter)
    coupled = adam_step(parameter, 0.1*parameter, zero, zero, 1)[0]
    decoupled = adam_step(parameter, zero, zero, zero, 1, weight_decay=0.1)[0]
    np.testing.assert_allclose(coupled, [1.9, -2.9], atol=1e-8)
    np.testing.assert_allclose(decoupled, [1.98, -2.97])
    print("Zero data gradient, Adam with coupled L2:", coupled)
    print("Zero data gradient, decoupled AdamW:", decoupled)
    unchanged = adam_step(parameter, zero, zero, zero, 1)[0]
    np.testing.assert_array_equal(unchanged, parameter)

    lr_max, lr_min, total_steps = 0.1, 0.001, 100
    rates = [lr_min+0.5*(lr_max-lr_min)*(1+np.cos(np.pi*t/total_steps))
             for t in (0, 50, 100)]
    np.testing.assert_allclose(rates, [0.1, 0.0505, 0.001])
    print("Cosine schedule at steps 0, 50, 100:", rates)
    print("All arithmetic checks passed. Prescribed gradients are not training results.")


if __name__ == "__main__":
    main()
