"""Scalar examples for lecture 3; run with Python 3, no third-party packages."""
from math import isclose, sqrt


def loss(w):
    return (w - 3.0) ** 2


def gradient(w):
    return 2.0 * (w - 3.0)


def main():
    print('Regularization: J(w) = (w - 3)^2 + lambda * w^2 / 2')
    for lam in (0, 2, 10):
        optimum = 6.0 / (2.0 + lam)
        assert isclose(gradient(optimum) + lam * optimum, 0, abs_tol=1e-12)
        print(f'lambda={lam:2d}, optimum={optimum:.1f}, data_loss={loss(optimum):.2f}')

    print('\nGradient descent: learning_rate=0.1, initial w=0')
    w = 0.0
    expected = [0.0, 0.6, 1.08, 1.464]
    for step in range(4):
        assert isclose(w, expected[step], abs_tol=1e-12)
        print(f'step={step}, w={w:.6f}, loss={loss(w):.6f}')
        w -= 0.1 * gradient(w)

    h = 0.001
    forward = (loss(h) - loss(0)) / h
    centered = (loss(h) - loss(-h)) / (2 * h)
    assert isclose(forward, -5.999, abs_tol=1e-9)
    assert isclose(centered, gradient(0), abs_tol=1e-9)
    print(f'\nNumerical derivative at 0: forward={forward:.6f}, centered={centered:.6f}')

    print('\nLearning rate effect after 20 updates')
    for eta in (0.01, 0.1, 1.0, 1.1):
        w = 0.0
        for _ in range(20):
            w -= eta * gradient(w)
        assert isclose(w - 3, -3 * (1 - 2 * eta) ** 20, rel_tol=1e-9, abs_tol=1e-12)
        print(f'eta={eta:.2f}, w={w:.6f}, loss={loss(w):.6f}')

    beta1, beta2, g = 0.9, 0.999, 2.0
    m = (1 - beta1) * g
    v = (1 - beta2) * g * g
    m_hat = m / (1 - beta1)
    v_hat = v / (1 - beta2)
    scaled = m_hat / sqrt(v_hat)  # Epsilon omitted for this nonzero scalar example.
    assert isclose(m_hat, 2) and isclose(v_hat, 4) and isclose(scaled, 1)
    print(f'\nAdam step 1: m={m:.6f}, v={v:.6f}, corrected_m={m_hat:g}, corrected_v={v_hat:g}')
    print(f'Update = -learning_rate * {scaled:g} (epsilon omitted)')


if __name__ == '__main__':
    main()
