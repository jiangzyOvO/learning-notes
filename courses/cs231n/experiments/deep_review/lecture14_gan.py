"""Scalar GAN gradient and alternating-update checks, no image training.

G(z)=a*z+b; D(x)=sigmoid(w*x+c).
D loss: mean real BCE + mean fake BCE; G uses non-saturating BCE.
"""
import numpy as np


def sigmoid(x):
    return np.exp(-np.logaddexp(0., -x))


def d_loss_grad(d_params, g_params, real, noise):
    w, c = d_params
    a, b = g_params
    fake = a*noise+b  # Held fixed for the D parameter derivative.
    real_logit, fake_logit = w*real+c, w*fake+c
    loss = np.logaddexp(0., -real_logit).mean() + np.logaddexp(0., fake_logit).mean()
    dreal = (sigmoid(real_logit)-1)/len(real)
    dfake = sigmoid(fake_logit)/len(fake)
    gradient = np.array([np.sum(dreal*real)+np.sum(dfake*fake), dreal.sum()+dfake.sum()])
    return float(loss), gradient


def g_loss_grad(g_params, d_params, noise):
    a, b = g_params
    w, c = d_params
    fake = a*noise+b
    logits = w*fake+c
    loss = np.logaddexp(0., -logits).mean()
    dfake = (sigmoid(logits)-1)*w/len(fake)
    gradient = np.array([np.sum(dfake*noise), dfake.sum()])
    return float(loss), gradient


def numerical_gradient(fn, params, eps=1e-5):
    gradient = np.zeros_like(params)
    for i in range(len(params)):
        original = params[i]
        params[i] = original+eps
        plus = fn()
        params[i] = original-eps
        minus = fn()
        params[i] = original
        gradient[i] = (plus-minus)/(2*eps)
    return gradient


def main():
    real, noise = np.array([1.,2.,1.5]), np.array([-1.,0.,1.])
    d_params, g_params = np.array([.8,-.1]), np.array([.3,0.])
    d_loss, d_grad = d_loss_grad(d_params, g_params, real, noise)
    d_error = np.max(np.abs(numerical_gradient(
        lambda: d_loss_grad(d_params, g_params, real, noise)[0], d_params)-d_grad))
    g_loss, g_grad = g_loss_grad(g_params, d_params, noise)
    g_error = np.max(np.abs(numerical_gradient(
        lambda: g_loss_grad(g_params, d_params, noise)[0], g_params)-g_grad))
    assert d_error < 1e-7 and g_error < 1e-7
    fixed_g = g_params.copy()
    d_params -= .05*d_grad
    d_after = d_loss_grad(d_params, g_params, real, noise)[0]
    np.testing.assert_array_equal(g_params, fixed_g)
    assert d_after < d_loss
    fixed_d = d_params.copy()
    g_before, g_grad_current = g_loss_grad(g_params, d_params, noise)
    g_params -= .05*g_grad_current
    g_after = g_loss_grad(g_params, d_params, noise)[0]
    np.testing.assert_array_equal(d_params, fixed_d)
    assert g_after < g_before
    # Derivatives wrt fake logit, not full generator parameters.
    tiny_probability = sigmoid(-10.)
    minimax_gradient = -tiny_probability
    nonsaturating_gradient = tiny_probability-1
    assert abs(nonsaturating_gradient) > 10_000*abs(minimax_gradient)
    p_data, p_g = np.array([.8,.2]), np.array([.4,.6])
    optimal_d = p_data/(p_data+p_g)
    np.testing.assert_allclose(optimal_d, [2/3, .25])
    np.testing.assert_allclose(p_data/(p_data+p_data), .5)
    print('D/G gradient errors:', d_error, g_error)
    print('D loss before/after D update:', d_loss, d_after)
    print('G loss before/after G update (same updated D):', g_before, g_after)
    print('Updated D/G params:', d_params, g_params)
    print('Fake logit=-10 minimax/non-saturating derivatives:', minimax_gradient, nonsaturating_gradient)
    print('Equal-prior optimal D for two modes:', optimal_d)
    print('All checks passed; no autograd framework, adversarial convergence or image quality claim')


if __name__ == '__main__':
    main()
