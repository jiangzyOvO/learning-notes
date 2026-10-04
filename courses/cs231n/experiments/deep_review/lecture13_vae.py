"""Scalar linear Gaussian VAE: sampled loss/backward and exact ELBO check.

Encoder mu=a*x+b, logvar=c*x+d. Decoder x|z ~ N(w*z+e, 1).
Prior z ~ N(0,1). Fixed epsilon during numerical gradient checks only.
No image network or dataset training.
"""
import numpy as np


def sampled_loss_and_grad(params, x, epsilon):
    a, b, c, d, w, e = params
    mu, logvar = a*x+b, c*x+d
    variance = np.exp(logvar)
    std = np.exp(.5*logvar)
    z = mu + std*epsilon
    mean = w*z+e
    residual = mean-x
    reconstruction = .5*(residual**2 + np.log(2*np.pi))
    kl = .5*(mu**2 + variance - 1 - logvar)
    loss = (reconstruction + kl).mean()
    N = len(x)
    dmean = residual/N
    dz = dmean*w
    dmu = dz + mu/N
    dlogvar = dz*.5*std*epsilon + .5*(variance-1)/N
    gradients = np.array([np.sum(dmu*x), np.sum(dmu),
                          np.sum(dlogvar*x), np.sum(dlogvar),
                          np.sum(dmean*z), np.sum(dmean)])
    dx = dmu*a + dlogvar*c - dmean
    return float(loss), gradients, dx, (reconstruction.mean(), kl.mean())


def exact_elbo(x, mu, variance, w, e):
    expected_reconstruction_log_prob = -.5*((x-w*mu-e)**2 + w*w*variance + np.log(2*np.pi))
    kl_prior = .5*(mu*mu+variance-1-np.log(variance))
    elbo = expected_reconstruction_log_prob-kl_prior
    marginal_variance = 1+w*w
    log_evidence = -.5*((x-e)**2/marginal_variance + np.log(2*np.pi*marginal_variance))
    posterior_variance = 1/marginal_variance
    posterior_mu = w*(x-e)/marginal_variance
    posterior_gap = .5*(np.log(posterior_variance/variance)
                       +(variance+(mu-posterior_mu)**2)/posterior_variance-1)
    return float(elbo), float(log_evidence), float(posterior_gap), posterior_mu, posterior_variance


def numeric_gradient(fn, array, eps=1e-5):
    result = np.zeros_like(array)
    for idx in np.ndindex(array.shape):
        original = array[idx]
        array[idx] = original+eps
        plus = fn()
        array[idx] = original-eps
        minus = fn()
        array[idx] = original
        result[idx] = (plus-minus)/(2*eps)
    return result


def main():
    params = np.array([.4, .1, -.2, -.3, .8, -.1])
    x = np.array([-1., 0., 1.])
    epsilon = np.array([.2, -.5, 1.])
    loss, gradient, dx, terms = sampled_loss_and_grad(params, x, epsilon)
    fn = lambda: sampled_loss_and_grad(params, x, epsilon)[0]
    param_error = np.max(np.abs(numeric_gradient(fn, params)-gradient))
    input_error = np.max(np.abs(numeric_gradient(fn, x)-dx))
    assert param_error < 1e-7 and input_error < 1e-7
    updated = params-.05*gradient
    after = sampled_loss_and_grad(updated, x, epsilon)[0]
    assert after < loss
    np.testing.assert_allclose(.5*(1+4-1-np.log(4)), 1.3068528194400546)
    np.testing.assert_allclose(.5*(0+1-1-np.log(1)), 0)
    np.testing.assert_allclose(1+np.exp(.5*np.log(4))*.5, 2)
    elbo, evidence, gap, posterior_mu, posterior_variance = exact_elbo(.7, .2, .8, .8, -.1)
    np.testing.assert_allclose(evidence-elbo, gap, atol=1e-12)
    assert gap >= 0
    exact_q_elbo, _, exact_gap, _, _ = exact_elbo(.7, posterior_mu, posterior_variance, .8, -.1)
    np.testing.assert_allclose(exact_q_elbo, evidence, atol=1e-12)
    np.testing.assert_allclose(exact_gap, 0, atol=1e-12)
    # If decoder ignores z (w=0), exact posterior is the prior.
    _, _, ignored_gap, ignored_mu, ignored_variance = exact_elbo(.7, 0., 1., 0., -.1)
    np.testing.assert_allclose([ignored_mu, ignored_variance, ignored_gap], [0,1,0])
    print('Sampled loss reconstruction/KL:', loss, terms)
    print('Parameter/input gradient errors:', param_error, input_error)
    print('Fixed-noise one-step loss before/after:', loss, after)
    print('Exact ELBO / log evidence / gap:', elbo, evidence, gap)
    print('Exact posterior mu/variance:', posterior_mu, posterior_variance)
    print('Exact-q ELBO / gap:', exact_q_elbo, exact_gap)
    rng = np.random.default_rng(130)
    z = rng.normal(size=4)
    decoder_means = updated[4]*z+updated[5]
    observation_samples = decoder_means+rng.normal(size=4)
    print('Prior z / decoder means / sampled observations:', z, decoder_means, observation_samples)
    print('All checks passed; no claim of learned image generation quality')


if __name__ == '__main__':
    main()
