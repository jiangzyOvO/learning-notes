"""Exact three-bit autoregressive distribution; prescribed conditional tables.

No neural image model or real dataset training. Nodes: root, 0, 1, 00, 01, 10, 11.
"""
import itertools
import numpy as np


def sigmoid(logits):
    return np.exp(-np.logaddexp(0., -logits))


def nll_and_grad(bits, logits):
    bits = np.asarray(bits, dtype=int)
    indices = [0, 1+bits[0], 3+2*bits[0]+bits[1]]
    selected = logits[indices]
    # Binary cross-entropy with logits, numerically stable.
    loss = np.sum(np.logaddexp(0., selected) - bits * selected)
    gradient = np.zeros_like(logits)
    gradient[indices] = sigmoid(selected) - bits
    return float(loss), gradient


def sample(logits, count, rng):
    bits = np.zeros((count, 3), dtype=int)
    probabilities = sigmoid(logits)
    bits[:, 0] = rng.random(count) < probabilities[0]
    bits[:, 1] = rng.random(count) < probabilities[1+bits[:, 0]]
    bits[:, 2] = rng.random(count) < probabilities[3+2*bits[:, 0]+bits[:, 1]]
    return bits


def main():
    prescribed = np.array([.8, .1, .5, .2, .7, .4, .25])
    logits = np.log(prescribed / (1-prescribed))
    sequences = np.array(list(itertools.product([0,1], repeat=3)))
    joint = np.array([np.exp(-nll_and_grad(bits, logits)[0]) for bits in sequences])
    np.testing.assert_allclose(joint.sum(), 1)
    loss, gradient = nll_and_grad([1,1,1], logits)
    np.testing.assert_allclose(np.exp(-loss), .8*.5*.25)
    numeric = np.zeros_like(logits)
    for i in range(len(logits)):
        original = logits[i]
        logits[i] = original + 1e-5
        plus = nll_and_grad([1,1,1], logits)[0]
        logits[i] = original - 1e-5
        minus = nll_and_grad([1,1,1], logits)[0]
        logits[i] = original
        numeric[i] = (plus-minus)/2e-5
    error = np.max(np.abs(numeric-gradient))
    assert error < 1e-7
    rng = np.random.default_rng(13)
    generated = sample(logits, 20_000, rng)
    ids = generated @ np.array([4,2,1])
    frequencies = np.bincount(ids, minlength=8) / len(ids)
    assert np.max(np.abs(frequencies-joint)) < .02
    print('Sequences / exact probabilities:', sequences.tolist(), joint)
    print('Probability(111) / NLL / gradient error:', np.exp(-loss), loss, error)
    print('Empirical frequencies from 20,000 exact-table samples:', frequencies)
    # Two-bit counterexample: p(first=1)=.6, p(second=1|0)=.99, p(second=1|1)=.6.
    two_bit_joint = np.array([.4*.01, .4*.99, .6*.4, .6*.6])
    assert np.argmax(two_bit_joint) == 1  # 01, probability .396
    greedy = [1,1]  # .6 then .6, probability .36
    print('Greedy:', greedy, 'probability .36; global best: 01 probability .396')
    # A fixed linear Gaussian decoder illustrates prior sampling, not trained VAE generation.
    z = rng.normal(size=4)
    decoder_mean = 2*z+1
    x_sample = decoder_mean + .5*rng.normal(size=4)
    print('Prior z / decoder means / observation samples:', z, decoder_mean, x_sample)
    print('Checks passed; no VAE encoder or ELBO optimization in this script')


if __name__ == '__main__':
    main()
