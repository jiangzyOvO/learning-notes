"""Lecture 3 numerical illustrations, not an assignment implementation.

Standard-library arithmetic only; no trained model or hardware benchmark.
"""
from math import exp, log, sqrt, sin, cos, pi, isclose


def softmax(scores):
    offset = max(scores)
    weights = [exp(x - offset) for x in scores]
    total = sum(weights)
    return [x / total for x in weights]


def rotate(v, angle):
    x, y = v
    return [x * cos(angle) - y * sin(angle),
            x * sin(angle) + y * cos(angle)]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def main():
    p = softmax([log(2), 0.0])
    assert isclose(p[0], 2 / 3)
    values = [[1.0, 0.0], [0.0, 3.0]]
    output = [sum(p[i] * values[i][j] for i in range(2)) for j in range(2)]
    assert isclose(output[0], 2 / 3) and isclose(output[1], 1.0)
    print('Attention weights:', p, 'weighted values:', output)
    x = [3.0, 4.0]
    rms = sqrt(sum(v * v for v in x) / len(x))
    normalized = [v / rms for v in x]  # gamma=1, epsilon=0 in this hand example
    assert isclose(sum(v * v for v in normalized) / 2, 1.0)
    print('RMS:', rms, 'RMS-normalized:', normalized)
    mean = sum(x) / len(x)
    std = sqrt(sum((v - mean) ** 2 for v in x) / len(x))
    layernorm = [(v - mean) / std for v in x]
    assert layernorm == [-1.0, 1.0]
    print('Layer-normalized:', layernorm)
    q, k = [1.0, 0.0], [1.0, 0.0]
    angle = pi / 2
    score = dot(rotate(q, 0), rotate(k, angle))
    shifted_score = dot(rotate(q, angle), rotate(k, 2 * angle))
    assert isclose(score, shifted_score, abs_tol=1e-12)
    assert isclose(dot(rotate(x, angle), rotate(x, angle)), dot(x, x))
    print('RoPE joint position shift:', score, shifted_score)
    d = 12
    ordinary_width, gated_width = 4 * d, 8 * d // 3
    ordinary_params = 2 * d * ordinary_width
    gated_params = 3 * d * gated_width
    assert ordinary_params == gated_params == 1152
    print('FFN counts, ordinary/gated:', ordinary_params, gated_params)
    logits = [1000.0, 1001.0]
    probs = softmax(logits)
    shifted = softmax([v - 500 for v in logits])
    assert all(isclose(a, b) for a, b in zip(probs, shifted))
    print('Stable softmax:', probs)
    print('All numerical checks passed; no model was trained.')


if __name__ == '__main__':
    main()
