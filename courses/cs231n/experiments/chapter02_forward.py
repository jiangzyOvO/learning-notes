"""第二章原创算例：只做前向计算，不训练模型。Python 3，无第三方依赖。"""
from math import exp, log, sqrt


def linear_scores(x, weights, biases):
    """每行权重对应一个类别：s_c = sum_j w_cj * x_j + b_c。"""
    if len(weights) != len(biases) or any(len(row) != len(x) for row in weights):
        raise ValueError("输入、权重与偏置形状不匹配")
    return [
        sum(w * value for w, value in zip(row, x)) + bias
        for row, bias in zip(weights, biases)
    ]


def probabilities_and_loss(scores, true_index):
    """减去最大值后计算 Softmax，并直接用 log-sum-exp 形式计算损失。"""
    largest = max(scores)
    shifted = [score - largest for score in scores]
    exp_scores = [exp(score) for score in shifted]
    denominator = sum(exp_scores)
    probabilities = [value / denominator for value in exp_scores]
    loss = log(denominator) - shifted[true_index]
    return probabilities, loss


if __name__ == "__main__":
    x = [2.0, 1.0]
    weights = [[1.0, 0.0], [-1.0, 2.0], [0.0, -1.0]]
    biases = [0.0, 0.0, 1.0]
    scores = linear_scores(x, weights, biases)
    probabilities, loss_a = probabilities_and_loss(scores, 0)
    _, loss_b = probabilities_and_loss(scores, 1)
    predicted_index = max(range(len(scores)), key=lambda i: scores[i])
    print("scores =", scores)
    print("predicted class =", ["A", "B", "C"][predicted_index])
    print("probabilities =", [round(p, 6) for p in probabilities])
    print("loss if true class is A =", round(loss_a, 6))
    print("loss if true class is B =", round(loss_b, 6))
    points = [(0.35, 0.05), (-0.5, 0.2), (0.1, 0.7), (-1.2, -0.8), (1.0, 1.1)]
    print("L2 distances to Q=(0,0) =", [round(sqrt(a*a+b*b), 6) for a, b in points])
    print("mean loss for true-class probabilities 0.8 and 0.6 =",
          round((-log(0.8)-log(0.6))/2, 6))
    shifted_probs, shifted_loss = probabilities_and_loss([1002.0, 1000.0, 1000.0], 0)
    print("adding 1000 preserves probabilities =",
          all(abs(a-b) < 1e-12 for a, b in zip(probabilities, shifted_probs)))
    print("adding 1000 preserves loss =", abs(loss_a-shifted_loss) < 1e-12)
