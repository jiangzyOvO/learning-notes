"""Reproduce the guided route's toy calculations; Python standard library only.

Run from the CS231n course directory: python3 experiments/guided_examples.py
No dataset, GPU or trained-model performance claims are involved.
"""
import math


def close(actual, expected, tol=1e-9):
    assert math.isclose(actual, expected, rel_tol=tol, abs_tol=tol), (actual, expected)


def matmul(a, b):
    return [[sum(x*y for x, y in zip(row, column)) for column in zip(*b)] for row in a]


def transpose(a):
    return [list(row) for row in zip(*a)]


def softmax(row):
    values = [math.exp(x-max(row)) for x in row]
    return [x/sum(values) for x in values]


def prediction_examples():
    x, z = [1., 2.], [4., 6.]
    l1 = sum(abs(a-b) for a, b in zip(x, z))
    l2 = math.sqrt(sum((a-b)**2 for a, b in zip(x, z)))
    close(l1, 7)
    close(l2, 5)
    p = softmax([2, 1])
    close(sum(p), 1)
    stable_ce = 1000 + math.log(1 + math.exp(-1000))
    close(stable_ce, 1000)
    print(f"01 距离 L1={l1:g}, L2={l2:g}; Softmax={p}; 稳定损失={stable_ce:g}")


def gradient_examples():
    loss = lambda w: (2*w-6)**2
    w, grad, lr = 1., -16., .1
    new_w = w-lr*grad
    close(new_w, 2.6)
    close(loss(new_w), .64)
    numerical = (loss(w+.01)-loss(w-.01))/.02
    close(numerical, grad)
    w1, w2, bias = 1-.05*(-12), 1-.05*(-6), .05*6
    close((2*w1+w2+bias-6)**2, 1.44)
    batch_grad = sum(2*(1*x-y)*x for x, y in [(1, 2), (2, 4)])/2
    close(batch_grad, -5)
    new_batch_w = 1-.1*batch_grad
    batch_loss = sum((new_batch_w*x-y)**2 for x, y in [(1, 2), (2, 4)])/2
    close(batch_loss, .625)
    close(((1-.05*(-8))*2*(1-.05*(-8))-4)**2, .0064)
    print(f"02 更新 w={new_w:g}, 损失={loss(new_w):g}; 批量更新损失={batch_loss:g}")


def affine_examples():
    x, w, target, bias = [[1., 2.], [3., 4.]], [[1.], [2.]], [[8.], [10.]], 1.

    def objective():
        pred = matmul(x, w)
        return sum((row[0]+bias-y[0])**2 for row, y in zip(pred, target))/len(x)

    pred = [[row[0]+bias] for row in matmul(x, w)]
    g = [[2*(row[0]-y[0])/len(x)] for row, y in zip(pred, target)]
    dw, dx, db = matmul(transpose(x), g), matmul(g, transpose(w)), sum(row[0] for row in g)
    assert pred == [[6.], [12.]] and dw == [[4.], [4.]] and dx == [[-2., -4.], [2., 4.]]
    close(db, 0)
    close(objective(), 4)
    eps = 1e-5
    errors = []
    for values, analytic in [(x, dx), (w, dw)]:
        for r, row in enumerate(values):
            for c, old in enumerate(row):
                values[r][c] = old+eps
                plus = objective()
                values[r][c] = old-eps
                minus = objective()
                values[r][c] = old
                errors.append(abs((plus-minus)/(2*eps)-analytic[r][c]))
    old = bias
    bias = old+eps
    plus = objective()
    bias = old-eps
    minus = objective()
    bias = old
    errors.append(abs((plus-minus)/(2*eps)-db))
    assert max(errors) < 1e-7
    print(f"03 矩阵 dW={dw}, db={db:g}, dX={dx}; 中心差分最大误差={max(errors):.3g}")


def image_examples():
    x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    kernel = [[1, 0], [0, 1]]
    conv = [[sum(x[r+i][c+j]*kernel[i][j] for i in range(2) for j in range(2))
             for c in range(2)] for r in range(2)]
    assert conv == [[6, 8], [12, 14]]
    x4 = [list(range(1+4*r, 5+4*r)) for r in range(4)]
    pool = [[max(x4[r+i][c+j] for i in range(2) for j in range(2))
             for c in (0, 2)] for r in (0, 2)]
    assert pool == [[6, 8], [14, 16]]
    sizes = [math.floor((32+2*p-3)/s)+1 for p, s in [(0, 1), (1, 1), (1, 2)]]
    assert sizes == [30, 32, 16]
    params = 16*(3*3*3+1)+32*(16*3*3+1)+(32*10+10)
    assert params == 5418
    print(f"04–05 卷积={conv}; 池化={pool}; 输出尺寸={sizes}; CNN 参数={params}")


def training_examples():
    data = [1., 2., 3.]
    mean = sum(data)/len(data)
    variance = sum((x-mean)**2 for x in data)/len(data)
    normalized = [(x-mean)/math.sqrt(variance) for x in data]  # 为手算忽略 epsilon
    close(sum(normalized), 0)
    dropout = [h*m/.5 for h, m in zip([2, 4, 6, 8], [1, 0, 1, 0])]
    assert dropout == [4, 0, 12, 0]
    print(f"06 He 标准差={math.sqrt(2/27):.6f}; 标准化={normalized}; Dropout={dropout}")


def attention_examples():
    q = k = [[1., 0.], [0., 1.]]
    v = [[2., 0.], [0., 4.]]
    scores = [[x/math.sqrt(2) for x in row] for row in matmul(q, transpose(k))]
    weights = [softmax(row) for row in scores]
    output = matmul(weights, v)
    for row in weights:
        close(sum(row), 1)
    close(output[0][0], 1.3395230986533138)
    close(output[0][1], 1.3209538026933725)
    assert (32//8)**2 == 16 and 3*8*8 == 192
    print(f"07 注意力权重={weights}; 输出={output}")


def task_examples():
    iou = 60/(100+100-60)
    close(iou, 3/7)
    x, z, t, dt = 2., 10., .5, -.1
    xt, velocity = (1-t)*x+t*z, z-x
    close(xt+dt*velocity, 5.2)
    close(.5*.8*.25, .1)
    overall = (900*.99+100*.60)/1000
    close(overall, .951)
    print(f"08–10 IoU={iou:.6f}; 已知路径的一步={xt+dt*velocity:g}; 总体准确率={overall:.1%}")


if __name__ == "__main__":
    for example in (prediction_examples, gradient_examples, affine_examples,
                    image_examples, training_examples, attention_examples, task_examples):
        example()
    print("全部教学数值核验通过；未进行真实数据集训练。")
