# 第三讲专题：批量梯度、L2 正则化与小批量 SGD

在[单样本梯度](03-gradient-descent-deep-review.md)的基础上，推导批量平均、L2 正则化与小批量 SGD。

## 1 多个样本共享参数

不是每张图片拥有一套自己的 W、b。当前批次所有样本使用同一组更新前参数，分别计算损失，合并为一个标量，再求梯度并更新一次。

本篇数据损失采用当前批次平均，实际批次大小记为 B：

$$
J_{\mathrm{data}}=\frac{1}{B}\sum_{i=1}^{B}L_i
$$

因此数据梯度也是单样本梯度的平均。若两个样本对一个权重的梯度分别为 2、-4，平均是 -1；学习率 0.1 时权重增加 0.1。梯度会相互抵消，反映当前批次目标的权衡，不保证每个样本都改善。

求和损失也能定义，但梯度与学习率尺度不同。不要损失用平均、梯度却用求和。复制整个批次一次，平均损失和平均数据梯度应不变，这是一个有用的实现核对。

## 2 批量 Softmax 梯度的矩阵形式

继续使用行样本 X 为 `(B,D)`，W 为 `(C,D)`，分数 S=X@W.T+b 为 `(B,C)`。P 表示概率矩阵，Y 表示 one-hot 目标矩阵：

$$
G=\frac{P-Y}{B},\qquad
\nabla_W J_{\mathrm{data}}=G^TX,\qquad
\nabla_b J_{\mathrm{data}}=\sum_{i=1}^{B}G_{i,:}
$$

G[i,c] 是平均数据损失对样本 i 的类别 c 分数的偏导。权重梯度元素为 sum_i(G[i,c]*X[i,j])，正是把每个样本的贡献累加。G 已除以 B，不要在 grad_W 或 grad_b 再除一次。

| 量 | 形状 | 含义 |
|---|---|---|
| X | `(B,D)` | 一批特征 |
| P、Y、G | `(B,C)` | 概率、目标、平均损失的分数梯度 |
| G.T @ X | `(C,D)` | 每个权重的数据梯度 |
| G.sum(axis=0) | `(C,)` | 每个类别偏置的数据梯度 |

代码无需真的构造 Y，直接在概率副本的正确类别位置减 1：

```python
grad_scores = probabilities.copy()
grad_scores[np.arange(B), y] -= 1
grad_scores /= B
grad_W = grad_scores.T @ X
grad_b = grad_scores.sum(axis=0)
```

## 3 正则化改变目标，学习率改变更新幅度

本篇采用明确的 L2 约定：

$$
J=J_{\mathrm{data}}+\frac{\lambda}{2}\sum_{c,j}W_{cj}^2
$$

λ 是非负正则化强度，通过验证集选择；本例不惩罚偏置。平方和越大，惩罚越大，目标倾向于限制过大的权重。它不能保证测试表现改善；过强可能欠拟合。输入特征尺度会影响这种偏好的含义。

对于单个权重，λ*w²/2 的导数为 λ*w，因此总梯度为：

$$
\nabla_W J=\nabla_W J_{\mathrm{data}}+\lambda W
$$

偏置梯度不变。若损失采用 λ*sum(W²)、没有 1/2，梯度必须为 2λW。代码必须和损失约定一致。

正则化不是“把所有梯度都缩小”：数据梯度与 λW 可同向、反向或抵消。它对权重施加趋近零的作用，但总更新仍由两项共同决定。

## 4 完整批量损失和梯度

下面为核心函数，假定数组形状正确、标签在有效范围内、输入为有限浮点数；实际 B 必须大于零：

```python
def loss_and_grad(W, b, X, y, reg=0.0):
    B = len(X)
    scores = X @ W.T + b
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    sum_exp = exp_scores.sum(axis=1, keepdims=True)
    probabilities = exp_scores / sum_exp

    losses = -shifted[np.arange(B), y] + np.log(sum_exp[:, 0])
    data_loss = losses.mean()
    reg_loss = 0.5 * reg * np.sum(W * W)

    grad_scores = probabilities.copy()
    grad_scores[np.arange(B), y] -= 1
    grad_scores /= B
    grad_W = grad_scores.T @ X + reg * W
    grad_b = grad_scores.sum(axis=0)
    return data_loss + reg_loss, data_loss, reg_loss, grad_W, grad_b
```

批量平均只作用于数据项，正则化按本目标约定加入一次。不要把整个总梯度再除以 B。返回值显式分开总损失、数据损失、正则化损失，便于判断变化原因。

## 5 普通 SGD 下的权重收缩

对普通 SGD 更新展开：

$$
W_{\mathrm{new}}=(1-\eta\lambda)W-\eta\nabla_W J_{\mathrm{data}}
$$

当 0<ηλ<1，第一部分缩小旧权重，第二部分仍由数据梯度决定。这说明这种约定下 L2 与普通 SGD 的乘性衰减等价；不要直接推广到 Adam 与 AdamW，后面会区分。

原例权重平方和为 7，λ=0.1 时正则化损失为 0.35。改变 λ 会改变优化目标；改变 η 会改变向目标移动的步长。两个超参数不是同一种作用。

## 6 为什么采用小批量

完整训练集若有 N 个样本，每次全部计算梯度成本很高。小批量 SGD 每次取一部分样本，用其平均梯度估计整体方向，在计算速度、内存和估计噪声之间权衡。

严格名称中单样本 SGD 的批量大小为 1；深度学习中 SGD 常也指 mini-batch SGD。随机性来自样本选择，不是随机猜测权重梯度。

固定参数、均匀随机抽样等条件下，小批量平均数据梯度的期望等于完整训练平均梯度。按轮打乱、不放回的后续批次与已有更新有关，不能机械声称每步条件期望都精确相等。单个小批量可能让全训练目标上升，不能只凭曲线抖动断定实现错误。

## 7 Epoch、step 和最后一批

每次参数更新是一个 step/iteration；遍历一轮训练集为一个 epoch。N=1000、batch size=128，保留最后一批时每轮 8 次更新，最后一批只有 104 个样本。数据损失和梯度必须除以实际 B=104。

训练结构：

```python
rng = np.random.default_rng(231)
for epoch in range(num_epochs):
    order = rng.permutation(len(X_train))
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        X_batch = X_train[indices]
        y_batch = y_train[indices]
        total, data, penalty, grad_W, grad_b = loss_and_grad(
            W, b, X_batch, y_batch, reg
        )
        W -= learning_rate * grad_W
        b -= learning_rate * grad_b
```

X 与 y 必须共享同一组索引，防止特征和标签错配。验证集用于选择超参数和判断泛化，不加入这些参数更新；最终测试集保留到评价阶段。分组相关或时序数据需要合适的划分，不能机械随机拆分。

## 8 运行核验与范围

[可运行代码](../experiments/deep_review/lecture03_batch_sgd.py)核对批量数据梯度等于逐样本梯度平均、批次整体复制不改变平均损失与总梯度、含 L2 的 W 和 b 梯度与中心差分一致。另用六个构造点演示五轮训练，每轮批量大小 [4,2]，共十次更新；训练目标降低不代表泛化能力已经核验。

```bash
python3 experiments/deep_review/lecture03_batch_sgd.py
```

[Momentum 与自适应优化器](03-optimizers-deep-review.md)：数据、损失和梯度已经明确，接着比较怎样利用当前及历史梯度更新参数。

继续阅读：[Momentum、Adam 与优化器状态](03-optimizers-deep-review.md)。

[第三讲主笔记](03-regularization-optimization.md) · [专题目录](00-deep-review-guide.md)
