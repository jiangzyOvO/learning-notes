# 第三讲专题：Momentum、Adam 与优化器状态

比较 Momentum、Adam 与 AdamW 的更新规则。数字例子使用预设梯度，便于观察优化器状态。前置：[批量梯度、L2 与 SGD](03-batch-regularization-deep-review.md)。

## 1 优化器接收梯度，而不是替模型计算梯度

前向、损失和反传产生梯度 g；优化器结合当前梯度、历史状态、学习率与配置更新参数。换优化器不会自动改变模型结构或标签，也不替代正确的损失定义。

普通 SGD：parameter_new=parameter-lr*gradient，没有本篇所说的历史动量或矩统计状态。Momentum 与 Adam 增加与每个参数同形状的状态，必须跨更新保留。

## 2 Momentum 累积方向

本篇采用非归一化累积约定，初始 u=0：

$$
u_t=\mu u_{t-1}+g_t,\qquad \theta_t=\theta_{t-1}-\eta_t u_t
$$

μ 为历史保留系数。预设连续梯度为 [2,2,2]、μ=0.9、η=0.1：

| 步 | 当前梯度 | u | 从参数 0 开始的更新后参数 |
|---|---|---|---|
| 1 | 2 | 2 | -0.2 |
| 2 | 2 | 3.8 | -0.58 |
| 3 | 2 | 5.42 | -1.122 |

一致方向被累积；如果梯度是 [2,-2,2]，u 则为 [2,-0.2,1.82]，正负部分抵消。不是所有情况下 Momentum 都减少抖动或更快收敛，过冲与学习率仍需关注。

有的教材使用 u=μu+(1-μ)g，或把学习率放在状态定义中，必须连同完整参数更新式阅读。这里的 u 不是后面 Adam 的二阶矩 v。

```python
velocity = np.zeros_like(W)  # 在训练循环外初始化
for step in range(num_steps):
    grad_W = compute_current_gradient(W)  # 伪代码：从当前参数计算
    velocity = 0.9 * velocity + grad_W
    W = W - learning_rate * velocity
```

若每一步都把 velocity 清零，就失去历史累积。

## 3 从 AdaGrad 与 RMSProp 理解尺度记录

以下为每个参数各自的逐元素计算，不是矩阵求逆。

- AdaGrad 累加 g²：r←r+g²，更新用 g/(sqrt(r)+epsilon)。固定 η、恒定梯度 2 时，前三步更新幅度约 η、η/√2、η/√3；旧记录不断累积。
- RMSProp 衰减平均 g²：r←ρr+(1-ρ)g²，再用同样的尺度归一化形式。旧记录逐渐淡出，不是固定窗口内的普通平均。

它们为各参数提供不同缩放，但仍保留一个全局学习率。AdaGrad 在原课件附录，作为理解 RMSProp 的补充入口。

## 4 Adam 同时记录方向与尺度

初始 m=v=0：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

m 是梯度的衰减平均，v 是梯度平方的衰减平均。v 是二阶原始矩估计，不是方差、二阶导数或 Hessian；g²、开方、除法都逐元素进行。两个状态与参数同形状。

## 5 偏差修正来自零初始化

第一步 g=2，β1=0.9、β2=0.999：m=0.2、v=0.004。它们较小部分来自状态初始化为零。

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad \hat v_t=\frac{v_t}{1-\beta_2^t}
$$

第一步修正后 m_hat=2、v_hat=4。若第二步梯度仍为 2，m=0.38、v=0.007996，修正后仍为 2、4。

原因：m_t=(1-β1)*sum(k=1..t, β1^(t-k)*g_k)，其中梯度权重总和为 1-β1^t。除以这个总和，消除零初始化造成的权重不足。v 同理。统计意义下仍需相应假设，不是消除所有噪声或保证无偏估计真实最优方向。

## 6 Adam 完整更新与代码

$$
\theta_t=\theta_{t-1}-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}
$$

上面的恒定梯度例子忽略很小 epsilon 时，每步参数减少 η。Adam 不会总是每步走相同距离；变动梯度使历史方向、平方统计和比例改变。

```python
# 循环外；每个参数分别保留 m、v
m = np.zeros_like(W)
v = np.zeros_like(W)
for t in range(1, num_steps + 1):
    g = compute_current_gradient(W)  # 伪代码
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g * g
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    W -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
```

t 从 1 开始，按参数更新次数计，不是 epoch 数。W、b 等各自保留状态，同一次全模型更新使用一致 t；不能在每更新一个矩阵时额外增加全局 t。epsilon 防止零分母，但不能修复所有非有限输入。恢复训练时应保留优化器状态与步数。

## 7 Adam 加 L2 与 AdamW 的区别

Adam 加 L2：将 g_data+λW 送进 m、v，因此惩罚项参与方向和尺度统计。AdamW：统计用数据梯度，再独立对旧权重施加乘性衰减。

$$
W_t=(1-\eta_t\lambda)W_{t-1}-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}
$$

使用零历史状态、零数据梯度、W=[2,-3]、η=0.1、λ=0.1：

| 方法 | 更新后权重，忽略 epsilon 的微小影响 |
|---|---|
| Adam 无惩罚 | [2,-3] |
| Adam 梯度中加入 L2 | [1.9,-2.9] |
| AdamW 独立衰减 | [1.98,-2.97] |

这是零历史第一步，不能推广为之后零数据梯度时总不更新，因为已有历史 m 还可能产生移动。例子只证明更新不同，不证明哪种总更好。偏置是否衰减需按参数组约定；本系列默认偏置不惩罚。

普通无动量 SGD 中 L2 与对应乘性衰减可等价，但自适应缩放破坏了该直接等价关系。见[解耦权重衰减论文](https://arxiv.org/abs/1711.05101)。

## 8 学习率调度与优化器是两件事

优化器决定历史信息怎样参与更新；调度决定全局 η 随步骤怎样变化。Adam 的自适应缩放不意味着不需要学习率或调度。

- 阶梯衰减：到指定步骤乘上小于 1 的系数。
- 线性衰减：从较高值逐步线性下降。
- 余弦衰减：平滑降低，例 η_max=0.1、η_min=0.001、T=100 时，t=0、50、100 分别为 0.1、0.0505、0.001。
- Warmup：开始阶段先小后大，再接后续调度；不是完整训练一直升大学习率。

这些不是保证成功的固定配方。损失、梯度、学习率、验证表现需要结合判断；不要仅看训练损失给优化器排绝对名次。

## 9 两个扩展入口

L1 正则化惩罚绝对值之和，在非零处梯度为符号函数，零点需次梯度等处理；常鼓励稀疏，但普通有限次梯度更新不保证精确零。Elastic Net 结合 L1 和 L2，具体约定见主笔记。

二阶优化关注曲率。Hessian 是参数之间的二阶偏导矩阵，P 个参数对应 `(P,P)`；Newton 方向通过解 H*Δθ=-g 得到。存储和求解完整稠密矩阵成本高，非凸问题还需处理非正定曲率。Adam 的梯度平方统计不是二阶优化；BFGS、L-BFGS 等附录内容作为进一步阅读。

## 10 运行记录与来源

[完整代码](../experiments/deep_review/lecture03_optimizers.py)复算 Momentum 一致与交替梯度、Adam 前两步修正、零历史零数据梯度下 Adam/L2 与 AdamW 不同、余弦调度端点。预设梯度没有来自训练模型，不用参数移动量声称收敛或泛化更好。

```bash
python3 experiments/deep_review/lecture03_optimizers.py
```

- [第三讲主笔记](03-regularization-optimization.md)：原课程课件与视频采样定位。
- [Adam 原始论文](https://arxiv.org/abs/1412.6980)
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)

第三讲现在连通目标、梯度、正则化、批量采样、优化器与调度。

[专题目录](00-deep-review-guide.md) · [实验索引](../PROGRESS.md)
