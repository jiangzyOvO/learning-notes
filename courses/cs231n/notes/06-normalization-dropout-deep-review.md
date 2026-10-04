# 第六讲专题：BatchNorm、LayerNorm 与 Dropout

比较 BatchNorm、LayerNorm 的统计轴和训练行为，再推导 Dropout 的缩放与反向传播。前置：[初始化与训练诊断](06-initialization-diagnostics-deep-review.md)。LayerNorm 同时为注意力模型提供基础。

## 1 归一化先问统计谁

输入预处理处理原始数据；归一化层处理网络中间的激活。BatchNorm 与 LayerNorm 的核心区别是均值、方差的统计轴，而非“哪个更高级”。归一化不把输出变成概率，也不改变输入输出形状。

对二维 X 为 `(N,D)`，一行一个样本、一列一个特征：BN 按列统计 N 个样本；本篇 LN 按行统计 D 个特征。CNN 的 NCHW BN 则按每个通道，在 N、H、W 上统计，不能只看二维示意。

## 2 BN 手算并保留可学习尺度

同一特征在两个样本中为 [1,3]：均值 μ=2，方差 v=((1-2)²+(3-2)²)/2=1。忽略 epsilon 时标准化为 [-1,1]；设置 gamma=2、beta=1，输出为 [-1,3]。

$$
\hat x=\frac{x-\mu}{\sqrt{v+\varepsilon}},\qquad y=\gamma\hat x+\beta
$$

epsilon 是小正数，零方差时避免除零。精确加入 epsilon 后 x_hat 方差为 v/(v+epsilon)，不严格等于 1；最终输出还受 gamma、beta 影响，所以 BN 不强迫后续响应始终零均值单位方差。

gamma、beta 是优化器学习的参数；批次均值、方差由当前输入计算；运行均值、方差是维护的状态缓冲。这三类量不能混为一组可学习权重。

## 3 同一张表的 BN 和 LN

```text
X = [[1,10],
     [3,30],
     [5,50]]
```

gamma=1、beta=0，忽略 epsilon 的微小影响：BN 每列得到 [-1.224745,0,1.224745]；LN 每行得到 [-1,1]。LN 是每个样本内部比较特征；它不依赖同批其他样本的数值。

将第三行第一项从 5 改为 105，第一行的 BN 第一特征从约 -1.224743 变为 -0.727599，第一行本身没改，统计量却改变了。第一行 LN 不变。对于 token 特征，LN 常在 `(N,T,D)` 的末轴 D 上做，每个 token 单独统计，而非默认把全部 token 混成一次统计。

## 4 BN 训练和评估为何不同

默认跟踪统计的 BN，训练前向用当前批次统计，并更新 running_mean、running_var；评估用已保存的运行统计，不更新它们。运行统计是多批次的移动估计，不是直接重新计算整个训练集的精确统计量。

以新批次权重 alpha=0.1：running_mean_new=0.9*old+0.1*batch_mean。这个“momentum”含义与优化器的历史动量不同，具体实现约定需查看文档。

脚本运行均值从 [0,0]、方差从 [1,1] 开始，处理上面的三样本一批后，均值为 [0.3,3]、运行方差为 [1.3,40.9]。训练标准化的方差除 N，更新运行方差时使用除 N-1 的估计，示范 PyTorch 的该项约定。仅一批的运行统计尚未充分适应数据，因此评估输出不应期待与训练输出相同。

PyTorch `track_running_stats=False` 是例外，评估仍用批次统计。BN 不一定在 batch size=1 时无统计可用：CNN 还包含 H、W；需看每个通道实际参与统计的数量与实现要求。

## 5 前向代码与形状

二维 BN 核心：

```python
mean = X.mean(axis=0, keepdims=True)
var = X.var(axis=0, keepdims=True)
x_hat = (X - mean) / np.sqrt(var + eps)
out = gamma * x_hat + beta
```

mean、var 为 `(1,D)`；gamma、beta 为 `(D,)`。二维 LN 改为 axis=1，mean、var 为 `(N,1)`，其余广播含义不变。CNN BN 的统计为 `axis=(0,2,3)`，`keepdims=True` 得 `(1,C,1,1)`，gamma、beta 按 `(1,C,1,1)` 广播。

归一化后接仿射变换具有每通道或每特征可学习自由度；归一化并非必然改善所有网络、也不能修复错误数据和反传。BN 的优化效果不能仅用“内部协变量偏移”概括。

## 6 反向不能把训练均值方差当常数

训练 BN 中，一个输入既直接进入标准化，又影响该列均值和方差。反向必须考虑这些分支，不能只乘 gamma/sqrt(var+eps)。LN 同样需考虑该样本内部的统计依赖。

对一组被共同归一化的 M 个元素，令 u=g*gamma，r=1/sqrt(var+eps)：

$$
\nabla_x L=\frac{r}{M}\left(Mu-\sum u-\hat x\sum(u\hat x)\right)
$$

这是逐组形式，求和沿归一化轴；本篇二维 BN 为轴 0，LN 为轴 1。gamma、beta 梯度仍沿它们共享的样本位置求和：dgamma=sum(g*x_hat,axis=0)、dbeta=sum(g,axis=0)。上游已有均值损失系数时，不在这里重复对整个损失平均。

第一遍不要求记住紧凑式，先理解统计量是计算图的一部分。评估 BN 若以固定运行统计计算输入导数，局部依赖则与训练不同；不等于 `.eval()` 不能求梯度。

## 7 Dropout 是随机掩码，不是删除参数

设保留概率 q，掩码 m 取 0 或 1，常用 inverted dropout：

$$
\tilde h=\frac{m h}{q}
$$

训练屏蔽部分激活，保留的乘 1/q；评估直接使用 h，不再乘 q。本篇 q 是保留概率，PyTorch Dropout 的参数 p 是丢弃概率，q=1-p。

h=[2,4]、q=0.5、某次掩码 [1,0]，训练输出 [4,0]，评估输出 [2,4]。掩码重新采样后保留位置可改变；不是永久删掉某个神经元或权重。

每个输入 h 的期望输出为 h，例如 2 有一半机会变 0、一半机会变 4。只保持该层单个激活的期望，不保证经过整个非线性网络的预测期望严格等于评估预测。

训练反向使用同一次前向掩码：grad_h=upstream*m/q。本例上游 [3,5] 得 [6,0]。反向不能重新采样。其本身没有可学习参数，比例也不应无条件设得越大越好。

## 8 模式、梯度记录、参数冻结是三个开关

| 操作 | 主要含义 | 不自动完成什么 |
|---|---|---|
| model.train() / model.eval() | 切换 BN、Dropout 等模块行为 | 不自动关闭梯度、不保证冻结参数 |
| torch.no_grad() | 暂停反向图记录 | 不切换 BN、Dropout 模式 |
| 参数 requires_grad=False | 不为该参数求梯度 | 不阻止训练模式 BN 更新运行统计 |

典型评估片段（展示 API 用法，NumPy 实验之外的接口说明）：

```python
model.eval()
with torch.no_grad():
    scores = model(X_val)
    predictions = scores.argmax(dim=1)
# 恢复训练前调用 model.train()
```

LN 通常在训练与评估都用当前输入内部统计，不采用 BN 的运行均值；但它所处网络中的 Dropout 等模块仍会改变行为。冻结骨干与评估模式的选择需明确，迁移学习下一篇继续。

## 核验与出处

[NumPy 代码](../experiments/deep_review/lecture06_normalization_dropout.py)核对二维 BN/LN 的轴、批次变化、运行统计读取、CNN BN 统计形状；输入、gamma、beta 的中心差分最大绝对误差 BN 约 1.61×10⁻¹⁰、LN 约 3.18×10⁻¹⁰。Dropout 枚举四种等概率掩码核对激活期望，核对固定掩码反向。不是完整归一化网络训练，不报告性能改善。

```bash
python3 experiments/deep_review/lecture06_normalization_dropout.py
```

- [第六讲课程主笔记](06-cnn-architectures.md)：沿用既有课件范围。
- [PyTorch BatchNorm2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm2d.html)
- [PyTorch LayerNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html)
- [PyTorch Dropout](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout.html)

下一部分进入残差架构与迁移，再进入第七讲 RNN / LSTM。

[专题目录](00-deep-review-guide.md) · [实验索引](../PROGRESS.md)

继续：[残差架构与迁移学习](06-residual-transfer-deep-review.md)。
