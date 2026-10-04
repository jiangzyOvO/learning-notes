# 第八讲专题：从手算到 QKV、掩码与反向传播

从单个查询的手算进入 QKV、因果掩码与反向传播。多头、位置、Transformer block 与 ViT 见[配套专题](08-transformer-vit-deep-review.md)。

## 1. 当前需要什么，就从其他位置读取什么

RNN/LSTM 沿时间更新状态。注意力提供按查询读取一组表示的办法：当前查询对每个位置打分，转换成权重，再把那些位置的内容加权汇总。注意力也可以与 RNN 一起使用，不是必须完全替换 RNN。

**Q/K 决定匹配权重，V 提供被汇总的内容。** Query（查询）、Key（匹配特征）、Value（内容）都是数值向量，不是人工写的文字问题与答案。点积是学习空间中的匹配分数，不保证等于人理解的语义相似性，也不是必然等于余弦相似度。

## 2. 先手算一个查询

指定一维 q=1，三个 key 为 [0,ln2,ln3]，三个 value 为 [10,20,40]。这里 d_k=1，缩放除数为 1。

| 位置 | key | 匹配分数 q×key | 分数的指数 | 归一化权重 | value |
|---|---|---|---|---|---|
| 1 | 0 | 0 | 1 | 1/6 | 10 |
| 2 | ln2 | ln2 | 2 | 2/6 | 20 |
| 3 | ln3 | ln3 | 3 | 3/6 | 40 |

```text
输出 z = (1/6)*10 + (2/6)*20 + (3/6)*40
       = 170/6 ≈ 28.333333
```

注意输出不是最高权重位置的 40，而是全部允许内容的加权汇总。权重是读取系数，不是 token 类别概率。把 q 改成 −1，指数变成 [1,1/2,1/3]，权重为 [6/11,3/11,2/11]，输出为 200/11≈18.181818。同一组内容，查询不同，读取结果不同。

## 3. 多查询与矩阵形状

```python
scores = Q @ K.T / np.sqrt(d_k)
A = softmax(scores, axis=1)
Z = A @ V
```

设查询数 N_q、被读取位置数 N_k，Q/K 特征维度 d_k，V 内容维度 d_v：

| 变量 | 形状 | 含义 |
|---|---|---|
| Q | (N_q,d_k) | 每行一个查询 |
| K | (N_k,d_k) | 每行一个匹配向量 |
| V | (N_k,d_v) | 每行一个内容向量 |
| scores、A | (N_q,N_k) | 每行查询对应各位置的分数、权重 |
| Z | (N_q,d_v) | 每个查询的读取结果 |

例如 Q=(2,4)、K=(3,4)、V=(3,5)，则分数和权重为 (2,3)，输出为 (2,5)。**查询数量决定输出行数，V 的维度决定输出列数。** K 与 V 按行对应同一个位置。Q/K 维度必须相等，V 维度可以不同。

**Softmax 沿 key 轴进行，每一行独立归一化，不要求每列和为 1。** 在本节二维代码里是 axis=1；若带批次，scores=(B,N_q,N_k)，通常沿 axis=-1。

稳定计算为：

```python
shifted = scores - scores.max(axis=1, keepdims=True)
exp = np.exp(shifted)
A = exp / exp.sum(axis=1, keepdims=True)
```

![打分、归一化与加权读取](../assets/beginner/attention-steps.png)

## 4. 为什么除以 sqrt(d_k)？

在分量独立、零均值、单位方差的简化分析下，点积是 d_k 项之和，方差约 d_k，标准差约 sqrt(d_k)。维度增大时，过大的分数可能使 Softmax 过度集中。除以 sqrt(d_k) 控制尺度。这些是分析假设，不表示所有实际特征满足它们。

## 5. QKV 从哪里来？

自注意力从同一个输入 X 的三组不同参数投影出 QKV：

```python
Q = X @ Wq
K = X @ Wk
V = X @ Wv
```

X=(N,D)，Wq/Wk=(D,d_k)，Wv=(D,d_v)。投影是学到的线性映射，不要求投影矩阵正交。来源相同不代表 Q=K=V，也不表示只看自己。各位置分别作为查询，读取允许位置。

交叉注意力的 Q 来自一组表示，K/V 来自另一组。例如生成文字的表示查询图片 patch 表示；来源长度可以不同，投影后的 Q/K 匹配维度仍必须相同。

## 6. 因果掩码与 padding

预测下一个 token 时使用右移输入：

```text
输入：START 我 爱 猫
目标：我    爱 猫 END
```

输入“我”的位置可以读取 START 和“我”，不能读取“爱”“猫”。允许当前输入位置，不是泄露当前预测目标。掩码表的行是查询，列是被读取位置：

```text
          START 我 爱 猫
START       1   0  0  0
我          1   1  0  0
爱          1   1  1  0
猫          1   1  1  1
```

1 表示允许。**先屏蔽分数，再做 Softmax**：

```python
scores = np.where(allowed, scores, -np.inf)
```

把不允许的分数设为 0 并不能屏蔽，因为 exp(0)=1。Softmax 后再清零会改变行和，若不重新处理就不是同一运算。本实现采用 allowed=True 表示允许，读其他接口必须核对布尔语义。

每行至少允许一个 key；全部屏蔽会产生未定义归一化，本教学代码明确抛出错误。Padding mask 防止读取补齐位置；padding 查询输出的处理与损失屏蔽需另外考虑。全模型的因果性还要求位置特征构造、输入特征和其他组件不泄露未来。

## 7. 反向传播仍然是矩阵求导与链式法则

上游给定 dZ=损失对 Z 的梯度。对 Z=A@V：

```python
dV = A.T @ dZ
dA = dZ @ V.T
```

一行 Softmax 满足 dA_j/dS_k=A_j*(delta_jk-A_k)。因此反向可压缩为：

```python
dS = A * (dA - (dA * A).sum(axis=1, keepdims=True))
```

不是只乘 A*(1-A)：Softmax 一行的不同分量彼此耦合。求和项包含这部分影响。固定掩码位置 A 为零，相应 dS 为零。

对 scores=Q@K.T/sqrt(d_k)：

```python
dQ = dS @ K / np.sqrt(d_k)
dK = dS.T @ Q / np.sqrt(d_k)
```

自注意力输入 X 有三个分支，所以：

```python
dWq = X.T @ dQ
dWk = X.T @ dK
dWv = X.T @ dV
dX = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T
```

这只包括当前注意力投影路径；完整 Transformer 还应加残差等其他路径。

## 8. 实现核验与下一节

[lecture08_attention.py](../experiments/deep_review/lecture08_attention.py)执行二维单头前向/反向，核对两个查询手算、行和、因果遮挡与全部屏蔽行拒绝。普通和因果 Q/K/V 梯度、X 与三个投影矩阵梯度通过中心差分，最大绝对误差约 1.17×10⁻⁹。没有训练模型、没有验证语言质量，也不是多头或完整 Transformer。

注意力权重只展示这个算子的加权方式，不自动证明模型的因果解释或语义理解正确。下一节连接多头、位置、残差、归一化、MLP 与 ViT。

来源：[官方第八讲课件](https://cs231n.stanford.edu/slides/2025/lecture_8.pdf)、[视频 P8](https://www.bilibili.com/video/BV1aXhJ64EmW/?p=8)。
