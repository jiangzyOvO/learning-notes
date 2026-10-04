# 03 从一个神经元到矩阵反传

> 本单元贯穿“前向 → 损失 → 反向 → 更新”。先手算一个标量网络，再解释矩阵里的转置。平方误差用于教学，不替代分类交叉熵。

## 1. 为什么不能只把线性层叠很多次？

把两层仿射计算接起来：

$$y=w_2(w_1x+b_1)+b_2=(w_2w_1)x+(w_2b_1+b_2).$$

它仍然可以合并为一个“权重乘输入，再加偏置”的式子。要表达更复杂的关系，需要非线性。

例如希望预测 `y=|x|`：在 x 为 −1、0、1 时，目标为 1、0、1。一条 `wx+b` 的直线不能同时经过这三个点。

ReLU 定义为：

$$\operatorname{ReLU}(a)=\max(0,a).$$

可以构造 `h₁=ReLU(x)`、`h₂=ReLU(−x)`，再计算 `y=h₁+h₂`，恰好得到 `|x|`。这是手工构造的演示，不是训练成果。

**线性层组合已有特征，非线性改变响应方式，下一层再继续组合。**

## 2. 手算一次完整的反向传播

设 x=2、目标 y=4，没有偏置，初始 w₁=w₂=1：

$$a=w_1x,\quad h=\operatorname{ReLU}(a),\quad \hat y=w_2h,\quad L=(\hat y-y)^2.$$

前向逐步得到：`a=2 → h=2 → ŷ=2 → L=4`。

反向从最后开始，每一步都遵循“上游梯度 × 当前运算的局部导数”：

| 需要求的梯度 | 计算 | 结果 |
|---|---|---|
| 对预测 ŷ | 2(ŷ−y) | −4 |
| 对 w₂ | (−4)×h | −8 |
| 对 h | (−4)×w₂ | −4 |
| 对 a | (−4)×ReLU'(a) | −4 |
| 对 w₁ | (−4)×x | −8 |

这里 a>0，ReLU 导数为 1；a<0 时为 0；a=0 处不可导，代码通常约定取 0。

所有梯度算完，以学习率 0.05 同时更新：w₁=w₂=1.4。重新前向计算：a=h=2.8，ŷ=3.92，L=0.0064。

> **反向传播负责算梯度，梯度下降负责更新。** 前向时的中间值需要保留，因为反向会使用它们。

## 3. 一层矩阵计算：每一行是一个样本

现在只看线性层 `Y=XW+b`。Y 表示输出，真实目标另外用 T 表示，避免把两者混在一起。

$$X=\begin{bmatrix}1&2\\3&4\end{bmatrix},\quad
W=\begin{bmatrix}1\\2\end{bmatrix},\quad b=1.$$

两行输入分别得到：

$$Y=\begin{bmatrix}1\times1+2\times2+1\\3\times1+4\times2+1\end{bmatrix}
=\begin{bmatrix}6\\12\end{bmatrix}.$$

设真实目标 `T=[[8],[10]]`，损失是两个样本平方误差的平均：

$$L=\frac{(6-8)^2+(12-10)^2}{2}=4.$$

设 `G=∂L/∂Y`，那么：

$$G=\frac{2(Y-T)}{N}=\begin{bmatrix}-2\\2\end{bmatrix},\qquad N=2.$$

本例恰好 N=2，系数 2/N=1；不能因此把一般公式里的系数省掉。

## 4. 为什么权重梯度是 X.T @ G？

先看第一个权重 w₁：它在样本 1 中乘输入 1，在样本 2 中乘输入 3。

$$\frac{\partial L}{\partial w_1}=1\times(-2)+3\times2=4.$$

第二个权重：

$$\frac{\partial L}{\partial w_2}=2\times(-2)+4\times2=4.$$

转置后的 X 把同一个特征在不同样本中的数放在一行：

$$X^T=\begin{bmatrix}1&3\\2&4\end{bmatrix}.$$

因此，上面两次“乘起来再沿样本求和”可以合并为：

$$\boxed{\frac{\partial L}{\partial W}=X^TG=\begin{bmatrix}4\\4\end{bmatrix}.}$$

矩阵乘法只是一次完成所有元素的链式法则与贡献累加；转置不是背诵出来的装饰。

## 5. 为什么偏置求和？为什么 dX 乘 W.T？

b 被加到每一行，所以对所有样本贡献求和：

$$\frac{\partial L}{\partial b}=(-2)+2=0.$$

梯度为 0 是当前点上两个贡献恰好抵消，不代表偏置没有作用。

对输入求导时，每个输出沿连接权重把梯度传回来：

$$\boxed{\frac{\partial L}{\partial X}=GW^T
=\begin{bmatrix}-2&-4\\2&4\end{bmatrix}.}$$

在多层网络中，这个 dX 通常继续传给前一层；不代表要修改原始图片。

| 量 | 一般形状 | NumPy 写法 |
|---|---|---|
| X、W、Y | `(N,D)`、`(D,M)`、`(N,M)` | `Y = X @ W + b` |
| G | `(N,M)` | 上游给出的 `dL/dY` |
| dW | `(D,M)` | `X.T @ G` |
| db | `(M,)` | `G.sum(axis=0)` |
| dX | `(N,D)` | `G @ W.T` |

**G 已经包含损失的平均系数时，dW、db 不要再除一次 N。**

## 6. 两层网络的代码怎样连接？

下面是完整可运行的小型前向、反向与一次更新。初始参数是为核验算例固定的教学数值，不是推荐的通用初始化方案。

```python
import numpy as np

X = np.array([[1., 2.], [3., 4.]])
y = np.array([[8.], [10.]])
W1 = np.eye(2)
b1 = np.zeros(2)
W2 = np.array([[1.], [2.]])
b2 = np.array([1.])
lr = 0.001
N = X.shape[0]

# 前向；本例损失对样本平均，对输出维度求和。
a = X @ W1 + b1
h = np.maximum(a, 0)
y_pred = h @ W2 + b2
loss = ((y_pred - y) ** 2).sum() / N

# 反向；所有梯度先用旧参数算完。
g = 2 * (y_pred - y) / N
dW2 = h.T @ g
db2 = g.sum(axis=0)
dh = g @ W2.T
da = dh * (a > 0)
dW1 = X.T @ da
db1 = da.sum(axis=0)

W1 -= lr * dW1
b1 -= lr * db1
W2 -= lr * dW2
b2 -= lr * db2
new_pred = np.maximum(X @ W1 + b1, 0) @ W2 + b2
new_loss = ((new_pred - y) ** 2).sum() / N
print(loss, new_loss)  # 4.0，约 3.8250
```

`a > 0` 是布尔掩码，与梯度逐元素相乘。若 a=`[2,-3,1]`，dh=`[4,5,6]`，则 da=`[4,0,6]`。

分类时可把输出损失换成 Softmax 交叉熵，反向起点随之改变；线性层的三条矩阵梯度公式仍然适用。

另一个 NumPy 示例用的是 **Sigmoid + 平方误差和**，因此隐藏层导数为 `h*(1-h)`，且损失没有平均因子。对应解释见[NumPy 网络逐行讲解](../notes/04-numpy-network-walkthrough.md)，不要与本例的 ReLU 混用。

## 7. 怎样检查梯度有没有算错？

用中心差分近似导数：

$$\frac{\partial L}{\partial w}\approx
\frac{L(w+\varepsilon)-L(w-\varepsilon)}{2\varepsilon}.$$

对 `L=(2w−6)²`，w=1，ε=0.01，两个损失为 15.8404 与 16.1604，结果为 −16，与解析导数一致。

检查矩阵时，一次只扰动一个元素，其余参数固定，之后恢复原值。它适合小规模调试，逐参数计算成本高，不适合作为大网络的常规训练方法。ε 太小受舍入影响；ReLU 折点附近也要谨慎比较。

## 复习时记住

**共享参数的梯度需要把各处贡献相加；梯度形状与被求导的数组一致。** 下一单元会看到卷积如何把同一组权重共享到不同位置。

[上一单元](02-learning-and-gradients.md) · [下一单元：卷积](04-convolution.md) · [课程第四章](../notes/04-neural-networks-backprop.md) · [路线目录](README.md)
