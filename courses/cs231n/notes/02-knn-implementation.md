# 第二讲专题 k-NN 的原理和完整实现

本篇对应第二讲前半部分：图像分类、距离、近邻投票、验证与复杂度。课堂主线见官方 2025 第二讲 PDF 第 20–46 页附近，具体页码按文件顺序；手算、代码和 40 点数据均为补充说明，不是课程数据集成绩。

前置：[第一讲数据布局](01-introduction.md)。主笔记：[图像分类与线性分类器](02-image-classification.md)。线性分类与 Softmax 见后续专题。

## 1 数据怎样放进分类器

训练特征为 `X_train`，形状 `(N_train,D)`；对应标签为 `y_train`，形状 `(N_train,)`。测试输入 X 为 `(N_test,D)`。D 是每个样本的特征数量，训练和测试必须使用相同特征顺序、布局与预处理。

本例每个样本只有两个虚构特征，A=0、B=1：

```python
import numpy as np

X_train = np.array([[1., 0.], [0., 2.], [2., 2.], [4., 0.], [-3., 0.]])
y_train = np.array([0, 1, 1, 0, 1])
X_test = np.array([[0., 0.], [4., 1.]])
```

这里 0 和 1 是类别编号，不是样本到某个对象的距离，也不表示 B 比 A 更大。

## 2 为什么先讲一种不做梯度下降的方法

k-NN 保存训练样本与标签。预测时查询这些已知样本，在选定度量下找最近的 k 个，再等权投票。本方法不训练一组线性层权重，也不需要反向传播。

`train` 在这种方法里主要是保存数据；其他模型的 `train` 可能包含参数优化。同一个词不能直接推出相同实现。完整数据读取、类型转换与存储仍有成本。

## 3 两种距离逐项展开

对 x=[1,2]、z=[4,6]：

$$
d_1(x,z)=|1-4|+|2-6|=7
$$

$$
d_2(x,z)=\sqrt{(1-4)^2+(2-6)^2}=5
$$

L1 用绝对值防止正负差抵消，L2 用平方汇总差异，再开方恢复相应的尺度。比较 L2 远近时可比较平方距离，因为开方在非负数上保持顺序。

度量会改变近邻选择。查询为 [0,0]，候选为 [3,0] 和 [2,2] 时：L1 距离分别为 3、4；L2 分别为 3、√8。前者选第一点，后者选第二点。

输入某一维放大也可能改变选择，所以特征尺度是模型的一部分。若预处理需要从数据拟合均值、标准差，应只在当前训练部分拟合；固定除以 255 不需要估计数据统计量。

## 4 一张测试图怎样得到预测

对第一个查询 [0,0]，按训练数组顺序列出：

| 数组索引 | 特征 | 标签 | L2 距离 |
|---|---|---|---|
| 0 | [1,0] | A | 1 |
| 1 | [0,2] | B | 2 |
| 2 | [2,2] | B | √8≈2.828 |
| 3 | [4,0] | A | 4 |
| 4 | [−3,0] | B | 3 |

从近到远的索引是 `[0,1,2,4,3]`。k=1 只参考 A，预测 A；k=3 参考 A、B、B，预测 B。没有给查询的真实标签，因此不能直接判断哪次正确。

另一个查询 [4,1] 的 k=1、k=3 预测均为 A，可运行脚本核对。

## 5 距离矩阵的每个元素是什么

$$
D_{ij}=d_2(X_i,X_{train,j})
$$

行表示测试样本，列表示训练样本。本例 D 的形状是 `(2,5)`，平方距离为：

$$
D^{\odot2}=\begin{bmatrix}1&4&8&16&9\\10&17&5&1&50\end{bmatrix}
$$

这里 D 的逐元素平方不是矩阵乘法 D@D。对每个元素开平方，才得到实际 L2 距离。

热图亮行表示某个测试样本距离很多训练样本较远；亮列对应某个训练样本距离很多测试样本较远。不能仅由热图断定原图一定更亮或是哪一种图像特征导致。

## 6 从双循环到无循环

双循环直接对应元素定义：

```python
# 片段：X 和 self.X_train 已是浮点二维数组。
dists = np.zeros((len(X), len(self.X_train)))
for i in range(len(X)):
    for j in range(len(self.X_train)):
        difference = X[i] - self.X_train[j]
        dists[i, j] = np.sqrt(np.sum(difference * difference))
```

无循环利用平方展开，先看一个元素：

$$
\sum_{m=1}^{D}(x_m-z_m)^2 = \sum_{m=1}^{D}x_m^2 + \sum_{m=1}^{D}z_m^2 - 2\sum_{m=1}^{D}x_mz_m
$$

即：

$$
\lVert x-z\rVert_2^2 = \lVert x\rVert_2^2 + \lVert z\rVert_2^2 - 2x^Tz
$$

**纯文本对照（不依赖公式渲染）：**

```text
平方距离 = x 的各元素平方和 + z 的各元素平方和
           - 2 × x 与 z 对应元素乘积的总和

例如 x = [1, 2]，z = [4, 6]：
平方距离 = (1² + 2²) + (4² + 6²) - 2 × (1×4 + 2×6)
         = 5 + 52 - 32
         = 25
L2 距离 = √25 = 5
```

于是整张表可以用矩阵运算计算：

```python
def compute_distances_no_loops(self, X):
    X = np.asarray(X, dtype=np.float64)
    train = np.asarray(self.X_train, dtype=np.float64)
    test_sq = np.sum(X * X, axis=1, keepdims=True)
    train_sq = np.sum(train * train, axis=1)
    cross = X @ train.T
    squared = test_sq + train_sq - 2 * cross
    return np.sqrt(np.maximum(squared, 0.0))
```

| 中间量 | 形状 | 元素含义 |
|---|---|---|
| test_sq | `(N_test,1)` | 每个测试向量的平方和 |
| train_sq | `(N_train,)` | 每个训练向量的平方和 |
| cross | `(N_test,N_train)` | 每对测试、训练向量的点积 |
| squared、dists | `(N_test,N_train)` | 每一对的平方距离、距离 |

`axis=1` 沿特征求和；`keepdims=True` 为测试平方和保留列形状。训练平方和 `(N_train,)` 从末轴对齐，广播时可按 `(1,N_train)` 参与运算。如果也保留为 `(N_train,1)`，通常不是需要的方向，可以再转置成一行。

`maximum` 截掉浮点舍入造成的小负值。它只解决开平方前的小负数，不能完全修复大数相减的精度损失。整数要在平方、相减之前转换；大规模数据还需要考虑整张距离表的内存。

向量化仍需完成这些数学运算，只是把计算交给底层数组实现，不表示时间复杂度变成常数。

## 7 从距离得到标签的代码

下面是教学实现中的方法，假定 train 已保存对齐的数据和非负整数标签。外层保留查询循环，只有距离计算要求无显式循环。

```python
def predict_labels(self, dists, k=1):
    predictions = np.empty(len(dists), dtype=self.y_train.dtype)
    for i in range(len(dists)):
        nearest_indices = np.argsort(dists[i], kind="stable")[:k]
        nearest_labels = self.y_train[nearest_indices]
        labels, counts = np.unique(nearest_labels, return_counts=True)
        predictions[i] = labels[np.argmax(counts)]
    return predictions
```

| 操作 | 含义 |
|---|---|
| `argsort` | 返回排序后的原数组索引，不是排好序的距离 |
| `[:k]` | 取前 k 个邻居 |
| `y_train[indices]` | 从索引找到这些邻居的标签 |
| `unique(...,return_counts=True)` | 统计各标签出现次数 |
| `argmax(counts)` | 找票数最多的标签所在位置 |

距离相同时，稳定排序保留训练数组顺序。票数平局时，`unique` 排序标签，`argmax` 取第一个最大值，因此本实现选编号较小的标签。这是人为约定，不表示较小标签更重要。

多分类即使 k 为奇数也可能平票，例如 k=3、三个类别各一票。完整脚本还检查了 k 的合法范围；上面片段省略输入检查以突出核心计算。

## 8 为什么不能只根据训练准确率选 k

1-NN 允许查询匹配自身、且没有冲突重复样本等问题时，可得到零训练误差。它只是能记住样本，不保证新数据正确率最高。

小 k 更依赖局部邻居，大 k 汇总更宽范围。在常见直觉中可看作平滑程度的取舍，但没有某个 k 对所有数据都最好。

朴素单查询扫描距离为 O(N_train D)，完整排序另有 O(N_train log N_train)；数据存储 O(N_train D)，批量距离表 O(N_test N_train)。近邻检索的其他算法不一定采用完全相同的流程。

决策边界一般不能用一个线性分数差解释。某些简单数据配置可以出现直线边界，不等于所有 k-NN 分类边界都线性。

## 9 五折验证怎样避免偷看答案

将开发用数据划分成五折，不包含最终测试集。每次一折验证、四折训练，让每个样本作为验证样本一次。

| 轮次 | 训练折 | 验证折 |
|---|---|---|
| 1 | 1、2、3、4 | 0 |
| 2 | 0、2、3、4 | 1 |
| 3 | 0、1、3、4 | 2 |
| 4 | 0、1、2、4 | 3 |
| 5 | 0、1、2、3 | 4 |

**折数与邻居数量是两个不同的 k。** 对每个候选邻居数保存五个准确率，比较平均值等验证信息，再决定配置。

核心片段使用共享样本索引，保证特征和标签不会打乱对应关系：

```python
# 片段：X、y 是开发用数据，KNearestNeighbor 已定义。
rng = np.random.default_rng(231)
folds = np.array_split(rng.permutation(len(X)), 5)
k_choices = [1, 3, 5]
k_to_accuracies = {k: [] for k in k_choices}

for held_out in range(5):
    val_idx = folds[held_out]
    train_idx = np.concatenate([
        fold for i, fold in enumerate(folds) if i != held_out
    ])
    model = KNearestNeighbor()
    model.train(X[train_idx], y[train_idx])
    dists = model.compute_distances_no_loops(X[val_idx])
    for k in k_choices:
        pred = model.predict_labels(dists, k)
        accuracy = np.mean(pred == y[val_idx])
        k_to_accuracies[k].append(float(accuracy))
```

同一折的距离不依赖 k，可计算一次后复用。每个 k 都必须不超过该折训练样本数。若折大小不等，简单平均给各折相同权重；希望每个验证样本相同权重时需按折大小汇总。本脚本 40 个点分为五折，每折恰好 8 个。

时间序列、同一主体、多次拍摄或近重复图像不能机械随机混折，需要按相关结构划分。最终配置确定后，可用全部开发训练数据建立分类器，再评价一直保留的独立测试集。

## 10 实际运行结果与范围

在 CS231n 目录（`courses/cs231n/`）运行，需要 NumPy：

```bash
python3 experiments/deep_review/lecture02_knn.py
```

完整代码包含 train、两种距离、预测、平票约定、五折验证和可复算的两组数据。双循环与矩阵距离逐元素一致；上面两查询 k=1 预测 `[0,0]`，k=3 预测 `[1,0]`。

固定随机种子、构造 40 个二维点的五折结果：

| 邻居数 | 五折准确率 | 平均 |
|---|---|---|
| 1 | 0.75、0.625、0.50、0.75、0.875 | 0.700 |
| 3 | 0.75、0.75、0.50、0.75、0.875 | 0.725 |
| 5 | 0.75、0.75、0.625、0.75、1.00 | 0.775 |

这里只在指定候选和构造数据上选出 k=5，没有独立测试成绩，不是 CIFAR-10 的结果，也不能推出所有任务 k=5 最好。

## 概念之间的关系

**相同表示 → 指定距离 → 最近索引 → 邻居标签 → 投票 → 验证选择配置。** 后续线性分类器会把另一种训练结果存入权重，不必对每张新图重新扫描全部训练图像。

## 来源与视频定位

- [2025 第二讲课件](https://cs231n.stanford.edu/slides/2025/lecture_2.pdf)
- [官方配套分类笔记](https://cs231n.github.io/classification/)
- [B 站 P2 25:00](https://www.bilibili.com/video/BV1aXhJ64EmW/?p=2&t=1500)：此前已核对的实现与复杂度画面点。
- [B 站 P2 35:00](https://www.bilibili.com/video/BV1aXhJ64EmW/?p=2&t=2100)：此前已核对的 L1 / L2 画面点。

时间为画面采样位置，不是精确段落边界。

[第二讲主笔记](02-image-classification.md) · [可运行完整代码](../experiments/deep_review/lecture02_knn.py) · [专题目录](00-deep-review-guide.md)
