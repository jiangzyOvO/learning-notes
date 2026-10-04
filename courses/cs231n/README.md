# CS231n 中文学习笔记

按 Stanford CS231n Spring 2025 课程整理的中文笔记，包含 18 讲课程内容、基础讲解、公式推导、NumPy 实现和实验示例。

## 从这里开始

**第一次学习，打开 [入门笔记](beginner/README.md)，然后从 [01 图像、分数与损失](beginner/01-data-and-prediction.md) 顺着读。** 每一单元都先解释为什么需要这个概念，再手算，最后连接公式和后续内容。

| 你的需要 | 阅读入口 |
|---|---|
| 从基础开始学习 | [10 单元入门路线](beginner/README.md) |
| 看到矩阵、转置、axis、keepdims 就卡住 | [数学与 NumPy 工具箱](notes/00-math-and-numpy.md) |
| 没弄清卷积的通道和尺寸 | [把 3×32×32 → 16×32×32 算清楚](beginner/04-convolution.md) |
| 不明白特征图怎样成为分类结果 | [连接一个完整 CNN](beginner/05-pooling-and-cnn.md) |
| 想跟随课程视频或查完整推导 | 下方的 [18 讲目录](#课程详细笔记) |
| 想按具体困惑查找 | [复习地图](REVIEW_MAP.md) |
| 查找代码及实验设置 | [实验索引](PROGRESS.md) |

## 入门笔记

| 顺序 | 从一个问题开始 |
|---|---|
| [01 图像、分数与损失](beginner/01-data-and-prediction.md) | 像素、k-NN、距离矩阵、线性分数、Softmax 与交叉熵 |
| [02 参数怎样学习](beginner/02-learning-and-gradients.md) | 一个权重的更新 → 偏置 → 批量平均 → 数据划分 |
| [03 神经网络与反向传播](beginner/03-networks-and-backprop.md) | 非线性 → 标量网络 → 矩阵梯度 → 完整小代码 |
| [04 卷积窗口](beginner/04-convolution.md) | 单窗口 → 多通道 → 填充、步长 → 参数量 |
| [05 完整 CNN](beginner/05-pooling-and-cnn.md) | 池化 → 第二层卷积 → GAP → 十个类别分数 |
| [06 稳定训练](beginner/06-training.md) | 初始化、学习率、BN、Dropout、过拟合与早停 |
| [07 残差与注意力](beginner/07-attention.md) | 残差加法 → 两个 token 的 QKV 手算 → ViT |
| [08 检测与分割](beginner/08-visual-tasks.md) | 输出由一个类别变成对象框、像素类别和掩码 |
| [09 自监督与生成](beginner/09-self-supervision-and-generation.md) | 对比、遮挡、自回归、VAE、GAN、Flow Matching |
| [10 完整课程地图](beginner/10-bigger-picture.md) | 时间、三维、图文、行动、算力与人的需求 |

重要关系用 **粗体** 标出；提示框解释易错点。图片有文字说明，公式定义符号和形状。布局、阅读顺序及来源说明见 [阅读指南](READING_GUIDE.md)。

## 课程详细笔记

| 章 | 笔记 | 重点 |
|---|---|---|
| 01 | [课程导论](notes/01-introduction.md) | 计算机视觉、历史与课程方向 |
| 02 | [图像分类与线性分类器](notes/02-image-classification.md) | k-NN、线性分数、Softmax、损失 |
| 03 | [正则化与优化](notes/03-regularization-optimization.md) | 梯度、SGD、Momentum、Adam、AdamW |
| 04 | [神经网络与反向传播](notes/04-neural-networks-backprop.md) | 非线性、计算图、链式法则、矩阵梯度 |
| 05 | [卷积神经网络](notes/05-convolutional-networks.md) | 卷积、形状、参数量、池化、感受野 |
| 06 | [训练 CNN 与经典架构](notes/06-cnn-architectures.md) | BN、Dropout、初始化、ResNet、迁移 |
| 07 | [循环神经网络](notes/07-recurrent-networks.md) | 序列、BPTT、语言生成、LSTM |
| 08 | [注意力与 Transformer](notes/08-attention-transformers.md) | QKV、掩码、多头、ViT、现代组件 |
| 09 | [检测、分割与模型可视化](notes/09-detection-segmentation.md) | FCN、U-Net、检测器、IoU、CAM/Grad-CAM |
| 10 | [视频理解](notes/10-video-understanding.md) | 采样、3D CNN、光流、SlowFast、音视频 |
| 11 | [大规模分布式训练](notes/11-distributed-training.md) | 数据并行、FSDP、重算、CP/TP/PP、利用率 |
| 12 | [自监督学习](notes/12-self-supervised-learning.md) | 预文本任务、MAE、InfoNCE、SimCLR、MoCo、DINO |
| 13 | [生成模型（一）](notes/13-generative-models-1.md) | 最大似然、自回归、VAE、ELBO、重参数化 |
| 14 | [生成模型（二）](notes/14-generative-models-2.md) | GAN、Rectified Flow、CFG、潜空间扩散 |
| 15 | [三维视觉](notes/15-3d-vision.md) | 点云、网格、体素、SDF、PointNet、NeRF |
| 16 | [视觉与语言](notes/16-vision-language.md) | CLIP、VLM、LLaVA、Flamingo、SAM、程序组合 |
| 17 | [机器人学习](notes/17-robot-learning.md) | 感知行动闭环、RL、规划、模仿、VLA |
| 18 | [以人为中心的 AI](notes/18-human-centered-ai.md) | 人类视觉、偏差、隐私、辅助与真实任务 |

## 推导与代码专题

- [全课程复习](notes/00-deep-review-summary.md)：概念联系与专题索引。
- [推导与实现目录](notes/00-deep-review-guide.md)：按课程分组查找详细推导和代码。
- [Softmax 数值稳定性](notes/02-softmax-numerical-stability.md)：为什么减最大值，为什么还要直接计算对数损失。
- [两层 NumPy 网络逐行讲解](notes/04-numpy-network-walkthrough.md)：Sigmoid、平方误差和、矩阵梯度与权重更新。与入门路线的 ReLU 算例分别标明。

## 跟着复算

在 CS231n 目录（`courses/cs231n/`）运行：

```bash
python3 experiments/guided_examples.py
```

该脚本只需 Python 标准库，复算距离、梯度、卷积、池化、注意力、IoU 与流路径的小型算例。更多 NumPy 实现见[实验索引](PROGRESS.md)。

原课程算例仍可运行：

```bash
python3 experiments/chapter02_forward.py
python3 experiments/chapter03_optimization.py
python3 experiments/remaining_examples.py
```

Sigmoid 网络的训练演示需要 NumPy：

```bash
python3 experiments/sigmoid_network.py
```

所有原创图存于 `assets/`，保留 PNG 与 SVG；新增配图源码为 [draw_guided.py](scripts/draw_guided.py)，使用 Matplotlib。图中的符号和数值在正文中有对应解释。

## 课程版本、来源与 GitHub

- [2025 官方课程安排](https://cs231n.stanford.edu/2025/schedule.html)
- [B 站视频](https://www.bilibili.com/video/BV1aXhJ64EmW/)
- [官方配套笔记](https://cs231n.github.io/)

课程版本与参考资料见 [SOURCES.md](SOURCES.md)。手算与配图用于解释原理，实验设置在各例中注明。

[返回学习笔记首页](../../README.md)。课件和视频通过来源链接访问。

[笔记模板](templates/lecture.md) · [常见问题](QUESTIONS.md) · [实验索引](PROGRESS.md)
