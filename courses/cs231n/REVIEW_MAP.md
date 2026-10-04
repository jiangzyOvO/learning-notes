# 全课程复习路线与概念索引

这是一份查阅地图，不是学习测验，也不要求按固定天数完成。每章正文都可以从 [完整目录](README.md) 打开。

**首次阅读**先沿[10 单元从零路线](beginner/README.md)学习，再看[阅读指南](READING_GUIDE.md)。各章新增的学习卡说明第一遍和第二遍分别读什么；公式看不懂时，可直接查[数学与 NumPy 工具箱](notes/00-math-and-numpy.md)。

## 按这次讲解回到具体例子

| 卡住的地方 | 先读的例子 |
|---|---|
| 图片的 3 是颜色、图像数量，还是类别数？ | [单元 01：图像与分数](beginner/01-data-and-prediction.md) |
| 负梯度为什么使权重增加？偏置也要更新吗？ | [单元 02：一个权重与多个参数](beginner/02-learning-and-gradients.md) |
| 为什么梯度公式有转置，偏置为什么求和？ | [单元 03：两个样本的矩阵反传](beginner/03-networks-and-backprop.md) |
| 16 个通道从哪里来，P=1 为什么不缩小？ | [单元 04：卷积窗口](beginner/04-convolution.md) |
| 16×16×16 的三个 16 是什么？GAP 与展平一样吗？ | [单元 05：完整 CNN](beginner/05-pooling-and-cnn.md) |
| Dropout 为什么把留下的值放大？BN 为什么分模式？ | [单元 06：稳定训练](beginner/06-training.md) |
| 注意力权重到底乘谁？token 怎样变成图片特征？ | [单元 07：完整 QKV 手算](beginner/07-attention.md) |
| 分类、检测、分割有什么区别？ | [单元 08：输出与标签](beginner/08-visual-tasks.md) |
| 没标签怎么有损失？生成时知道真实图片吗？ | [单元 09：训练与生成分开看](beginner/09-self-supervision-and-generation.md) |
| 视频、三维、语言、机器人怎样连起来？ | [单元 10：完整地图](beginner/10-bigger-picture.md) |

## 1. 先建立训练的完整链条

[第二章](notes/02-image-classification.md)定义分数与损失，[第三章](notes/03-regularization-optimization.md)说明更新方向和步幅，[第四章](notes/04-neural-networks-backprop.md)说明梯度如何实际算出来。

把三章放在一起，就是：

$$x\rightarrow s=f_\theta(x)\rightarrow L(s,y)
\rightarrow\nabla_\theta L\rightarrow\theta_{new}.$$

前向得出预测，损失评价预测，反向计算敏感性，优化器更新参数。正则化改变目标中的偏好；数据划分与验证负责判断是否适用于新样本。

## 2. 再看模型怎样利用输入结构

| 数据或结构 | 主要章节 | 为什么需要这种结构 |
|---|---|---|
| 图像局部邻域 | [第五章](notes/05-convolutional-networks.md) | 利用邻近关系与共享局部模式 |
| 深层网络 | [第六章](notes/06-cnn-architectures.md) | 让信息、梯度与数据利用更稳定 |
| 有顺序的序列 | [第七章](notes/07-recurrent-networks.md) | 把历史带入当前计算 |
| 跨位置关系 | [第八章](notes/08-attention-transformers.md) | 按查询动态读取其他位置 |
| 物体与像素位置 | [第九章](notes/09-detection-segmentation.md) | 从一个全图标签走到结构化输出 |
| 时间与运动 | [第十章](notes/10-video-understanding.md) | 区分静态外观、动作顺序与长程活动 |

CNN 与 Transformer 都在学习表示，但连接方式、归纳偏置与计算成本不同；比较时不要只问谁“更先进”。

## 3. 然后区分训练资源与监督来源

[第十一章](notes/11-distributed-training.md)回答怎样把计算放到多张卡上；[第十二章](notes/12-self-supervised-learning.md)回答没有人工类别标签时如何构造学习信号。前者主要改变执行方式，后者改变训练目标与数据关系。

自监督不等于不计算损失；分布式也不等于模型自动变好。

## 4. 生成模型先看训练，再看生成

| 方法 | 训练时主要做什么 | 生成时主要做什么 |
|---|---|---|
| [自回归](notes/13-generative-models-1.md) | 拟合各位置的条件概率 | 按条件依赖逐步采样 |
| [VAE](notes/13-generative-models-1.md) | 优化重建对数似然与先验匹配的下界 | 从先验抽潜变量，再解码 |
| [GAN](notes/14-generative-models-2.md) | 生成器与判别器交替优化 | 噪声经生成器得到样本 |
| [Rectified Flow](notes/14-generative-models-2.md) | 在插值点回归速度 | 从噪声端反向积分 |

“都能生成图片”不意味着训练目标、概率含义与采样成本相同。遇到新公式，先问变量是什么、时间朝哪边、网络实际预测什么。

## 5. 最后连接空间、语言、行动与人

[第十五章](notes/15-3d-vision.md)选择怎样表示三维；[第十六章](notes/16-vision-language.md)连接视觉和语言；[第十七章](notes/17-robot-learning.md)让动作改变未来观测；[第十八章](notes/18-human-centered-ai.md)把这些能力与人的需求联系起来。

这几章会复用前面大量概念：机器人策略可使用 Transformer，动作序列可以用生成模型表达，视觉语言模型可使用对比预训练，三维重建也使用梯度和可微渲染。

## 6. 按困惑查章节

| 遇到的困惑 | 回到哪里 |
|---|---|
| 一个向量为什么既叫 100 维，又是“一维数组”？ | [工具箱：数组与形状](notes/00-math-and-numpy.md#1-标量向量矩阵张量) |
| 为什么 train_sq 不用 keepdims，转置怎么用？ | [工具箱：axis 与广播](notes/00-math-and-numpy.md#5-axis-与-keepdims到底对谁求和) |
| 距离图中的亮行、亮列是什么意思？ | [第二章：距离矩阵](notes/02-image-classification.md#从一个距离到距离矩阵) |
| Softmax 为什么减最大值？ | [数值稳定专题](notes/02-softmax-numerical-stability.md) |
| λ、学习率、梯度有什么区别？ | [第三章](notes/03-regularization-optimization.md) |
| 为什么反传要转置、偏置求和？ | [第四章](notes/04-neural-networks-backprop.md) |
| 截图里的 h*(1-h) 和 grad_w1 是怎么来的？ | [NumPy 网络逐行讲解](notes/04-numpy-network-walkthrough.md) |
| 通道、卷积核、参数量怎么数？ | [第五章](notes/05-convolutional-networks.md) |
| train/eval 为什么预测不同？ | [第六章](notes/06-cnn-architectures.md) |
| LSTM 的 h 和 c 有什么不同？ | [第七章](notes/07-recurrent-networks.md) |
| Q、K、V 的形状与 softmax 轴？ | [第八章](notes/08-attention-transformers.md) |
| 分类对了为什么检测仍算错？ | [第九章](notes/09-detection-segmentation.md) |
| 帧数、帧率与覆盖时间的关系？ | [第十章](notes/10-video-understanding.md) |
| 参数装得下为什么训练放不下？ | [第十一章](notes/11-distributed-training.md) |
| 没有标签为什么还能算损失？ | [第十二章](notes/12-self-supervised-learning.md) |
| VAE 的 KL 为什么是加号？ | [第十三章](notes/13-generative-models-1.md)中负 ELBO 的约定 |
| 速度预测和噪声预测一样吗？ | [第十四章](notes/14-generative-models-2.md) |
| NeRF 是表面还是体密度？ | [第十五章](notes/15-3d-vision.md) |
| CLIP 能聊天吗，SAM 能识别类别吗？ | [第十六章](notes/16-vision-language.md) |
| 监督学习与强化学习差在哪？ | [第十七章](notes/17-robot-learning.md) |
| 高准确率为什么还可能不好用？ | [第十八章](notes/18-human-centered-ai.md) |

## 7. 符号容易同名，先看本章定义

- N 可能是样本数、token 数或点数，不能跨章节直接代入。
- D 可以指特征维度，也可以是 GAN 判别器，依上下文区分。
- z 可能是潜变量或噪声；第十四章专门区分潜空间编码与采样噪声。
- t 在序列中通常是离散位置，在流模型中可以是连续时间。
- v 在 Momentum、Adam 和 Rectified Flow 中分别可能表示累积量、二阶矩或速度。
- 同一个线性层可按行样本或列样本写公式；权重转置约定随之变化，先核对形状。

## 8. 配套例子

[可运行脚本](experiments/remaining_examples.py)仅用标准库复现各章数值，不需要 GPU。它帮助核对一小步计算，不代表运行了完整网络、模型训练或数据集实验。
