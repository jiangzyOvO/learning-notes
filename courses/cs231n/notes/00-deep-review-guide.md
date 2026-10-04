# CS231n 推导与实现目录

按课程内容分组，收录公式推导、手算和 Python / NumPy 实现。课程正文见 [18 讲目录](../README.md#课程详细笔记)，概念之间的联系见 [全课程复习](00-deep-review-summary.md)。

## 专题目录

| 分组 | 核心问题 | 推导与实现 |
|---|---|---|
| 1 | 机器看到的图像是什么？ | [图像与布局](01-introduction.md) |
| 2 | 怎样从输入得到类别与损失？ | [k-NN](02-knn-implementation.md)、[线性分类](02-linear-classifier-deep-review.md)、[稳定 Softmax](02-softmax-numerical-stability.md) |
| 3 | 参数怎样朝更低损失更新？ | [梯度下降](03-gradient-descent-deep-review.md)、[批量与正则](03-batch-regularization-deep-review.md)、[优化器](03-optimizers-deep-review.md) |
| 4 | 多层网络的梯度怎样计算？ | [标量反传](04-scalar-backprop-deep-review.md)、[矩阵反传](04-matrix-backprop-deep-review.md) |
| 5 | 卷积怎样处理空间与通道？ | [卷积前向](05-convolution-forward-deep-review.md)、[反传与池化](05-backward-pooling-deep-review.md) |
| 6 | 怎样稳定训练并处理序列？ | [初始化](06-initialization-diagnostics-deep-review.md)、[归一化](06-normalization-dropout-deep-review.md)、[残差迁移](06-residual-transfer-deep-review.md)、[RNN](07-rnn-bptt-deep-review.md)、[LSTM](07-lstm-generation-deep-review.md) |
| 7 | token 怎样交换信息？ | [QKV](08-attention-qkv-deep-review.md)、[Transformer / ViT](08-transformer-vit-deep-review.md) |
| 8 | 怎样定位、分割、理解时间并扩大计算？ | [检测](09-detection-deep-review.md)、[分割与 CAM](09-segmentation-cam-deep-review.md)、[视频](10-video-deep-review.md)、[分布式](11-distributed-deep-review.md) |
| 9 | 怎样学习表示与生成分布？ | [对比学习](12-contrastive-deep-review.md)、[MAE / DINO](12-mae-dino-deep-review.md)、[自回归](13-probability-autoregressive-deep-review.md)、[VAE](13-vae-elbo-deep-review.md)、[GAN](14-gan-deep-review.md)、[流与扩散](14-flow-diffusion-deep-review.md) |
| 10 | 怎样理解空间、语言、行动与人的需要？ | [三维](15-geometry-deep-review.md)、[视觉语言](16-vision-language-deep-review.md)、[机器人](17-robot-control-deep-review.md)、[以人为中心](18-human-centered-deep-review.md) |

## 阅读与代码约定

先明确输入、输出和张量形状，再展开一个元素的计算，最后连接批量代码。遇到前置符号可查 [数学与 NumPy 工具箱](00-math-and-numpy.md)。

代码中的损失求和、平均方式、权重布局和训练模式均在对应专题中说明。数值梯度检查需避开不可导点；小型算例用于检查计算，真实任务还需划分训练、验证和测试数据。

视频分集按 Spring 2025 课程对应；时间定位是画面采样位置，不是精确段落边界。课件、视频和补充资料见 [来源](../SOURCES.md)。

[入门笔记](../beginner/README.md) · [实验索引](../PROGRESS.md) · [全课程复习](00-deep-review-summary.md)
