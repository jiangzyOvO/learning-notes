# CS231n 全课程复习

按 CS231n Spring 2025 课程整理，汇总主要概念、推导与代码入口。

## 课程内容之间的联系

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

## 用六个问题复习任意模型

1. 输入、输出、标签分别是什么？每个维度有什么含义？
2. 网络前向怎样把输入变为预测？
3. 损失衡量什么，在哪些样本或位置上求和与平均？
4. 哪些量需要梯度，哪些量固定或停止梯度？
5. 参数怎样更新，训练与推理分别需要什么状态？
6. 用什么数据和指标判断效果，结果能支持什么结论？

例如 CLIP：输入成对图文 → 两个编码器 → 归一化相似度矩阵 → 双向交叉熵 → 更新编码器 → 用匹配或下游任务评价。机器人则还要在行动后重新观察，评价整段任务。

## 生成、三维与行动的代码示例

| 内容 | 教学脚本 | 示例内容 |
|---|---|---|
| 流匹配 | [lecture14_flow.py](../experiments/deep_review/lecture14_flow.py) | 速度回归梯度、反向积分、解析 Gaussian 场、CFG 算术 |
| 三维 | [lecture15_geometry.py](../experiments/deep_review/lecture15_geometry.py) | 投影歧义、点集重排、Chamfer、SDF、射线合成 |
| 图文 | [lecture16_clip.py](../experiments/deep_review/lecture16_clip.py) | 双向损失、归一化梯度、给定向量匹配、投影形状 |
| 机器人 | [lecture17_control.py](../experiments/deep_review/lecture17_control.py) | 回报、终止目标、动力学拟合、反馈与离散 MPC |
| 评价 | [lecture18_evaluation.py](../experiments/deep_review/lecture18_evaluation.py) | 分组加权、低基率报警、独立阶段概率 |

这些是小数据或解析例子，没有训练完整图像生成器、CLIP、NeRF 或机器人策略。脚本检查失败时会触发断言；运行环境需要 Python 和 NumPy。

## 从读懂走向独立应用

先尝试在不查看实现时重写核心算子，再用独立小例子或差分检查解释错误。

实际任务需要划分训练、验证和测试数据，并保存设置与失败案例。CNN、Transformer、检测、分割、自监督和生成模型的完整训练，可在理解核心算子后结合课程作业开展。

需要回看基础时读 [数学与 NumPy 工具箱](00-math-and-numpy.md) 和 [入门十单元](../beginner/README.md)；需要对应视频和详细内容时读 [18 讲目录](../README.md#课程详细笔记)。

[专题目录](00-deep-review-guide.md) · [实验索引](../PROGRESS.md) · [来源](../SOURCES.md)
