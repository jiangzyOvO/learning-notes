# 代码与实验索引

示例用于复算公式、观察数值和检查梯度。每个脚本的输入、损失及结果说明见对应笔记；下表按课程顺序排列。

## NumPy 实现

需要 Python 和 NumPy。在 CS231n 目录（`courses/cs231n/`）运行，例如：

```bash
python3 experiments/deep_review/lecture04_matrix_backprop.py
```

| 脚本 | 实验设置与内容 |
|---|---|
| [lecture01_images.py](experiments/deep_review/lecture01_images.py) | 图像布局、展平、转置与平移 |
| [lecture02_knn.py](experiments/deep_review/lecture02_knn.py) | k-NN 距离、投票与五折验证；40 个构造点 |
| [lecture02_linear.py](experiments/deep_review/lecture02_linear.py) | 指定参数的线性分类前向 |
| [lecture02_softmax.py](experiments/deep_review/lecture02_softmax.py) | 稳定 Softmax 与极端分数损失 |
| [lecture03_gradient.py](experiments/deep_review/lecture03_gradient.py) | 单样本分类器梯度与一次更新 |
| [lecture03_batch_sgd.py](experiments/deep_review/lecture03_batch_sgd.py) | 六个构造点的批量梯度、L2 与 SGD |
| [lecture03_optimizers.py](experiments/deep_review/lecture03_optimizers.py) | 预设梯度下的优化器状态与更新 |
| [lecture04_scalar_backprop.py](experiments/deep_review/lecture04_scalar_backprop.py) | 标量计算图与数值梯度 |
| [lecture04_matrix_backprop.py](experiments/deep_review/lecture04_matrix_backprop.py) | 两层分类网络梯度；两个构造样本 |
| [lecture05_convolution.py](experiments/deep_review/lecture05_convolution.py) | 卷积前向与零数组形状检查 |
| [lecture05_backward_pooling.py](experiments/deep_review/lecture05_backward_pooling.py) | 卷积和池化梯度、CNN 形状与感受野 |
| [lecture06_initialization.py](experiments/deep_review/lecture06_initialization.py) | 初始化尺度与四点 XOR 拟合 |
| [lecture06_normalization_dropout.py](experiments/deep_review/lecture06_normalization_dropout.py) | BN、LN、Dropout 的 NumPy 算子 |
| [lecture06_residual_transfer.py](experiments/deep_review/lecture06_residual_transfer.py) | 残差梯度与指定骨干的头部训练 |
| [lecture07_rnn_bptt.py](experiments/deep_review/lecture07_rnn_bptt.py) | 三步标量 RNN 与 BPTT |
| [lecture07_lstm.py](experiments/deep_review/lecture07_lstm.py) | 三步批量 LSTM 梯度与 END 停止逻辑 |
| [lecture08_attention.py](experiments/deep_review/lecture08_attention.py) | 单头普通与因果注意力梯度 |
| [lecture08_transformer.py](experiments/deep_review/lecture08_transformer.py) | 多头、Transformer block 与 ViT 前向 |
| [lecture09_detection.py](experiments/deep_review/lecture09_detection.py) | 指定框的 IoU、NMS 与简化匹配 |
| [lecture09_segmentation_cam.py](experiments/deep_review/lecture09_segmentation_cam.py) | 像素损失、上采样与 CAM 特例 |
| [lecture10_video.py](experiments/deep_review/lecture10_video.py) | 视频采样、时间顺序与 3D 卷积前向 |
| [lecture11_distributed.py](experiments/deep_review/lecture11_distributed.py) | 单进程梯度聚合、累积与矩阵切分 |
| [lecture12_contrastive.py](experiments/deep_review/lecture12_contrastive.py) | 指定向量的对比损失与归一化梯度 |
| [lecture12_mask_teacher.py](experiments/deep_review/lecture12_mask_teacher.py) | 遮挡 MSE、固定教师目标和 EMA |
| [lecture13_autoregressive.py](experiments/deep_review/lecture13_autoregressive.py) | 三位条件表的概率、NLL 与采样 |
| [lecture13_vae.py](experiments/deep_review/lecture13_vae.py) | 标量线性高斯 VAE 与 ELBO |
| [lecture14_gan.py](experiments/deep_review/lecture14_gan.py) | 标量 GAN 损失与交替更新 |
| [lecture14_flow.py](experiments/deep_review/lecture14_flow.py) | 速度回归、解析 Gaussian 场积分与 CFG |
| [lecture15_geometry.py](experiments/deep_review/lecture15_geometry.py) | 投影、Chamfer、SDF 与射线合成 |
| [lecture16_clip.py](experiments/deep_review/lecture16_clip.py) | 指定向量的 CLIP 双向损失与梯度 |
| [lecture17_control.py](experiments/deep_review/lecture17_control.py) | 一维动力学、反馈与离散两步 MPC |
| [lecture18_evaluation.py](experiments/deep_review/lecture18_evaluation.py) | 构造分组指标、低基率与独立阶段概率 |

## 基础算例

- [guided_examples.py](experiments/guided_examples.py)：标准库实现的距离、梯度、卷积、池化、注意力、IoU 与流路径。
- [chapter02_forward.py](experiments/chapter02_forward.py)：第二讲前向算例。
- [chapter03_optimization.py](experiments/chapter03_optimization.py)：第三讲优化算例。
- [remaining_examples.py](experiments/remaining_examples.py)：后续课程的基础数值例子。
- [sigmoid_network.py](experiments/sigmoid_network.py)：NumPy 两层 Sigmoid 网络，平方误差求和。固定随机数据下 2000 次更新，训练损失约 32666.41→3.57；权重梯度中心差分误差约 4.85×10⁻¹¹。未划分验证和测试集。

标准库脚本无需额外依赖；Sigmoid 网络需要 NumPy。

## 结果的适用范围

多数例子采用构造数据、指定参数或解析模型。卷积、注意力和损失的差分检查针对具体可导点；完整 CNN 与 ViT 示例中有只检查前向形状的部分。分布式实验为单进程模拟，机器人实验为一维无噪声控制。

这些结果用于验证计算，不用于比较真实图像模型的准确率、生成质量、多卡吞吐量或机器人成功率。完整任务评测需要独立的数据划分与相应指标。

[课程目录](README.md#课程详细笔记) · [推导与实现](notes/00-deep-review-guide.md) · [参考资料](SOURCES.md)
