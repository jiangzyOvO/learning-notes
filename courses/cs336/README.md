# Stanford CS336 语言模型从零构建

以 Stanford CS336 Spring 2026 官方讲义与配套代码为主要依据，面向初学者解释语言模型的原理、实现与资源约束。先按学习者要求推进第 1–18 讲；2026 官网共安排 19 讲，其中第 18、19 讲为嘉宾讲座。

## 课程笔记

| 讲次 | 主题 | 阅读入口 |
|---|---|---|
| 01 | 课程总览与 Tokenization | [详细笔记](notes/01-overview-tokenization.md) · [代码示例](experiments/lecture01_tokenization.py) |
| 02 | PyTorch 张量操作与资源核算 | [详细笔记](notes/02-pytorch-resource-accounting.md) · [核算示例](experiments/lecture02_accounting.py) |
| 03 | 模型架构与超参数 | [详细笔记](notes/03-architectures-hyperparameters.md) · [数值示例](experiments/lecture03_math.py) |

## 配套作业

- [Assignment 1：Basics](assignments/assignment1-basics/README.md)：2026-10-08 保存的作业进度，包含 BPE、tokenizer，以及截至 TransformerBlock 的模型组件。模型组件与 softmax 的检查为 12 passed、4 deselected；完整语言模型、训练流程和实验尚未完成。作业目录保留官方测试与依赖配置，具体检查命令见作业说明。

## 后续课程

以下按 [2026 官方目录](https://cs336.stanford.edu/)记录，笔记随学习逐讲整理。

| 讲次 | 官方主题 |
|---|---|
| 04 | 注意力替代方案与混合专家 MoE |
| 05 | GPU 与 TPU |
| 06 | GPU kernels 与 Triton |
| 07 | 并行计算 |
| 08 | 并行计算 |
| 09 | 扩展规律 |
| 10 | 推理 |
| 11 | 扩展规律 |
| 12 | 评价 |
| 13 | 数据来源与数据集 |
| 14 | 数据过滤、去重、混合与合成 |
| 15 | 中期与后训练 SFT/RLHF |
| 16 | 后训练与可验证奖励强化学习 RLVR |
| 17 | 对齐与多模态 |
| 18 | 嘉宾讲座 Daniel Selsam |
| 19 | 嘉宾讲座 Dan Fu，补充课程 |

第一讲先建立文字、token、编号、预测之间的关系，再解释字符级、字节级、词级方案，以及 BPE 的训练、编码和解码。

## 阅读方法

先读每章概览，再从具体例子进入概念与公式。代码示例用于观察中间结果；补充推导和教学例子会明确标注。第一次遇到术语时先理解其用途，再记英文名称。

## 资料

- [课程官网](https://cs336.stanford.edu/)
- [所选视频列表](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)
- [官方讲义代码](https://github.com/stanford-cs336/lectures)
- [来源索引](SOURCES.md)

[返回学习笔记首页](../../README.md)
