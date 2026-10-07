# CS336 来源索引

## 第一讲 课程总览与 Tokenization

- 课程版本：Stanford CS336 Spring 2026。
- [官方视频](https://www.youtube.com/watch?v=JuoVZkPBiKk)：Overview, Tokenization。
- [官方第一讲讲义代码](https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py)：课程动机、语言模型发展、五个课程单元、tokenizer 接口及字符、字节、词和 BPE 示例。
- [课程官网](https://cs336.stanford.edu/)：课程定位、前置知识、作业与讲次安排。

本版笔记依据 2026 年 10 月 6 日读取的官方配套讲义整理。视频页面标题已核对，字幕与逐段画面未取得，因此没有标注视频时间轴；讲义 main 分支可能继续更新；[本次读取对应的固定版本](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_01.py)可供复查。正文中的概率算例、UTF-8 表格解释和资源数值算例为教学补充，BPE 主例采用讲义的 `the cat in the hat` 与三轮合并。

教学脚本是独立的精简实现，保留 byte-level BPE 的关键计算；不含预分词、特殊 token、流式处理或高效训练优化。

## 第二讲 PyTorch 张量操作与资源核算

- [官方 lecture_02.py](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py)：张量、dtype、设备、einops、FLOPs、算术强度、Roofline、梯度、深层网络、AdaGrad、训练循环、梯度累积与激活重计算。
- [官方可执行讲义](https://cs336.stanford.edu/lectures/?trace=lecture_02)。
- 读取日期：2026 年 10 月 6 日。通过官方仓库文件与可访问的 CDN 同源镜像核对；读取的完整代码 SHA256：`f731b3849afc970a062f1fe692d7d80f5952cf82c2a4f6f2803a8fb9a9dabfd6`。main 分支可能更新。
- 教学补充：小尺寸矩阵、输入 ID 到 embedding 的形状、梯度显式推导、一次 SGD、梯度累积加权与简化模型的适用条件。
- 标准库脚本验证数值与手算关系，不执行 PyTorch/einops 或 GPU 基准。正文接口按官方讲义及库文档解释，实际设备执行留待环境与作业实践。
- 按学习者最新要求以官方讲义推进，不以视频读取作为学习前提。

## 第三讲 模型架构与超参数

- [官方 lecture_03.pdf](https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf)，67 页，读取日期 2026 年 10 月 6 日。下载文件与官方 GitHub API 返回文件的 SHA256 一致：`20a953541b41a099bc69b8d809139a4ccdb2aefedaa13e50c766245fe4b5167e`。
- 全部页面已提取文字；结构图与关键公式另核对第 4、14、34、43、54、56 页的渲染画面。
- 页码映射：整体结构 2–4；归一化与偏置 10–19；激活、门控、串并行 20–29；位置与 RoPE 30–35；超参数和正则化 36–51；稳定性 52–56；注意力变体 57–66；总结 67。
- 教学补充：下一项目标与形状、因果掩码、Q/K/V 基础机制、注意力手算、归一化展开、门控参数数和 RoPE 二维旋转。
- 课件第 14 页 RMSNorm 使用范数简写，笔记明确使用均方根平均定义；第 43 页部分模型尺寸与表列比例不自洽，笔记不照抄为配置建议。
- 数值脚本仅验证教学算例，不构建模型、运行 GPU 或替代正式作业。
- [2026 官网](https://cs336.stanford.edu/)安排 19 讲；按学习者要求优先推进第 1–18 讲，第 19 讲保留为补充入口。
