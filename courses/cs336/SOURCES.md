# CS336 来源索引

## 第一讲 课程总览与 Tokenization

- 课程版本：Stanford CS336 Spring 2026。
- [官方视频](https://www.youtube.com/watch?v=JuoVZkPBiKk)：Overview, Tokenization。
- [官方第一讲讲义代码](https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py)：课程动机、语言模型发展、五个课程单元、tokenizer 接口及字符、字节、词和 BPE 示例。
- [课程官网](https://cs336.stanford.edu/)：课程定位、前置知识、作业与讲次安排。

本版笔记依据 2026 年 10 月 6 日读取的官方配套讲义整理。视频页面标题已核对，字幕与逐段画面未取得，因此没有标注视频时间轴；讲义 main 分支可能继续更新；[本次读取对应的固定版本](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_01.py)可供复查。正文中的概率算例、UTF-8 表格解释和资源数值算例为教学补充，BPE 主例采用讲义的 `the cat in the hat` 与三轮合并。

教学脚本是独立的精简实现，保留 byte-level BPE 的关键计算；不含预分词、特殊 token、流式处理或高效训练优化。
