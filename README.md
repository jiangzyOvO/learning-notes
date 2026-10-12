# 学习笔记

课程笔记、论文阅读、专题整理与代码示例。各课程和论文保留独立目录，包含正文、参考资料、图片或实验。

## 课程

| 课程 | 内容 | 阅读入口 |
|---|---|---|
| Stanford CS231n 2025 | 计算机视觉、神经网络、生成模型与机器人学习 | [课程目录](courses/cs231n/README.md) · [全课程复习](courses/cs231n/notes/00-deep-review-summary.md) |
| Stanford CS336 2026 | 从零构建语言模型，理解训练、系统与数据 | [课程目录](courses/cs336/README.md) · [第一讲](courses/cs336/notes/01-overview-tokenization.md) |

## 专题

跨课程的概念整理放在 [topics/](topics/README.md)。

## 论文阅读

论文笔记按论文独立整理，完整列表见 [papers/](papers/README.md)。

| 论文 | 内容 | 阅读入口 |
|---|---|---|
| DeepSeekMath | 数学数据筛选、SFT、PPO 与 GRPO、奖励到参数更新的推导 | [阅读笔记](papers/deepseekmath/README.md) |

## 目录结构

```text
learning-notes/
├── README.md
├── courses/
│   └── cs231n/
│       ├── README.md
│       ├── notes/
│       ├── beginner/
│       ├── experiments/
│       ├── assets/
│       └── SOURCES.md
├── topics/
└── papers/
    ├── README.md
    └── deepseekmath/
        ├── README.md
        └── assets/
```

新课程放在 `courses/课程名称/`，在该目录下建立课程首页，并将入口加入上方课程表。图片、代码和参考资料与课程放在一起，使用相对链接。

新论文放在 `papers/论文名称/`，以 `README.md` 作为阅读入口，图片放入同目录下的 `assets/`，并更新论文目录。
