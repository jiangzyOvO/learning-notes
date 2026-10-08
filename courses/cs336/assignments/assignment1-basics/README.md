# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## 当前作业进度

2026-10-08 保存的学习进度：已实现 BPE 训练、tokenizer、Linear、Embedding、RMSNorm、SiLU、SwiGLU、稳定 softmax、缩放点积注意力、因果多头注意力、RoPE 和 Pre-Norm TransformerBlock。

[模型代码](cs336_basics/model.py)包含截至 TransformerBlock 的实现；[测试适配器](tests/adapters.py)连接已完成的组件。完整 TransformerLM、损失函数、优化器、训练流程和实验尚未完成。

模型组件与 softmax 的检查结果为 **12 passed、4 deselected**，对应命令如下。未完成组件的测试入口仍保留 `NotImplementedError`。

```sh
uv run pytest tests/test_model.py tests/test_nn_utils.py -k 'not transformer_lm and not cross_entropy and not gradient_clipping' -q
```

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Tests for unfinished components still fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

