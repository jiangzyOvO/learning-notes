# 1. 为什么需要资源核算？
## 1.1 场景一：时间估算
```text
问题：老板问：“在 1024 张 H100 显卡上，训练一个 70B（700 亿参数）的模型，数据量是 15T （15为万亿 tokens），大概要多久？“
```
如果直接跑去写代码测试，可能几天甚至几个月都出不来结果。因此，我们需要学会 “Napkin math“ （餐巾纸计算，即快速估算）。
### Step1：计算总工作量
FLOPS（Floating Point Operations，浮点运算次数）是衡量计算复杂度的常用指标，表示一个算法或模型在执行过程中所需的浮点加减乘除等运算的总次数。（训练模型本质上是进行浮点运算（FLOPs））
有一个经验公式：
$$总计算量 = 6 \times 参数量 \times Token 数量$$
- 前向传播（Forward Pass）：计算一次约为 $2 \times 参数量（乘法 + 加法）$
- 反向传播（Backward Pass）：计算梯度约为前向传播的 2 倍工作量，即 $4 \times 参数量$
```text
前向传播：
y = w1x1 + w2x2 + w3x3 ...
一共有 n 次乘法和 n-1 次加法，共 2n-1
所以可以粗略理解为每个参数参与一个乘法和一次累加，约等于 2 次浮点运算。

反向传播：
我们需要分别反向计算参数梯度和输入梯度，所以大约是 2 倍的关系。
```
### Step2：计算硬件算力
查阅显卡的算力，比如 NVIDIA H100 的白皮书中，其 FP16/BF16 的峰值算力约为 1979 TFLOPS（每秒万次浮点运算）。
	但是要注意的是，这个值是 NVIDIA H100 GPU 在使用 FP16 或 BF16 数据类型，且启用结构化稀疏（Structured Sparsity）时可以达到的最大计算吞吐量。对于我们训练普通的稠密模型（如标准的 LLaMA-1到LLaMA-3系列、Qwen3-0.6B/1.7B/4B/8B/14B/32B等），理论峰值减半，约 990 TFLOPS。
	然后关键的是，990 TFLOPS 是 H100 的理论峰值，实际运行模型时，由于各种软硬件开销，几乎不可能达到 100% 模型算力利用率（MFU，Model FLOPs Utilization），通常按 30%-60% 的利用率估算更现实。
### Step3：得出结果
$$时间 = \frac{总工作量}{总算力}$$

## 1.2 场景二：内存估算
```text
问题：在 8 张 H100 GPU 上，使用 AdamW 优化器（朴素实现，即所有数据都用 float32 存储，没有混合精度或压缩），你能训练的最大模型参数量是多少？
```
一个错误的思维：
8 张 H100（每张 80GB 显存），显存共 640GB。参数采用 float32 存储，一个参数 4 字节，所以 $640GB / 4字节 = 1600亿参数（160B）$？这是不对的。

训练时，显存里存的不只是参数（Parameters），还有：
- 梯度（Gradients）：和参数一样大，需要 4 字节
- 优化器状态（Optimizer States）：比如常用的 AdamW 优化器，需要存储每个参数的一阶矩估计（梯度的指数移动平均）和二阶矩估计（梯度平方的指数移动平均）
因此，每个参数在训练时大约占用 16 字节：
- 参数（FP32）：4 bytes
- 梯度（FP32）：4 bytes
- 优化器状态（FP32）：8 bytes（2 个变量 x 4 bytes）
$$最大参数量 = \frac{640 \times 10^9 bytes}{16 bytes/param} \approx 400亿（40B）$$
	重要的点还有，这个计算忽略了激活值内存，所以只是一个理论上界。
# 2. 张量（Tensors）
## 2.1 张量基础
张量是深度学习中存储一切的基本构建块，包括：
- 数据（data）
- 模型参数（parameters）
- 梯度（gradients）
- 优化器状态（optimizer state）
- 激活值（activations）
### 创建方式
```Python
x = torch.tensor([[1., 2, 3], [4, 5, 6]])  # 从Python列表创建
x = torch.zeros(4, 8)  # 4x8的零矩阵
x = torch.ones(4, 8)   # 4x8的全1矩阵
x = torch.randn(4, 8)  # 4x8的正态分布随机数

# 分配但不初始化值（用于自定义初始化的值）
x = torch.empty(4, 8)  # 未初始化的4x8矩阵
nn.init.trunc_normal_(x, mean=0, std=1, a=-2, b=2)  # 截断正态分布初始化，分布均值为0，标准差为1，截断范围，只保留[-2, 2]区间内的值，超出此范围的值会被重新采样
```
### 张量的秩（Rank）
**张量的秩就是维度数：**
```Python
x = torch.zeros(4)        # 秩 1 张量（向量），shape = (4,)
x = torch.zeros(4, 8)     # 秩 2 张量（矩阵），shape = (4, 8)
x = torch.zeros(4, 8, 2)  # 秩 3 张量，shape = (4, 8, 2)
```
在 Transformer 中，常见的张量是秩 4 的：
```Python
B = 32   # Batch size（批次大小）
S = 16   # Sequence length（序列长度）
H = 16   # Number of heads（注意力头数）
D = 64   # Hidden dimension per head（每个头的隐藏维度）
x = torch.zeros(B, S, H, D)
```
## 2.2 张量的操作
大多数张量都是通过对其他张量执行操作来创建的，每个操作都会占用一定的内存和计算资源。
### 张量视图（view）
许多操作只是提供张量一个不同“视图”。这不会创建副本，因此在一个张量中的修改会影响另一个。
```Python
x = torch.tensor([[1., 2, 3], [4, 5, 6]])

# 这些操作不复制数据
y = x[0]           # 获取第0行
y = x[:, 1]        # 获取第1列
y = x.view(3, 2)   # 重塑为3x2
y = x.transpose(1, 0)  # 转置

# 验证是否共享存储
assert same_storage(x, y)  # same_storage检查底层存储是否相同

# 注意：修改x会影响y
x[0][0] = 100
assert y[0][0] == 100  # 值也被修改
```
	并非所有视图都是“连续”的，当一个视图的数据在内存中不是按顺序存储时，它就是非连续的（Non-contiguous），比如说转置，显示方式变了，但底层数据没有重新排列。

如果需要对一个非连续张量进一步操作，可以先调用 .contiguous() 方法：
```Python
y = x.transpose(1, 0).contiguous().view(2, 3) 
assert not same_storage(x, y) # 现在创建了新存储
```
### 逐元素操作（Element-wise Operations）
几个常用的逐元素数学函数：
```Python
x = torch.tensor([1, 4, 9])
assert torch.equal(x.pow(2), torch.tensor([1, 16, 81]))    # 幂运算 (pow)，每个元素平方
assert torch.equal(x.sqrt(), torch.tensor([1, 2, 3]))      # 对每个元素开平方根
assert torch.equal(x.rsqrt(), torch.tensor([1, 1/2, 1/3])) # rsqrt 是 reciprocal of sqrt 的缩写，即对每个元素先开方再取倒数
```
两个实用的工具函数 triu（上三角）与 trill（下三角），它们在计算因果注意力掩码（casual attention mask）时非常有用。
```Python
# 实用操作：下三角矩阵（用于因果注意力掩码）
x = torch.ones(3, 3).tril()  # 创建一个3x3全1矩阵，然后取其下三角部分（含对角线）
# 结果：
# [[1, 0, 0],
#  [1, 1, 0],
#  [1, 1, 1]]
# 含义：第0个词只能看到第0个词，第1个词能看到第0、1个词，依此类推
```
```Python
x = torch.ones(3, 3).triu()   # 上三角（含对角线）
# [[1, 1, 1],
#  [0, 1, 1],
#  [0, 0, 1]]

y = torch.ones(3, 3).triu(1)  # 严格上三角（不含对角线），取反后即为因果掩码
# [[0, 1, 1],
#  [0, 0, 1],
#  [0, 0, 0]]
```
## 2.3 Einops 库对张量进行优化
PyTorch 中，张量维度通常是 [batch, sequence, hidden]。如果使用 PyTorch 原生的维度操作，需要时刻记住张量的维度顺序，很容易搞晕。尽管 Einops 增加了少量语法开销，但其清晰的维度命名显著降低了调试难度，特别是在复杂的模型架构中。
### 用 jaxtyping 命名维度
```Python
# 传统方式（容易写错维度顺序）
x = torch.ones(2, 2, 1, 3)  # batch seq heads hidden

# Jaxtyping方式（在类型注解中命名维度）
from jaxtyping import Float
x: Float[torch.Tensor, "batch seq heads hidden"] = torch.ones(2, 2, 1, 3)
```
### 用 einops.einsum 替代矩阵乘法 + 转置
```Python
from einops import einsum

# 定义两个张量
x: Float[torch.Tensor, "batch seq1 hidden"] = torch.ones(2, 3, 4)  
y: Float[torch.Tensor, "batch seq2 hidden"] = torch.ones(2, 3, 4)  

# 传统方式
z = x @ y.transpose(-2, -1)  # batch, sequence, sequence

# Einops方式
z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
# 未在输出中命名的维度（hidden）会被自动求和
```
这种方法会将没在输出中出现的 hidden 维度，自动对该维度求和，本质是爱因斯坦求和约定（Einstein summation）的直观实现。
### 用 einops.reduce 替代 mean(dim=...)
```Python
from einops import reduce

x: Float[torch.Tensor, "batch seq hidden"] = torch.ones(2, 3, 4)

# 传统写法
y = x.mean(dim=-1)  # 对最后一个维度 hidden 上求平均

# Einops 写法
y = reduce(x, "... hidden -> ...", "mean")
```
从 ... hidden 变成 ...，说明 hidden 维度被“聚合”了，聚合方式是 ”mean“ （也可用 “sum“，“max”等）。这种写法的好处是明确表达“我打算把哪个维度压缩掉“，语义清晰。
### 用 einops.rearrange 拆分/合并维度
这是 Einops 最强大的功能之一，用于处理扁平化与重组。
```Python
from einops import rearrange

# 情景：hidden维度实际上是heads×hidden1的扁平化表示
x: Float[torch.Tensor, "batch seq total_hidden"] = torch.ones(2, 3, 8)
w: Float[torch.Tensor, "hidden1 hidden2"] = torch.ones(4, 4)

# 拆分 total_hidden为 heads 和 hidden1
x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)
```
接下来就可以对拆分的部分进行矩阵乘法了：
```Python
x = einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2")
```
```Python
合并维度
x = rearrange(x, "... heads hidden2 -> ... (heads hidden2)")
```
# 3.  内存（Memory）
张量的内存占用由两个因素决定：
- 元素数量：张量的形状（如 4x8 矩阵有 32 个元素）
- 数据类型：每个元素占用的字节数
## 3.1 浮点类型
在 LLM 训练中，选择数据类型本质上是在做显存占用、计算速度与数值稳定性之间的权衡（Trade-off）。
### Float32（FP32 / 单精度浮点数）
- 规格：占用 4 字节（32 bits）。结构为：1 位符号位 + 8 位指数位 + 23 位尾数位。
- 地位：PyTorch 的默认数据类型，也是科学计算领域的“黄金标准”。
- 优点：数值精度高，动态范围大，训练最稳定，几乎不会出现数值溢出问题。
- 缺点：它占用的显存是 16 位格式的两倍，且在现代 GPU（如 H100）上的计算吞吐量远低于低精度格式。
- 炼丹用途：通常用于存储参数的主副本（Master Weights）和优化器状态（Optimizer States），以确保在梯度累积和参数更新时不会因为精度丢失而导致模型无法收敛。
### Float16（FP16 / 半精度浮点数）
- 规格：占用 2 字节（16 bits）。结构为：1 位符号位 + 5 位指数位 + 10 位尾数位。
- 优点：显存占用比 FP32 减少一半，计算速度快。
- 致命缺陷：**动态范围（Dynamic Range）太窄**。
  - 由于指数位只有 5 位，它无法表示非常小的数（会发生下溢 Underflow，直接变成 0）或非常大的数（会发生上溢 Overflow，变成 Infinity）。
- 用途：这是上一代 GPU（如 V100）混合精度训练的主流。为了解决溢出问题，必须使用复杂的损失函数（Loss Scaling）技术。目前在 LLM 训练中正逐渐被 BF16 取代。
### BFloat16（BF16 / Brain Floating Point）
- 规格：占用 2 字节（16 bits）。结构为：1 位符号为 + 8位指数位 + 7 位尾数位。
- 来源：由 Google Brain 专为深度学习设计。
- 设计逻辑：“要范围，不要精度”。
  - 深度学习模型（尤其是神经网络）对小数点后几位精度不敏感，但对数值的范围特别敏感。
  - BF16 直接截断了 FP32 的尾数，但保留了和 FP32 相同的 8 位指数位。
- 优点：
  - 拥有和 FP32 一样宽广的动态范围，不需要 Loss Scaling 也能稳定训练。
  - 显存占用和 FP16 一样少。
  - 在 A100/H100 等新硬件上计算速度极快。
- 用途：**当前 LLM 训练的绝对主流选择**。通常用于存储激活值（Activations）以及进行前向和反向传播的矩阵乘法计算。
	Percy 在课上解释说：“BF16 牺牲了精度来换取范围。对于深度学习来说，范围比精度重要得多，因为数值稳定性的主要威胁是溢出/下溢，而不是尾数精度不够。“
### FP8（8 位浮点数）
- 规格：占用 1 字节（8 bits）。
- 变体：
  - E4M3：4 位指数，3 位尾数（精度稍高，范围较小）。
  - E5M2：5 位指数，2 位尾数（范围稍大，精度更低）。
- 优点：极致的显存压缩和计算吞吐量。
- 硬件限制：仅在 NVIDIA H100 及更新的架构（配合 Transformer Engine）上才被原生支持。
- 炼丹用途：目前主要用于**推理阶段的量化（Quantization）**。虽然理论上可以用于训练（H100 支持），但由于精度极低，训练极其不稳定，属于比较前沿的研究领域。
### FP4 / NVFP4（4 位浮点数）
2025 年，NVIDIA 开发了 NVFP4，每个值仅 4 bits！
	Percy 在课上强调了一个关键细节：NVFP4 不是单纯每个值都只有 4 比特的动态范围。实际上每个值拥有 4 比特的自由度，但数据被划分成 block（块），每个块有一个共享的缩放因子（scale factor），可以整体放大或缩小。这样单个值实际拥有超过 4 比特的动态范围，但相邻值之间的比值仍然受限于 4 比特。“

不同精度的运算速度完全不同，Percy 特别强调，现在的 GPU 已经不太优化 FP32 了。
## 3.2 张量在内存中的存储机制
以一个 4x4 的二维张量作为例子：
![[Pasted image 20261103205201.png]]

以步长的方式来访问我们想要的位置上的数据：
```Python
# 步长（stride）：在每个维度上移动时跳过的元素数
assert x.stride(0) == 4  # 行维度：移动到下一行需跳过4个元素
assert x.stride(1) == 1  # 列维度：移动到下一列需跳过1个元素

# 计算元素位置：行索引r，列索引c
r, c = 1, 2
index = r * x.stride(0) + c * x.stride(1)  # 位置6
```

关键点：很多操作（如 view，transpose，slice）不会复制数据，只是修改了 stride。这是 Zero-copy，非常高效。
## 3.3 将张量（tensors）从 CPU 内存移动到 GPU 内存
默认情况下，张量是存储在 CPU 内存上的，但是，为了利用 GPU 的大规模并行性加速运算，我们需要将它们移至 GPU 中。
![[Pasted image 20261103205549.png]]

从 CPU 移至 GPU，需要以下几个步骤：
- 首先，我们先来看下我们当前机器上有没有GPU：
    ```python
    if not torch.cuda.is_available():
        return
    ```
- 接下来，获取GPU信息:
    ```python
    num_gpus = torch.cuda.device_count() # 获取GPU数量
    for i in range(num_gpus):
        properties = torch.cuda.get_device_properties(i) # 获取第i块GPU的属性（如型号、显存大小等）
    memory_allocated = torch.cuda.memory_allocated() # 获取当前已分配给PyTorch的显存总量
    ```
- 移动张量到GPU上有两种方法：
    - 方法一：移动现有张量
        ```python
        y = x.to("cuda:0") # 将张量x移动到编号为0的GPU上
        assert y.device == torch.device("cuda", 0) # 验证移动成功
        ```
    - 方法二：直接在GPU上创建张量
        ```python
        z = torch.zeros(32, 32, device="cuda:0") # 在GPU上直接创建一个32x32的零矩阵
        ```
        这种方法更高效，因为它避免了先在CPU创建再移动的过程。
- 最后，验证内存分配
    ```python
    new_memory_allocated = torch.cuda.memory_allocated() # 再次查询当前显存占用
    memory_used = new_memory_allocated - memory_allocated # 计算新增的显存占用
    assert memory_used == 2 * (32 * 32 * 4) # 验证：新增了2个32x32的矩阵，每个元素是4字节的float
    ```
# 4. 计算效率
## 4.1 浮点运算总数（FLOPs）
- FLOPs：执行的浮点运算总数，这是一个衡量计算量的指标。
- FLOP/s（每秒浮点运算）：硬件每秒能执行的浮点运算次数，也写作 FLOPS。这是一个衡量“速度”的指标。
### 线性模型的计算量
```Python
if torch.cuda.is_available():
    B = 16384  # Number of points
    D = 32768  # Dimension
    K = 8192   # Number of outputs
else:
    B = 1024
    D = 256
    K = 64
device = get_device()
x = torch.ones(B, D, device=device)
w = torch.randn(D, K, device=device)
y = x @ w
```
对于其中一个输出，需要 D 次乘法和 D-1 次加法，
因此，矩阵乘法的计算量：$总 FLOPs \approx 2 \times B \times D \times K$，其中 $D \times K$ 正好是线性层的参数数量，所以可以表示为 $总 FLOPs \approx 2 \times 数据量 \times 模型参数量$，
或者在语言模型中，$总 FLOPs \approx 2 \times tokens \times 模型参数量$。
	虽然相较于线性模型，Transformer 还有注意力、LayerNorm 等额外操作，但矩阵乘法还是占主导，所以这个公式对 Transformer 也基本成立。
### 其他操作的 FLOPs
- 逐元素操作（Element-wise Operations）：对张量中每个元素独立应用某个函数（如 x + 1、x.sqrt()、torch.relu(x) 等）。若张量形状为 m × n，则逐元素操作的 FLOPs 为 O(m × n)。
- 张量加法（Tensor Addition）：两个相同形状张量的对应元素相加，如 z = x + y。若 x 和 y 均为 m × n，则总 FLOPs = m × n（每个元素 1 次加法）

FLOPs 只是一个衡量计算量的数字，如何将 FLOPs 转化为实际可运行时间呢？
```Python
actual_time = time_matmul(x, w)  # 实际执行矩阵乘法所需的时间（秒）
actual_flop_per_sec = actual_num_flops / actual_time  # 实际每秒能完成多少次浮点运算
```
另外，FLOP/s 的性能很大程度上取决于数据类型。
### 实际计时与 Benchmarking
```Python
def benchmark(func, num_trials=5):
    if torch.cuda.is_available():
        torch.cuda.synchronize()   # 确保之前的 CUDA 操作已完成
    # 运行 num_trials 次并取平均
    ...
    torch.cuda.synchronize()   # 操作后的同步点
```
	torch.cuda.synchronize() 是必须的：GPU 操作是异步的。此外通常需要多次运行取平均值来减少噪声。
## 4.2 MFU（Model FLOPs Utilization）
$$MFU = \frac{实测 FLOPS}{硬件理论峰值 FLOPS}$$
- MFU $\geq$ 0.5：被认为是相当不错的性能。
- MFU 接近 1.0：非常难达到。
## 4.3 PyTorch 中的梯度计算
### 梯度反向传播
```Python
x = torch.tensor([1., 2, 3]) # 输入数据，不需要计算梯度
w = torch.tensor([1., 1, 1], requires_grad=True) # 模型参数，需要计算梯度
pred_y = x @ w # 预测值（矩阵乘法） 
loss = 0.5 * (pred_y - 5).pow(2) # 计算损失值
```
requires_grad=True 这个参数它告诉 PyTorch “请为这个张量 w 构建计算图，并在反向传播时计算它的梯度。“

反向传播：
```Python
loss.backward() # 触发反向传播
assert torch.equal(w.grad, torch.tensor([1, 2, 3])) # 验证模型参数 w 的梯度是否计算正确
```
调用 loss.backward() 后，PyTorch 会从 loss 开始，沿着计算图回溯，自动计算出每个需要梯度的张量的梯度。
### 反向传播（梯度）所需的浮点运算次数
```Python
x = torch.ones(B, D, device=device) 
w1 = torch.randn(D, D, device=device, requires_grad=True) 
w2 = torch.randn(D, K, device=device, requires_grad=True) 

Model: x --w1--> h1 --w2--> h2 -> loss 

h1 = x @ w1 # 第一层：输入x乘以权重w1得到隐状态h1 
h2 = h1 @ w2     # 第二层：h1乘以权重w2得到输出h2 
loss = h2.pow(2).mean() # 计算损失（均方误差）
```
$W_2.grad = h1^T @ h_2.grad$，形状为（D，K），计算的 FLOPs 为：$2 \times B \times D \times K$。
$h_1.grad = h2.grad @ W_2^T$，形状为（B，D），计算的 FLOPs 为：$2 \times B \times D \times K$。
	在第二层，我们注意到计算的 FLOPs 确实为前向的两倍，但是第一层 x 未设置 requires_grad=True，所以并不恰好是前向的两倍。

通过上述计算，我们得出了一个普适性的结论：在深度学习中，**训练一次（前向+反向）的总 FLOPs 约等于 6 倍的“数据点数量 × 参数数量”**。
# 5. 模型构建与训练基础
## 5.1 参数初始化（Initialization）
### 参数初始化的重要性及常用方法
如果直接使用 torch.randn 进行标准高斯分布初始化参数，随着层数的变深，数值会变得非常大（爆炸）或非常小（消失）。
	假设输入和权重都近似满足：均值为0，标准差为1，作适当的独立性假设，每个乘积 $w_i x_i$ 的方差为 1，100项相加后方差就会来到 100。如果再接很多层，放大效应就会累积，像这个层标准差就乘了 10 倍，导致爆炸，反过来就是消失。

为了解决这个问题，可以采用 Xavier 初始化，通过除以 $\sqrt{输入维度}$ 来缩放权重，并且使用截断正态分布（truncated normal distribution），将生成的随机数限制在一个合理的范围之内。
```Python
w = nn.Parameter(nn.init.trunc_normal_(torch.empty(input_dim, output_dim), 
                                      std=1 / np.sqrt(input_dim), 
                                      a=-3, b=3))
```
## 5.2 使用 pytorch 自定义模型
### 定义模型结构
定义一个名为 Cruncher 的自定义模型类，包含 num_layers 个隐藏层和一个输出层。
主要使用的是 Linear 类，每个 Linear 层的输入和输出维度都是 dim，这意味着它们是“恒等”变换，但内部的权重是可学习的。
```Python
D = 64 # Dimension
num_layers = 2
model = Cruncher(dim=D, num_layers=num_layers)
```
```Python
class Cruncher(nn.Module):
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            Linear(dim, dim)
            for i in range(num_layers)
        ])
        self.final = Linear(dim, 1)  # 创建一个最终的 Linear 层，它将 dim 维的特征映射到 1 维的标量输出
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply linear layers
        B, D = x.size()
        for layer in self.layers:
            x = layer(x)
        # Apply final head
        x = self.final(x)
        assert x.size() == torch.Size([B, 1])
        # Remove the last dimension
        x = x.squeeze(-1) # 移除最后一个维度（即大小为1的那个维度），使得最终输出是一个一维张量 (B,)，更符合我们对“预测值”的直观理解（例如，对每个样本输出一个分数）
        assert x.size() == torch.Size([B])
        return x
```
### 检查模型参数
使用 model.state_dict() 将返回一个字典，包含了模型中所有可学习参数（nn.Parameter）的名称和值。
```Python
param_sizes = [
    (name, param.numel())  # param.numel()：返回该参数张量中的元素总数
    for name, param in model.state_dict().items()
]
assert param_sizes == [
    ("layers.0.weight", D * D),
    ("layers.1.weight", D * D),
    ("final.weight", D),
]
```
### 将模型移至 GPU
get_device() 函数会自动选择可用的 GPU 或 CPU。
```Python
device = get_device()
model = model.to(device)
```
### 在数据上运行模型
```Python
B = 8 # Batch size
x = torch.randn(B, D, device=device)
y = model(x)
assert y.size() == torch.Size([B])
```
## 5.3 管理随机性（Randomness）
### 随机性的来源
- 参数初始化：模型权重通常从随机分布（如正态分布）中采样。
- Dropout：训练时随机“关闭”一部分神经元。
- 数据排序：数据加载器通常会打乱数据顺序。
- 其他：如数据增强、优化器中的动量等。
### 设置随机种子
有三个库需要分别设置随机种子，通常在程序开始时一次行全部设置。
大多数情况下，为了确保完全的可重现性，最简单、最安全的做法就是将这三个种子都设为同一个值。

PyTorch：
```Python
seed = 0
torch.manual_seed(seed)
```
NumPy（许多数据预处理，比如数据加载和切分等，都依赖 NumPy 的随机函数）：
```Python
import numpy as np
np.random.seed(seed)
```
Python：
```Python
import random
random.seed(seed)
```
## 5.4 数据加载
在语言建模中，输入数据通常是经过分词器（tokenizer）处理后的整数序列（例如，句子“Hello world“ 可能被编码为 [1, 2]。
但是对于 LLaMA 这样高达 2.8 TB 的超大数据集，不可能将其全部加载到内存中。解决方法是使用 numpy.memmap 创建一个“内存映射”对象。
```Python
data = np.memmap("data.npy", dtype=np.int32) # 创建内存映射
assert np.array_equal(data, orig_data) # 验证加载的数据是否正确
```

在实际训练中，我们不会直接操作整个数据集，而是通过一个“数据加载器”来生成所需的批次。
```Python
def get_batch(data: np.array, batch_size: int, sequence_length: int, device: str) -> torch.Tensor:
    # 随机采样起始位置
    start_indices = torch.randint(len(data) - sequence_length, (batch_size,)) # 使用 torch.randint 在 [0, len(data) - sequence_length] 范围内随机生成 batch_size 个起始索引。这样可以确保每个序列都能完整地从数据中截取出来，不会越界
    assert start_indices.size() == torch.Size([batch_size]) # 断言验证了生成的索引数量正确

    # 根据起始索引提取数据
    x = torch.tensor([data[start:start + sequence_length] for start in start_indices]) # 遍历 start_indices 中的每一个起始位置 start，从 data 中切片出长度为 sequence_length 的子序列。最后将这些子序列转换成一个 PyTorch 张量 x，其形状为 (batch_size, sequence_length)
    assert x.size() == torch.Size([batch_size, sequence_length])

    # 固定内存（Pinned Memory）优化
    if torch.cuda.is_available():
        x = x.pin_memory()

    # 异步数据传输
    x = x.to(device, non_blocking=True)
```
调用 get_batch() 函数会从 data 中随机采样 B 个起始位置，然后截取每个位置后长度为 L 的序列，最终返回一个形状为（B，L）的张量 x。
### 固定内存（Pinned Memory）优化：
默认情况下 CPU 上的张量存储在“分页内存“（paged memory）中。当需要将数据从 CPU 传输到 GPU 时，操作系统必须先将数据复制到一个“非分页”的内存区域，然后再通过 PCIe 总线发送给 GPU。这个过程是同步的，会阻塞当前进程，通过调用 .pin_memory()，我们可以将 CPU 张量显式标记为“固定”，这使得 GPU 驱动程序可以直接访问这块内存，无需额外的拷贝步骤。
### 异步传输：
设置 non_blocking=True 参数，这告诉 PyTorch 数据传输可以在后台异步进行，不会阻塞当前的 Python 线程。
通过结合“固定内存”和“异步传输”，我们可以实现以下高效的流水线操作：
- 在 GPU 上处理当前批次的数据。
- 在 CPU 上同时加载下一个批次的数据（例如，从硬盘读取或从内存映射文件中加载）。
这种并行化可以显著减少 GPU 的空闲等待时间，从而大幅提升整体训练吞吐量。
## 5.5 优化器（Optimizer）
**常用优化器介绍**
- SGD (随机梯度下降)：最基础的优化器，直接用学习率乘以梯度更新参数。
- Momentum (动量法)：在 SGD 基础上增加了一个“动量”项，即梯度的指数移动平均值，有助于加速收敛并减少震荡。
- AdaGrad：根据历史梯度的平方值来调整每个参数的学习率，对稀疏特征更友好。
- RMSProp：对 AdaGrad 的改进，使用梯度平方的指数加权平均代替简单累加，避免学习率过早衰减。
- Adam：融合了 RMSProp 和 Momentum 的思想，是目前最流行的优化器。

AdaGrad：
```Python
class AdaGrad(torch.optim.Optimizer):
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad.data
                # 获取状态（梯度平方和）
                state = self.state[p]
                if 'sum_squared_grad' not in state:
                    state['sum_squared_grad'] = torch.zeros_like(p.data)
                
                # 更新状态：累加梯度的平方
                state['sum_squared_grad'] += grad ** 2
                
                # 更新参数：除以 根号下(状态)
                std = state['sum_squared_grad'].sqrt() + 1e-10
                p.data -= group['lr'] * grad / std
```
### 优化器的使用
```Python
# 实例化优化器
optimizer = AdaGrad(model.parameters(), lr=0.01) # model.parameters()：将模型中所有可学习的参数传递给优化器

# 计算梯度
loss.backward() # 计算损失函数对所有参数的梯度

# 执行一步更新
optimizer.step() # 根据梯度和优化器内部状态，更新模型参数
```
释放内存：
```Python
optimizer.zero_grad(set_to_none=True)
```
## 5.6 资源合算
### 内存占用分析
对于一个深度线性模型，总内存需求由四部分组成：
- 参数 (Parameters)：模型中所有可学习权重的数量。
- 激活值 (Activations)：前向传播过程中产生的中间结果，需要保存下来用于反向传播。
- 梯度 (Gradients)：反向传播计算出的梯度，其数量与参数相同。
- 优化器状态 (Optimizer States)：优化器维护的额外状态信息（如 AdaGrad 的 g2），其数量也与参数相同。
假设所有数据都使用 float32 格式（每个元素占 4 字节），则总内存为：
```Python
total_memory = 4 * (num_parameters + num_activations + num_gradients + num_optimizer_states)
```
- 参数数量：(D * D * num_layers) + D
- 激活值数量：B * D * num_layers （每个批次、每个样本、每层都需要保存激活值）
- 梯度数量：num_parameters
- 优化器状态数量：num_parameters
## 5.7 训练循环：整合所有组件
```Python
def train_loop():
    # 数据生成
    D = 16
    true_w = torch.arange(D, dtype=torch.float32, device=get_device()) # 创建一个真实的权重向量 [0, 1, 2, ..., 15]

    # 数据加载器，用于生成训练批次
    def get_batch(B: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(B, D).to(get_device()) # 从标准正态分布中随机采样 B 个 D 维的输入样本
        true_y = x @ true_w # 根据真实的权重 true_w 计算出对应的“真实标签”，即 y = x @ w_true
        return (x, true_y) # 作为模型的输入和目标输出

    # 执行训练
    train("simple", get_batch, D=D, num_layers=0, B=4, num_train_steps=10, lr=0.01)
    # 超参数调优（hyperparameter tuning）
    train("simple", get_batch, D=D, num_layers=0, B=4, num_train_steps=10, lr=0.1) # 通过改变学习率 lr 从 0.01 到 0.1，观察不同学习率对训练效果的影响
```

train()：
```Python
def train(name: str, get_batch,
          D: int, num_layers: int,
          B: int, num_train_steps: int, lr: float):

    # 初始化模型和优化器
    model = Cruncher(dim=D, num_layers=0).to(get_device())
    optimizer = SGD(model.parameters(), lr=lr)

    # 主训练循环
    for t in range(num_train_steps): # 循环执行 num_train_steps 次，每次迭代称为一个“训练步”
        # 获取数据
        x, y = get_batch(B=B) # 获取当前批次的输入 x 和目标输出 y

        # 前向传播（计算损失）
        pred_y = model(x)
        loss = F.mse_loss(pred_y, y) # 计算预测值 pred_y 和真实值 y 之间的均方误差（Mean Squared Error），作为损失值 loss

        # 反向传播（计算梯度）
        loss.backward() # 调用 loss.backward()，触发自动微分机制，计算损失函数对所有模型参数的梯度，并将这些梯度存储在 .grad 属性中
        
        # 更新参数
        optimizer.step() # 根据计算出的梯度和优化器的更新规则（这里是 SGD），更新模型参数
        optimizer.zero_grad(set_to_none=True) # 清空上一步计算得到的梯度，为下一次迭代做准备
```
## 5.8 检查点（Checkpointing）
大型语言模型的训练耗时极长，过程中几乎不可避免地会发生崩溃或中断，我们需要一种机制来保存进度，以便在中断后能够从中断点恢复，而不是从头开始。
### 保存检查点
```Python
# 1. 创建一个字典，包含模型和优化器的状态
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
}

# 2. 使用 torch.save 将字典序列化并保存到磁盘文件
torch.save(checkpoint, "model_checkpoint.pt") # 文件名通常使用 .pt 或 .pth 后缀
```
### 加载检查点
```Python
# 1. 从磁盘文件加载保存的字典
loaded_checkpoint = torch.load("model_checkpoint.pt")

# 2. （后续步骤）将状态加载回模型和优化器
# model.load_state_dict(loaded_checkpoint["model"])
# optimizer.load_state_dict(loaded_checkpoint["optimizer"])
```
## 5.9 混合精度训练 (Mixed Precision)
- 前向传播（Forward Pass）：使用 bfloat16 或 fp8。这包括所有中间激活值（activations）。因为激活值通常不需要极高的精度，使用低精度可以显著节省内存。
- 其余部分：使用 float32。这包括模型参数（parameters）、梯度（gradients）以及优化器状态（optimizer states）。这些是训练的核心，需要更高的精度来保证数值稳定性和收敛性。
- PyTorch 的 AMP 库 (Automatic Mixed Precision)：PyTorch 提供了自动混合精度（AMP）库，会自动将安全的操作（如矩阵乘法）转为 bf16，将危险操作（如 exp、softmax）保留为 fp32
# 6. 算术强度与 Roofline 分析
## 6.1 不只有“算”
Percy 画了一张卡通图来简化 GPU 的工作原理：
![[Pasted image 20261104105039.png]]
1. **从 HBM（高带宽内存）发送输入到计算核心**
2. **执行计算**
3. **将输出从计算核心送回 HBM**
"你不只是在做一堆 MatMul 然后看它们花了多长时间。你还得把数据搬来搬去。"

总时间取决于两个因素：
- **加速器速度**（FLOP/s）- 算得有多快
- **内存带宽**（Bytes/s）- 数据搬得有多快
## 6.2 ReLU 的算术强度分析
内存移动（Bytes）：
- 读入 x：2 × n（bf16 是 2 字节/元素）
- 写出 y：2 × n
- 总计：4n
计算量（FLOPs）：
- n 次比较（max(x, 0)）
- 共 n FLOPs
**通信时间** = 4n / (3.35 × 10¹²) ≈ 1.2 × 10⁻⁶ 秒 **计算时间** = n / (989.5 × 10¹²) ≈ 1.0 × 10⁻⁹ 秒
## 6.3 算数强度
加速器强度 = FLOP/s / Bytes/s → H100 约 295 FLOP/byte
算术强度 = FLOPs / Bytes → 该操作每搬运 1 字节能做多少 FLOP

- 算术强度 < 加速器强度 → **Memory-bound**（瓶颈在数据传输）
- 算术强度 > 加速器强度 → **Compute-bound**（瓶颈在计算）
	"对于 H100，加速器强度大约是 295。这个数字值得记一下——对于 bf16 来说，每搬运 1 字节你需要做大约 300 次浮点运算才能摆脱内存瓶颈。"
## 6.4 操作的算术对比
|操作|FLOPs|Bytes|算术强度|瓶颈|
|---|---|---|---|---|
|**ReLU**|n|4n|≈ 0.25|Memory-bound|
|**GeLU**|20n|4n|≈ 5|Memory-bound|
|**Dot Product**|2n-1|4n+2|≈ 0.5|Memory-bound|
|**Matrix-Vector**|n(2n-1)|2n²+4n|≈ 1|Memory-bound|
|**Matrix-Matrix**|n²(2n-1)|6n²|≈ n/3 ≈ 340|**Compute-bound**|
**关键直觉**：
1. **ReLU vs GeLU**：尽管 GeLU 公式包含 tanh、多项式等复杂运算，FLOPs 是 ReLU 的 20 倍，但两者在 GPU 上的实际耗时几乎一样——因为它们都是 memory-bound，瓶颈在数据搬运而不是计算。"你以为 GeLU 很复杂所以肯定慢，但实际上它不构成瓶颈所在。"
2. **算术强度 = 0.25 意味着什么**：Percy 说"如果有人告诉你一个算法的算术强度是 0.25，你应该立刻说'这太糟糕了'。"
3. **矩阵乘法的 O(n) 优势**：对于 n×n 矩阵乘法，搬运了 O(n²) 的数据，但做了 O(n³) 的计算，所以算术强度是 O(n)。矩阵越大，算术强度越高。
4. **为什么大 batch size 很重要**："当你在加速器强度以下时，把矩阵变小并不会让速度变快——因为瓶颈不在计算。只有当你超过加速器强度的转折点，你才真正在**饱和使用你的 GPU**。"
## 6.5 Roofline
![[Pasted image 20261104105946.png]]
