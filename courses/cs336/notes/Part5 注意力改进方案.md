从 《Attention Is All You Need》提出以来，Transformer 逐渐成为现代大语言模型的核心机制。但随着参数规模与上下文窗口持续扩展，原始 Transformer 架构在计算复杂度、显存开销和模型容量的瓶颈日益突出。近年的研究主要围绕两条路线：
- 扩展上下文扩展能力：在有限资源下高效处理更长序列。代表工作包括：FlashAttention 优化 Attention Kernel 提升 GPU 利用率、PagedAttention 改进 KV Cache 内存管理、MLA 压缩 Key-Value 表示降低缓存开销以及 Hybrid Attention 兼顾计算效率与建模能力。共同的目标是降低上下文带来的计算与存储代价，同时缓解性能退化。
- 扩展模型容量：在计算量基本不变的前提下提升表达能力。MoE 通过动态路由仅激活少量专家参与计算，以近乎恒定的 FLOPs 大幅扩展参数规模，已成为主流大模型的关键架构选择。
# 1. Attention 的改进方法
![[Pasted image 20261105224551.png]]
随着处理的上下文长度增加，标准 Attention 机制面临多重瓶颈：
- 注意注意力分数矩阵的计算与存储随序列长度呈二次增长，同时 KV Cache 随上下文**线性**累积，两者共同加剧显存压力，导致计算处理速度急剧下降；
- 注意力分数在长距离上的分散与位置编码的外推漂移，使模型对长文本的建模能力逐渐衰减——这一现象在层数更深的模型中尤为突出。
## 1.1 Hybrid Attention
对于一个 Transformer 层，计算流程为 $Softmax(\frac{QK^T}{\sqrt{d}})V \to MLP$，对应的时间复杂度为：
$$O(n^2d + n^2 + n^2d) \approx O(n^2d)$$
具体的：

| 操作                       | 怎么数                                                       | 复杂度       |
| ------------------------ | --------------------------------------------------------- | --------- |
| $QK^\top$ 打分             | $[n,d]\times[d,n]\to[n,n]$。有 $n^2$ 个分数，每个分数是一次 $d$ 维点积    | $O(n^2d)$ |
| 除以 $\sqrt{d}$，再做 Softmax | 打分矩阵有 $n$ 行，每行有 $n$ 个数，逐行处理                               | $O(n^2)$  |
| 用注意力权重乘 $V$              | $[n,n]\times[n,d]\to[n,d]$。每个 Query 都对 $n$ 个 $d$ 维向量做加权求和 | $O(n^2d)$ |

标准 Attention 的计算量会随上下文序列呈二次增长，那么线性注意力机制的情况是怎么样的呢？
$$S_t = S_{t-1} + k_tv_t^T,\quad y_t = W_tS_t$$
其中，$S_{t-1}$ 表示截至第 t-1 时刻的隐藏状态压缩，这里将所有 k 和 v 的外积相加，所以到时候直接与 q 相乘即可，所以不用从头开始读取 cache。

但是，部分的 LLM 仍然以标准 Attention 为主，主要原因有两点：
- 训练效率：标准 Attention 可以通过矩阵并行计算一次性处理整个序列，硬件利用率极高；线性注意力机制虽然可以通过并行扫描等技术加速训练，但在当前硬件架构上的并行效率和吞吐量相较于标准 Attention 仍有差距。
- 表达能力：标准 Attention 能对序列中任意两个 token 进行精确的全局交互，信息损失较少；线性注意力机制将所有历史压缩为固定大小的状态，不可避免地会丢失细粒度的上下文信息，在需要更多精确的回忆（比如复杂推理、长距离事实检索）的任务上表现较弱。

Hybrid Attention 结合两点，在模型的部分层使用标准 Attention 以保留全局精确交互能力，在其余层使用线性注意力机制以降低整体计算和显存开销。
### 1. Nemotron 3 Super
近期，这一思路重新受到关注，例如 Nemotron 3 Super、Kimi K3 等模型相继采用了 Hybrid Attention。
Nemotron 3 Super 采用了 Mamba 与标准 Attention 交替组合的架构：Mamba 层负责高效的序列建模，以线性复杂度处理长距离依赖；标准 Attention层 则补齐线性注意力机制在精确全局交互上的不足。两者相互配合，在保持推理效率的同时提升了整体的语义建模能力。
### 2. KDA
Kimi 团队提出了 KDA ，在线性注意力的递推公式中引入 $Diag(\alpha_t)$对隐藏状态施加逐维度的衰减控制，从而更细粒度地管理对历史信息的遗忘与保留，同时增强对序列位置关系的感知能力。
对应的推理表示为：
$$S_t = (II - \beta k_tk_t^T)Diag(\alpha_t)S_{t-1}+\beta k_tv_t^T$$
它是一种线性注意力机制，其核心思想是在 Gated DeltaNet 的基础上，引入了按不同隐藏维度的可学习遗忘机制（$Diag(\alpha_t)$）。
## 1.2 DeepSeek Sparse Attention
Hybrid Attention 通过替换部分层的注意力机制来降低整体复杂度，DeepSeek 团队提出了另一种思路：在标准 Attention 计算之前引入轻量级索引器，预测每个 Query 最值得关注的少量 Key Token。
所以可以将复杂度从 $O(n^2)$ 降低到 $O(nk)$ 。

具体的，
对于输入的 Query 和 Key，轻量级索引器首先计算每个 Key Token 相对于 Query 的相关性分数，随后根据相关性分数选取前 k 个最相关的候选 token，得到候选集合。最后，仅在候选集合上执行标准的 Scaled Dot-Product Attention。

在训练上，DSA 采用两阶段训练方式：
- 第一阶段使用较短上下文并采用标准 Attention 进行预训练，使模型充分学习基础语言建模能力。第二阶段则在第一阶段模型参数的基础上引入轻量级索引器，并切换至长上下文继续训练，使模型逐步适应稀疏注意力计算模式，同时学习有效的候选 token 选择策略。
- 为减少索引器随机初始化对模型性能造成较大影响，在第二阶段开始时，通常先冰冻住主模型参数，仅优化索引器，并以标准 Attention 的注意力分布作为教师信号，通过 **KL散度** 损失对齐稀疏注意力与标准注意力的分布。

采用分阶段训练而非直接联合训练的主要原因在于，语言建模和稀疏索引学习属于两个优化目标。
# 2. MoE 原理分析
MoE 是一种稀疏激活架构，其基本思想是将 Transformer 中原本的单一 FFN 替换为多个并行的专家网络（Experts），并引入一个路由器来决定每个输入 token 应该由哪些专家进行处理。在一次前向传播过程中，每个 token 通常只会激活少量专家（如 Top-1 或 Top-2），其余专家保持不参与计算。因此，虽然整个模型拥有庞大的参数规模，但每次计算仅激活其中一小部分参数，从而实现了模型容量大、计算开销相对较低的特点。
	值得注意的是——多数研究表明，MoE 架构在参数规模较大、数据和计算资源充足时优势最为明显；在小规模或资源受限的环境下，其表现可能不如对应的稠密（Dense）模型。
## 2.1 路由机制与负载均衡
在 MoE 模型中，路由机制也称门控机制，负责在每次前向传播时从全部专家中选择少量专家参与计算。
当前主流的路由方式是基于可学习门控得分的 Top-K 路由，并在此基础上衍生出两种执行策略：TC 与 EC，两者都需要依赖可学习的门控得分机制，并通常会配合负载均衡策略以避免专家不均衡。

假设一共有 N 个专家，输入为 x，门控函数为 $G$ ：用于决定每个专家的权重，$E_i$ 表示第 i 个专家的输出，则 TC 与 EC 的通用门控核心计算公式为：
$$y = \sum_{i \in T}G_i(x)E_i(x)$$
其中 $G_i(x)$ 的计算包含两个步骤：
- 打分：计算 $h(x) = x \cdot W_g$
- 稀疏化：仅保留分数最高的 k 个专家，并对这些分数进行 Softmax 归一化。未被选中的 N - k 个专家权重被强制清零。
	在 TC 模式下，每个 token 选择最合适处理的专家；在 EC 模式下，每个专家主动选择最适合处理的 token。
### 简易实现 TC MoE
#### Step1. 定义专家网络
```Python
class Expert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(
            # 维度升高
            nn.Linear(dim, dim * 4),
            # 非线性激活，提高表达能力
            nn.ReLU(),
            # 还原到初始维度
            nn.Linear(dim * 4, dim)  
        )

    def forward(self, x):
        return self.ffn(x)  # 前向传播
```
#### Step2. 定义 TC MoE 网络
```Python
class TC_MoE(nn.Module):
    def __init__(self, dim, num_experts, k):
        super().__init__()
        # 设置专家数量
        self.num_experts = num_experts
        # 设置每个token选用的专家数量
        self.k = k
        # 路由器：将输入映射到专家特征空间
        self.router = nn.Linear(dim, num_experts)
        # 创建专家模块列表（每个专家是独立的）
        self.experts = nn.ModuleList([Expert(dim) for _ in range(num_experts)])
    def forward(self, x, tokens=None, verbose=False):
        # 获取批量大小和特征维度
        B, D = x.shape
        # 计算每个专家对每个token的分数，使用softmax得到概率分布
        gate_scores = F.softmax(self.router(x), dim=-1)  # gate_scores: [B, E]

        # token选取分数最高的k个专家及其分数
        # topk_scores: [B, k]（对应被选中的专家概率值）
        # topk_idx:    [B, k]（对应被选中的专家索引）
        topk_scores, topk_idx = gate_scores.topk(self.k, dim=-1)
 
        # 初始化输出张量与输入同形状
        out = torch.zeros_like(x)

        # 每个token对应的每一个top-k位置单独处理（同一个token可能被不同专家处理）
        for i in range(self.k):
            # B表示处理的token总数
            # expert_ids表示每个token在第i个Top-K选择的专家编号，形状：[B]
            expert_ids = topk_idx[:, i]
            # expert_weight表示第每个token在第i个Top-K选择上专家所占的权重，形状：[B]
            expert_weight = topk_scores[:, i]

            # 用于累加当前第i个选择位置上所有专家的输出
            expert_output = torch.zeros_like(x)

            # 遍历所有专家，让对应专家处理被分配给它的token
            # e_id表示对应Top-K专家处理的token索引值
            for e_id, expert in enumerate(self.experts):
                # 创建掩码当token在第i个选择位置的专家索引等于当前专家e_id时为1，否则为0
                # mask形状为[B, 1]，用于在计算专家网络前把不属于该专家的token置0
                mask = (expert_ids == e_id).float().unsqueeze(1)

                # mask.sum()表示属于该专家的token数量；若为0表示该专家在本轮没有任务
                if mask.sum() == 0:
                    continue

                # 只把属于该专家的token送入该专家的前馈网络
                # 注意：这里采用x * mask的方式，能保持张量形状一致并保留反向传播路径
                expert_output += expert(x * mask)

            # 将第i个选择位置上专家的输出按对应权重加权并累加到最终out
            # expert_weight.unsqueeze(1)变为[B, 1]以便广播乘到[B, D]
            out += expert_output * expert_weight.unsqueeze(1)

        # out：每个token在Top-K专家上的加权聚合的向量表示
        return out
```
### 简易实现 EC MoE
```Python
class EC_MoE(nn.Module):
    def __init__(self, dim, num_experts, k):
        super().__init__()
        # 专家的总数量
        self.num_experts = num_experts
        # 每个专家最多选多少个token  
        self.k = k
        # 用于给每个token输出E个专家得分的路由器                
        self.router = nn.Linear(dim, num_experts)  
        self.experts = nn.ModuleList([Expert(dim) for _ in range(num_experts)])  
    def forward(self, x, tokens=None, verbose=False):
        # 获取输入token的数量B_total和维度D
        # B_total表示所有token的总数（批次 × token数）
        B_total, D = x.shape

        # 路由器计算每个token、每个专家的匹配得分输出维度：[B_total, num_experts]
        # softmax确保所有专家得分加起来为1
        gate_scores = F.softmax(self.router(x), dim=-1)

        # EC模式： “专家挑token”
        # 转置后变成[num_experts, B_total]
        # scores_T[e][t] = 第e个专家对第t个token的评分
        scores_T = gate_scores.transpose(0, 1)

        # 每个专家从所有token中挑选top-k个最相关的token
        # topk_idx: 每个专家选中Top-K token的索引
        # topk_scores: 对应的路由得分
        # 维度：[num_experts, k]
        topk_scores, topk_idx = scores_T.topk(min(self.k, B_total), dim=-1)

        # dispatch_weights大小：[B_total, num_experts]
        # 初始化dispatch_weights
        dispatch_weights = x.new_zeros((B_total, self.num_experts))

        # 对每个专家e，把top-k token的得分写入对应位置
        for e in range(self.num_experts):
            # topk_idx[e]是一个Top-K token索引列表
            # topk_scores[e]是Top-K token的得分
            # 填写dispatch_weights：每个专家对各个token评分归一化处理
            for t_idx, s in zip(topk_idx[e].tolist(), topk_scores[e].tolist()):
                dispatch_weights[t_idx, e] = s

        # 初始化输出out，与输入x大小相同
        out = torch.zeros_like(x)

        # 对每个专家进行前向计算
        for e_id, expert in enumerate(self.experts):

            # mask: 这个专家是否选择了该token
            # mask[t] == 1 → token t被这个专家选中
            # 维度：[B_total, 1]
            mask = (dispatch_weights[:, e_id] > 0).float().unsqueeze(1)

            # 如果专家没有选择任何token，则跳过计算
            if mask.sum() == 0:
                continue

            # 确保每个专家只处理其选中的Top-K token
            # mask会把不属于该专家的token设置为0，不同专家可能会处理相同的token
            expert_out = expert(x * mask)

            # 将专家输出按其权重加回到最终输出中
            # dispatch_weights[:, e_id]是每一组Top-K token对这个专家的权重
            out += expert_out * dispatch_weights[:, e_id].unsqueeze(1)
        return out
```

两种路由的差异：
- 在 TC 模式下，每个 token 主动选择自己最合适的 Top-K 专家。优势是每个 token 都会被至少尝试分配到专家上，因此语义完整性较高，减少了信息丢失的风险。缺点问题是专家负载不均衡，少数“热门”专家会处理绝大部分token，得到充分训练并显著优于其他专家，而大量“冷门”专家长期闲置能力停滞。
- 在 EC 模式下，每个专家主动从所有 token 中挑选自己最想处理的 Top-K，从而显著缓解或消除专家负载不均衡，有利于专家能力的均衡提升。但代价是部分 token 可能完全未被任何专家处理（被丢弃），导致语义信息缺失或上下文片段被跳过。
## 2.2 MoE 变体
实际训练中，有两个常见问题会阻碍专家真正形成“专业领域”：
- 知识混合：分配给某个专家的token可能多样化即涵盖多种不同类型的知识。就像一位老师被要求同时教数学、历史和美术，他很难在自己的课堂上把每门课都讲得透彻。
- 不同专家处理的 token 可能存在重叠的知识需求。结果就像几位老师都在准备相同的教材，每个人都在重复劳动，无法凸显各自的“专业特长”，也导致专家之间缺乏明确的“专业分工”。
### DeepSpeed-MoE
### Switch Transformer
## 2.3 混合专家与稠密模型
MoE 相较于传统稠密模型的优势是它可以**扩大模型参数规模的同时保持计算量基本不变**，从而显著提升模型的表示能力与性能；并且由于 MoE 的专家是**稀疏激活**的，每次仅有少量专家参与计算，因此各专家通常是前馈网络可以作为独立模块分布在不同设备上。路由器只需根据输入将对应的token 发送到相应设备，计算便在该专家所在设备上独立完成。这种天然的结构切分方式使 MoE 能实现高效的**专家级并行**，成为构建超大规模模型时必不可少的并行化策略，也是现代大模型在多机多卡环境下突破容量与性能瓶颈的重要基础。

在 MoE 研究中，常见两条实践路径：
 - 稠密 $\to$ 稀疏：把已训练好的稠密模型 upcycling 为 MoE，以复用先前的训练成果与权重；
 - 从零开始训练 MoE：从随机或专门初始化开始训练 MoE，使专家与路由器从头共同演化。
# 3. MoE 的应用
MoE 不仅用于 Transformer 技术，在 Transformer 之外也被大量应用：
- 在**CNN**中作为动态卷积提升视觉建模的多样性；
- 在**语音识别**中让不同专家专注于不同音素或噪声条件；
- 在**推荐系统**解决多任务排序问题；
- 在**强化学习**中分解为多策略、多技能专家；
- 在**多模态模型**中实现跨模态专家协作。
## 3.1 简易 MoE + LLM
### Step1. 构建字节级分词器
```Python
class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 259
        self.bos = 256 # 序列开始，告诉LLM一个独立的文本片段或输入样本从这里开始。
        self.eos = 257 # 序列结束，告诉LLM一个文本片段到这里结束。
        self.pad = 258 # 填充，在模型训练或推理时，通常需要将多条长短不一的文本组成一个批次。
        # <pad>会被添加到较短序列的末尾使批次中所有序列长度一致，便于高效的矩阵运算。
```
大小为 256 + 3：
- 基础字节编码：数量 256 个，它们代表了计算机中所有可能的单字节值从 0 到 255。这种方法能够确保任何文本，不论其语言或编码，都能被无损地编码成一串数字 Token ID。
- 特殊功能编码：数量 3 个，这些 Token 专门用于提供文本结构信息，确保模型能够正确处理和理解文本段落的边界和批处理时的对齐，便于计算。
```Python
def encode(self, text, add_bos=True, add_eos=True):
    # 把输入文本信息进行utf-8字符集编码，得到字节序列b
    # 每个字节值0-255对应一个Token ID
    b = text.encode('utf-8', errors='surrogatepass')
    ids = list(b)   # 将UTF-8字节序列转换为Token ID列表
    if add_bos:
        # 标记文本开头，添加<bos> Token ID
        ids = [self.bos] + ids 
    if add_eos:
        # 标记文本结束，添加<eos> Token ID
        ids = ids + [self.eos] 
    return ids # 返回最终处理完成的Token ID序列
def batch_encode(self, texts, pad_to=None):
    # pad_to 用于规定批处理中每条 Token ID 序列的目标长度（强制对齐）
    encs = [self.encode(t) for t in texts] 
    # 如果未指定pad_to，则使用当前批次中最长序列的长度；否则使用 pad_to 规定的长度
    maxlen = max(len(x) for x in encs) if pad_to is None else pad_to
    pad = self.pad

    # 将所有序列填充到maxlen长度，填充方式是在每条序列末尾添加[pad]，强制对齐形成规则的张量
    arr = [x + [pad] * (maxlen - len(x)) for x in encs] 
    
    # 记录原始序列的真实长度，这条信息将用于Attention，避免模型关注到[pad] Token
    lengths = torch.LongTensor([len(x) for x in encs])  

    # 经过填充对齐Token ID张量输入给模型，原始序列的真实长度张量用于Attention
    return torch.LongTensor(arr), lengths
```
### Step2. 构建自注意力层
```Python
class SimpleSelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        # 检查模型的隐藏层维度d_model能否被头数量nhead整除
        assert d_model % nhead == 0 
        self.nhead = nhead           # 多头注意力机制的头数量
        self.d_k = d_model // nhead  # 每一个头分配到的维度
        # 投影层将输入x投影到Q、K、V三个张量，总输出维度为3 * d_model。
        self.qkv = nn.Linear(d_model, d_model * 3) 
        self.out = nn.Linear(d_model, d_model)
    def forward(self, x, mask=None):
        B, T, D = x.shape # 输入张量的信息
        # 线性投影 Q, K, V
        qkv = self.qkv(x)  # 对输入[B, T, D]进行投影，得到形状为[B, T, 3*D]的融合张量
        q, k, v = qkv.chunk(3, dim=-1) # 沿最后一个维度均切分成Q, K, V，形状均为[B, T, D]

        # 多头拆分借助view()，Q, K, V改变[B, T, D]->[B, T, nhead, d_k]
        # transpose转置，Q, K, V改变[B, T, nhead, d_k]->[B, nhead, T, d_k]
        q = q.view(B, T, self.nhead, self.d_k).transpose(1, 2) # 先拆分再转置
        k = k.view(B, T, self.nhead, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.d_k).transpose(1, 2)

        # 计算Q和K的内积相似度，形状为[B, nhead, T, T]
        # 除以 √d_k 尺度缩放防止内积结果过大，导致归一化处理以后梯度消失
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Attention Mask掩码操作
        if mask is not None:
            # mask通常为[B, T]，生成掩码~mask.bool()需要掩码的地方为1，乘以-1e9将mask处的得分设为一个极小的负数。
            attn_mask = (~(mask.bool().unsqueeze(1).unsqueeze(2))) * -1e9
            scores = scores + attn_mask  # 对得分进行掩码操作

        # Softmax归一化：将得分转换为注意力权重，极小负数的位置权重趋近于0（完成屏蔽）。
        attn = F.softmax(scores, dim=-1) 
        # 注意力加权，权重attn乘以Value，得到加权求和的输出[B, nhead, T, d_k]。
        out = torch.matmul(attn, v)

        # 先转置恢复 [B, T, nhead, d_k]，
        # 然后用contiguous().view() 将所有头的输出拼接回原始的D维度[B, T, D]。
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)
```
### Step3. 构建 MoE 层
```Python
class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, n_experts=4, k=1, capacity_factor=1.25, noisy_gating=True):
        super().__init__()
        assert k in (1,2) # 确保K激活专家数是1或2
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.k = k
        self.capacity_factor = capacity_factor
        self.noisy_gating = noisy_gating

        # 门控网络：负责计算每个token与n_experts个专家的匹配得分（Logits）。
        self.w_gating = nn.Linear(d_model, n_experts, bias=False)
        if noisy_gating:
            # 噪声网络：引入噪声有助于在训练时平均分配token到不同的专家，缓解负载不均衡问题。
            self.w_noise = nn.Linear(d_model, n_experts, bias=False)

        # 专家网络，每个专家是一个独立的FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(), # 使用GELU激活函数
                nn.Linear(d_ff, d_model)
            ) for _ in range(n_experts)
        ])

    def _noisy_logits(self, x):
        """
            x : 展平后的输入token向量，形状为[N, D] (N=B*T)。
            Tensor: 带噪声的专家logits，形状为[N, E]。
        """
        logits = self.w_gating(x)
        
        # 在训练模式且开启noisy_gating时引入随机噪声
        if self.noisy_gating and self.training:
            # 使用sigmoid将w_noise的输出映射到[0, 1]，作为噪声的标准差
            noise_std = torch.sigmoid(self.w_noise(x))

            # 加上正态分布噪声这增强了随机性，有助于训练时的负载均衡
            logits = logits + torch.randn_like(logits) * noise_std
        return logits

    def forward(self, x, mask=None):
        B, T, D = x.shape
        N = B * T
        x_flat = x.view(N, D)  # [B, T, D] -> [N, D]

        logits = self._noisy_logits(x_flat)
        scores = F.softmax(logits, dim=-1) # 归一化的专家选择权重，[N, E]

        if self.k == 1:
            top1 = torch.argmax(scores, dim=-1)  # 每个token选出的Top-1专家索引，[N]
            # Dispatch Mask：[N, E]，标记每个token选中的Top-1专家，用1表示选中
            dispatch_mask = F.one_hot(top1, num_classes=self.n_experts).to(x.dtype)
            # 提取每个token选中的Top-1专家的得分作为最终组合权重，得到[N]
            combine_weights = torch.gather(scores, 1, top1.unsqueeze(1)).squeeze(1)
            # 计算规定每个专家最多处理的token数量
            capacity = int((N/self.n_experts)*self.capacity_factor)+1

            expert_inputs = []
            expert_indices = []
            for e in range(self.n_experts):
                # 找到专家e应该处理的token原始索引值，[N]
                idx = torch.nonzero(dispatch_mask[:, e], as_tuple=False).squeeze(-1)
                if idx.numel() > capacity:
                    # 专家e容量检查，如果超过容量，丢弃多余的token
                    idx = idx[:capacity]
                # 保存专家e需要处理的token
                expert_inputs.append(x_flat[idx])
                # 记录专家e需要处理token的原始索引值
                expert_indices.append(idx)
            # 初始化输出
            out_flat = torch.zeros_like(x_flat)

            # 遍历每个专家
            for e in range(self.n_experts):
                if expert_inputs[e].size(0)==0:
                    continue   # 专家e没有处理的token
                # 第e个专家处理token
                y = self.experts[e](expert_inputs[e])
                out_flat[expert_indices[e]] = y  # 将专家e的输出放回其在原始序列中的位置
            out_flat = out_flat * combine_weights.unsqueeze(1)  # 所有专家处理的结果乘以组合权重
            return out_flat.view(B, T, D)
        else:
            # Top-2 简化实现
            # 每个token选出Top-2专家的得分和索引，[N, 2]
            topk_vals, topk_idx = torch.topk(scores, k=2, dim=-1)
            # 计算出每个专家最大处理token数量
            capacity = int((N/self.n_experts)*self.capacity_factor)+1
            expert_buckets = [[] for _ in range(self.n_experts)] # 初始化存储空间
            for i in range(N):
                for j in range(2):
                    e = int(topk_idx[i,j].item())      # Top-K专家索引值
                    w = float(topk_vals[i,j].item())   # 相应的token组合权重
                    expert_buckets[e].append((i,w)) # 存储：token原始索引、权重

            out_flat = torch.zeros_like(x_flat) # 初始化输出结果
            for e in range(self.n_experts):
                bucket = expert_buckets[e]
                if len(bucket)==0:
                    continue
                if len(bucket) > capacity:
                    bucket = bucket[:capacity]  # 每个专家丢弃超过容量的token

                # token的原始索引值转化为张量：[C] ->（C=容量限制后的数量）
                idxs = torch.tensor([i for i,_ in bucket], device=x.device, dtype=torch.long)
                # 对应的组合权重转化为张量：[C]
                weights = torch.tensor([w for _,w in bucket], device=x.device, dtype=x.dtype)
                inp = x_flat[idxs]  # 获取专家e需要处理的token，[C, D]
                y = self.experts[e](inp)
                # 专家的输出乘以权重后，累加到输出张量上（Top-2叠加），可能有不止一个专家处理同一个token
                out_flat[idxs] += y * weights.unsqueeze(1)
            return out_flat.view(B,T,D)
```
# 4. DeepSeek 创新
## 4.1 DeepSeek V3
DeepSeek 的架构主要通过以下两个策略来实现专家专业化：
- 细粒度专家分割：将每个原专家分割成若干个更小的专家，从而显著增加总体专家个数，即将 N 个专家扩展为 mN 个小专家，为专家间更细致的分工提供可能。
- 保持计算成本恒定：模型在前向时，激活更多个小专家，例如将原来的 Top-K 扩展为 mK 个。
- 组合灵活性与指数级组合空间增长：细粒度化专家后，可供选择的专家集合组合数量呈阶乘、组合数爆炸式增长，从而显著提高路由器为某一输入构建专家联盟的自由度与多样性。
- 共享专家
- 负载均衡
## 4.2 DeepSeek V4
# 5. MoE 与深度学习