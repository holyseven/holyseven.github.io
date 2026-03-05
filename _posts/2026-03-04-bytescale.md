---
layout: post
title: "【论文笔记】ByteScale: 在 12000+ GPU 上高效训练 2048K 长上下文 LLM"
date: 2026-03-04
categories: [paper-notes]
tags: [distributed-training, long-context, parallelism, systems, LLM]
paper_title: "ByteScale: Efficient Scaling of LLM Training with a 2048K Context Length on More Than 12,000 GPUs"
paper_authors: "Hao Ge et al."
paper_link: "https://arxiv.org/abs/2502.21231"
---

# ByteScale: 当长短序列混合训练遇上万卡集群

> **论文**：[ByteScale: Efficient Scaling of LLM Training with a 2048K Context Length on More Than 12,000 GPUs](https://arxiv.org/abs/2502.21231)
> **作者**：Hao Ge, Junda Feng, Qi Huang, Fangcheng Fu, Xiaonan Nie, Lei Zuo, Haibin Lin, Bin Cui, Xin Liu
> **机构**：Peking University, ByteDance Seed

---

LLM 的上下文窗口已经从几千 token 拉到了百万级别（Gemini 1.5 Pro 支持 2M），但训练侧的系统效率并没有跟上。这篇论文指出了一个被广泛忽视的问题：**真实训练数据中长短序列高度混合**，而现有框架用静态的 DP $\times$ CP 网格去处理，短序列被迫走和长序列一样的通信路径，长短序列的计算量差异又导致设备间严重的负载不均。ByteScale 的核心思路是用一套动态的 Hybrid Data Parallelism (HDP) 取代静态网格，让每条序列按需分配恰好够用的设备数量，再辅以负载均衡调度，最终在 12000+ GPU 的生产集群上实现了最高 7.89 倍的加速。

---

## 问题：静态网格 vs 动态数据

先看数据的真实分布。论文 profile 了两个数据集（开源 GitHub 和字节内部 Byted），发现序列长度呈现极端的偏态分布——在 Byted 数据集中，约 80% 的 sample 长度在 4K 以内，但 0.05% 的超长序列（$\geq$ 2M）却贡献了 12.1% 的 token。

现有框架（如 Megatron-LM、DeepSpeed）的做法是把 DP 和 CP 当作两个正交的维度，建一个静态 2D mesh。CP 的 degree 必须设得足够大以容纳最长的序列。这带来两个问题：

**冗余通信**：假设每个设备能容纳 8K token，要支持 1M 上下文就需要 CP=128。于是那些只有 4K、8K 的短序列也被 pack 到 1M 后切成 128 份，在 128 个设备间做 ring-P2P 通信——完全没有必要。更糟的是，CP 需要 $O(S^2)$ 的计算来 overlap $O(S)$ 的通信，短序列算力不够根本掩盖不住通信开销。

**计算不均**：即使通过 packing 让每个设备分到相同数量的 token，由于 attention 的计算量正比于 $S^2$，包含一条长序列的 packed batch 和只有短序列的 packed batch 执行时间可以差好几倍。下游的 DP 和 PP 都依赖同步，慢的设备拖住快的，产生大量 bubble。

![Figure 5: 不均衡数据遇上 Pipeline Parallelism — 不同 micro-batch 执行时间差异导致大量 PP 和 DP bubble](/assets/images/bytescale/x5.png)

![Figure 6: 计算不均的量化 — 相同 token 数的 micro-batch，FLOPs 差异显著](/assets/images/bytescale/x6.png)

---

## ByteScale 的设计

ByteScale 包含三个组件：Profiler（离线 profile 环境和代价模型）、Communication Optimizer（消除冗余通信）、Balance Scheduler（均衡计算负载）。

![Figure 7: ByteScale 系统总览](/assets/images/bytescale/x7.png)

### Hybrid Data Parallelism (HDP)

HDP 是整个系统的核心抽象。它将 DP 和 CP 统一为一个概念：**在 $d_{\text{hdp}} = d_{\text{dp}} \times d_{\text{cp}}$ 个设备上均匀分配 token**。与传统方案不同，HDP 允许不同 rank 之间有异构行为：

- 短序列可以整条放在单个 rank 上，完全不需要跨设备通信
- 长序列只在恰好需要的设备数上做 CP 切分（而非固定切成 $d_{\text{cp}}$ 份）
- 通信组是动态建立的——一个 rank 可能在处理不同 micro-batch 时属于不同大小的通信组

![Figure 8: HDP 示意 — 短序列 S3、S5 各占一个 rank 无需通信，长序列 S4 只用 2 个 rank](/assets/images/bytescale/x8.png)

一个关键细节：动态创建 NCCL 通信组开销很大且占额外显存（5-10 GB）。ByteScale 利用 HDP rank 间预建的全局通信组，通过 P2P 通信复用已有 group，避免了频繁创建临时 group。

梯度正确性方面，论文证明 HDP 的梯度累积结果与标准 DP 数学等价——无论 token 怎么分配，最终 All-Reduce 汇聚的都是所有 token 对参数的梯度之和，只需在 loss 计算时按 token 数而非 sample 数做归一化。

### Selective Offloading：用 CPU 内存换 GPU 数量

对于长序列，即便用了 HDP 动态分组，所需的 rank 数仍然很多（1M 序列需要 128 个 rank）。ByteScale 利用一个关键观察：attention 的计算复杂度是 $O(S^2)$，而 activation 的内存是 $O(S)$。当序列足够长时，$O(S^2)$ 的计算时间足以完全掩盖 $O(S)$ 的 PCIe 传输时间。

因此，可以把部分 activation offload 到 CPU 内存（利用 FILO 特性，前面层的 activation 最后才在 backward 时使用），从而减少所需的 GPU 数量。offload ratio $r$ 通过一个代价模型自动求解，对每个 micro-batch 独立设定——长序列用高 ratio，短序列不 offload。

![Figure 11: Data-Aware Selective Offloading — 根据序列长度自动决定 offload 比例](/assets/images/bytescale/x11.png)

### Balance Scheduler：让所有设备同时完工

消除了冗余通信后，剩下的瓶颈就是计算不均衡。ByteScale 的 Balance Scheduler 有两个关键 insight：

**Insight 1（PP 场景）**：把相近长度的序列聚集到同一条 pipeline 里，减少 pipeline 内不同 micro-batch 间的执行时间差异。同时允许不同 pipeline 处理不同数量的 micro-batch——短序列多的 pipeline 多跑几个 batch 来填满时间。

**Insight 2（纯 DP 场景）**：只需保证同一时刻各 rank 在执行计算量相近的 micro-batch，不需要跨时间步均衡。

具体策略是：先按长度排序并分桶（每桶 FLOPs 总量相近），然后贪心地把桶分配给当前空闲时间最多的 rank。执行时间短的 rank 会被分到更多的 micro-batch。

![Figure 13: Balance Strategy — (a) DP-Balance 保证同一时刻各 rank 执行时间相近；(b) PP-Balance 把相似长度聚集到同一 pipeline](/assets/images/bytescale/x13.png)

![Figure 12: 均衡后的 Pipeline — 不同 pipeline 处理不同数量的 micro-batch](/assets/images/bytescale/x12.png)

---

## 实验结果

实验在字节跳动 12000+ GPU 生产集群上完成，模型从 LLaMA-7B 到 Mistral-8×22B（141B 参数），上下文长度从 256K 到 2048K。

### 端到端吞吐

![Figure 17: 端到端评估 — tokens/sec 吞吐量对比](/assets/images/bytescale/x17.png)

核心数据：

| 模型 | 上下文 | 数据集 | Baseline (tok/s) | HDP Naive | HDP Balance | 加速比 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| LLaMA-7B | 2M | GitHub | ~6K | ~21K | ~47K | **7.89×** |
| LLaMA-30B | 2M | GitHub | ~2K | ~7K | ~14K | 6.21× |
| LLaMA-70B | 2M | GitHub | ~1K | ~2.5K | ~4.3K | 4.28× |
| LLaMA-7B | 2M | Byted | ~14K | ~31K | ~55K | 3.89× |

一个值得关注的规律：上下文每增长 2 倍，baseline 吞吐几乎减半，而 HDP Balance 只下降约 8%。这说明 ByteScale 的优势随着长短序列比例差异增大而增大。

DP-Balance 策略（不用 PP 的小模型）加速比普遍高于 PP-Balance 策略（需要 PP 的大模型），因为前者只需要在每个时间步内做局部均衡，而后者需要全局均衡。

### Case Study：一个训练步的详细剖析

论文选取 LLaMA-7B + 2M + Byted 配置做了深入分析，结果非常直观：

![Figure 18: Case Study — 三种方案下各 rank 的执行时间对比](/assets/images/bytescale/x18.png)

- **Baseline**：97.6% 的时间花在通信上（P2P 通信时间远超计算时间），总耗时 8m40s
- **HDP Naive**：通信被压缩，但 rank 间执行时间方差极大（min=60s, max=279s），最快的 rank 等最慢的 rank 171s
- **HDP Balance**：所有 rank 几乎同时完成，总耗时 2m37s

### 消融实验

![Figure 20: 各组件消融 — 依次叠加各优化的加速比](/assets/images/bytescale/x20.png)

| 组件 | 累积加速比 | 增量贡献 |
|:---|:---:|:---|
| Dynamic Communication | 1.59× | 消除短序列冗余通信 |
| + Selective Offloading | 2.01× | 压缩长序列所需 rank 数 |
| + Balance Strategy | 3.69× | 消除计算不均衡（贡献最大） |
| + Remote Dataloader | 3.89× | CPU 预取 overlap 数据加载 |

Balance Strategy 贡献了最大的增量加速，这和直觉一致：在大规模集群上，设备利用率的均匀性比单个设备的峰值性能更重要。

---

## 思考与讨论

**HDP 的核心洞察其实很朴素**：不要让短序列走长序列的通信路径。这个问题之前不是没人意识到，而是在静态并行框架下很难优雅解决。ByteScale 的贡献在于把 DP 和 CP 统一成一个可以动态配置的维度，并处理了 NCCL group 管理、梯度等价性证明、offloading 策略等一系列工程细节。

**Selective Offloading 的适用条件值得关注**。它依赖 $O(S^2)$ 计算掩盖 $O(S)$ 传输——这在标准 attention 下成立，但如果未来 linear attention 或稀疏 attention 成为主流（计算复杂度降到 $O(S)$），offloading 的效果会打折扣。论文目前的代价模型是为二次复杂度 attention 设计的。

**Balance Scheduler 的贪心策略在极端分布下的表现是一个自然的后续问题**。当前的启发式算法在实验中效果很好，但理论上最优的负载均衡是 NP-hard 问题。如果数据分布更极端（比如一个 global batch 中只有一条超长序列和大量短序列），贪心策略是否还能给出接近最优的方案？论文没有讨论这个边界情况。

**从应用场景看，ByteScale 的价值可能随着 RL-based training（如 DeepSeek-R1）的普及而进一步放大**。RL 训练中 response 长度天然高度可变，比预训练数据的长度偏态更极端，恰好是 ByteScale 最擅长处理的场景。
