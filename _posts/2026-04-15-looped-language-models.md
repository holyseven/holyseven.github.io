---
layout: post
title: "【论文笔记】Looped Language Models：用循环共享参数实现隐式推理"
date: 2026-04-15
categories: [paper-notes]
tags: [LLM, architecture, latent-reasoning, parameter-efficiency, adaptive-computation]
paper_title: "Scaling Latent Reasoning via Looped Language Models"
paper_authors: "Rui-Jie Zhu et al."
paper_link: "https://arxiv.org/abs/2510.25741"
---

> **论文**：[Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741)
> **作者**：Rui-Jie Zhu, Zixuan Wang, Kai Hua, Tianyu Zhang, Ziniu Li, Haoran Que, Boyi Wei, Zixin Wen, Fan Yin, He Xing, Lu Li, Jiajun Shi, Kaijing Ma, Shanda Li, Taylor Kergan, Andrew Smith, Xingwei Qu, Mude Hui, Bohong Wu, Qiyang Min, Hongzhi Huang, Xun Zhou, Wei Ye, Jiaheng Liu, Jian Yang, Yunfeng Shi, Chenghua Lin, Enduo Zhao, Tianle Cai, Ge Zhang, Wenhao Huang, Yoshua Bengio, Jason Eshraghian
> **机构**：ByteDance Seed, UC Santa Cruz, Princeton, Mila, CMU, PKU, UPenn, U of Manchester, M-A-P 等

## 先从一个问题出发

现在 LLM 提升推理能力的主流做法是 Chain-of-Thought（CoT）：让模型把思考过程写成文字，一步步推导出答案。这种做法有效，但代价是输出变长，占用大量上下文窗口，而且推理能力的获得被推迟到 post-training 阶段。

有没有可能让模型在不增加输出长度的情况下"多想几遍"？这篇论文给出了一种方案：把 transformer 的层重复使用多次——同一组参数在 forward pass 中被反复调用，每次调用都进一步精炼 hidden state。这就是 Looped Language Model（LoopLM）。作者基于此训练了 Ouro 系列模型（1.4B 和 2.6B），在 7.7T tokens 上预训练后，用 1.4B 参数达到了 4B 级别 dense 模型的性能，2.6B 则能匹配甚至超过 8B 模型。

## 什么是 Looped Language Model

### 基本结构

标准 transformer 将 $L$ 层依次堆叠，每层参数不同：

$$
F(\cdot) = \text{lmhead} \circ \mathcal{T}_{\theta_L} \circ \cdots \circ \mathcal{T}_{\theta_1} \circ \text{emb}(\cdot)
$$

LoopLM 的做法是：只有一组 $L$ 层的参数 $\mathcal{M}^L$，但重复使用 $t$ 次：

$$
F^{(t)}(\cdot) = \text{lmhead} \circ \underbrace{\mathcal{M}^L \circ \mathcal{M}^L \circ \cdots \circ \mathcal{M}^L}_{t \text{ iterations}} \circ \text{emb}(\cdot)
$$

当 $t=1$ 时就退化为标准 transformer。增大 $t$ 相当于加深计算图的深度而不增加参数量。

![Figure 3](/assets/images/looped-language-models/x3.png)
*Figure 3: LoopLM 架构示意。左侧为训练流程：N 层参数重复 $T_{\max}$ 次，每次都计算 LM loss 和退出概率。右侧为推理流程：根据累积退出概率决定何时停止。*

可以从两个视角理解这个架构：

- **参数共享视角**：这是一种参数效率技术，类似 ALBERT 的跨层权重共享，用更少参数实现更深的计算
- **隐式推理视角**：每次循环相当于一步"潜在思考"，hidden state 在迭代中逐步精炼，形成一种不依赖文本输出的 latent chain-of-thought

### 自适应退出机制

不是所有 token 都需要同样多的计算。简单的 token 可能 1-2 轮就够了，复杂的 token 需要完整的 4 轮。为此，模型在每个循环步 $t$ 都有一个 exit gate，输出当前步的退出概率：

$$
\lambda_t(x) = \sigma(\text{Linear}_\phi(h^{(t)})) \in (0, 1)
$$

从这些逐步退出概率可以构造出一个在所有步骤上的离散分布 $p_\phi(t|x)$——直觉上，它表示"给定输入 $x$，模型应该在第 $t$ 步退出"的概率。推理时，设定一个阈值 $q$，当累积退出概率 $\text{CDF}(m|x) \geq q$ 时即停止计算。$q$ 越小越倾向于早退出（省计算），$q$ 越大允许更深的计算。

### 训练目标

训练时最核心的问题是：如果只优化 next-token prediction loss，模型会发现"多循环几次总是更好"，于是 $p_\phi$ 会坍缩到总是选择最大步数 $T_{\max}$。这就失去了自适应的意义。

解决方案是加入熵正则化：

$$
\mathcal{L} = \underbrace{\sum_{t=1}^{T_{\max}} p_\phi(t|x) \cdot \mathcal{L}^{(t)}}_{\text{期望 task loss}} - \underbrace{\beta \cdot H(p_\phi(\cdot|x))}_{\text{熵正则}}
$$

第一项是对各步 loss 的加权期望——退出概率高的步骤对总 loss 贡献大。第二项的熵正则阻止退出分布坍缩到某一步。系数 $\beta$ 控制探索与利用的平衡。

这个目标也可以解读为一个 variational inference 问题：退出步骤 $z$ 是隐变量，$p_\phi(z|x)$ 是 variational posterior，uniform 分布是 prior，整个 loss 等价于负 ELBO。

训练分两个阶段：Stage I 在预训练中联合学习 exit gate 和 LM 参数；Stage II 冻结 LM，单独微调 gate 参数来锐化退出决策——此时引入一个 adaptive exit loss，基于"当前步相比上一步的 loss 改进量"来生成训练信号，教 gate 判断继续计算是否还有收益。

## Ouro 模型的训练

![Figure 4](/assets/images/looped-language-models/x4.png)
*Figure 4: Ouro 端到端训练流水线。从共享 warmup 到 Stable Training，然后分叉为 1.4B（保留原始 24 层）和 2.6B（通过 layer duplication 扩展到 48 层），经过四个阶段最终 SFT 得到 Ouro-Thinking。*

总训练量为 7.7T tokens，横跨 5 个阶段：

| 阶段 | 序列长度 | Tokens | 循环步数 | 重点 |
|------|---------|--------|---------|------|
| 1a: Pre-train I | 4K | 3T | 8 | Web 数据 |
| 1b: Pre-train II | 4K | 3T | 4 | Web 数据（稳定性调整） |
| 2: CT Annealing | 16K | 1.4T | 4 | 高质量数据 + 数学代码 |
| 3: LongCT | 64K | 20B | 4 | 长上下文 |
| 4: Mid-training | 32K | 300B | 4 | SFT 质量数据 |

一个有趣的工程发现：最初用 8 步循环时训练出现 loss spike 和梯度振荡，原因可能是梯度在多次循环中被放大。降到 4 步后稳定了。2.6B 模型是在 1.4B 的 24 层基础上通过 layer duplication 扩展到 48 层——由于参数本身就是共享的，这种"upcycling"比标准 transformer 的层复制更自然。

Ouro-Thinking 版本通过在 8.3M 样本（数学 3.5M、代码 3.2M、科学 808K、对话 767K）上做 SFT 获得。作者也尝试了 RL（DAPO、GRPO），但因为 LoopLM 的动态退出机制与 vLLM/SGLang 的固定执行路径不兼容，效果并不理想。

## 实验结果

### Base model：1.4B 参数追平 4B，2.6B 超过 8B

| 模型 | 参数 | MMLU | BBH | MATH500 | GSM8K | HumanEval |
|------|------|------|-----|---------|-------|-----------|
| Qwen2.5-1.5B | 1.5B | 60.99 | 43.66 | 17.60 | 60.73 | 52.40 |
| Qwen3-4B | 4.0B | 73.19 | 70.95 | 59.60 | 72.86 | 77.40 |
| **Ouro 1.4B R4** | 1.4B | 67.35 | 71.02 | 82.40 | 78.92 | 74.40 |
| Qwen3-8B | 8.0B | 76.95 | 77.65 | 62.30 | 83.09 | 84.80 |
| **Ouro 2.6B R4** | 2.6B | 74.60 | 80.46 | 90.85 | 81.58 | 78.70 |

Ouro 1.4B 在 BBH（71.02 vs 70.95）和 MATH500（82.40 vs 59.60）上超过了参数量近 3 倍的 Qwen3-4B。2.6B 模型在 MMLU-Pro、BBH、MATH500 上超过了 Qwen3-8B。优势集中在需要多步推理的任务上，而非纯知识记忆任务。

### Reasoning model：小模型 vs 大模型

| 模型 | AIME24 pass@1 | AIME25 pass@1 | OlympiadBench | BeyondAIME |
|------|-------------|-------------|---------------|------------|
| Ouro-1.4B-Thinking R4 | 65.0 | 46.3 | 71.6 | 34.0 |
| Qwen3-4B | 61.3 | 51.3 | 73.2 | 31.0 |
| Ouro-2.6B-Thinking R4 | 64.7 | 50.3 | 76.4 | 39.0 |
| Qwen3-8B | 73.0 | 66.7 | 75.3 | 38.0 |

Ouro-1.4B-Thinking 在 AIME24 和 BeyondAIME 上击败了 Qwen3-4B，2.6B 版本在 OlympiadBench 和 BeyondAIME 上超过了 Qwen3-8B。

### 循环深度与外推

模型在训练深度 $T=4$ 处达到峰值性能。超过训练深度后，benchmark 性能出现温和下降（如 1.4B 在 MMLU 上从 $T=4$ 的 67.45 降到 $T=8$ 的 64.49），说明外推能力有限。但一个有趣的现象是：安全对齐指标在超过训练深度后反而持续改善。

### Early exit 策略

![Figure 5](/assets/images/looped-language-models/x5.png)
*Figure 5: MMLU 上不同 early exit 策略的准确率-计算量权衡。横轴为平均退出轮数，纵轴为准确率。带有 adaptive exit training 的 Ponder gate（橙色菱形）在所有计算预算下都表现最好。*

四种策略的对比显示：经过专门 adaptive exit 训练的 gate 在所有计算预算下都最优；未经专门训练的 gate 也明显优于 static baseline；hidden state 差值阈值方法的表现出人意料地有竞争力。从 1 轮到 2 轮的性能跃升（40% → 60%）远大于 3 到 4 轮的增量，说明大多数 example 在中等深度就接近最优，只有少数需要完整深度。

### KV Cache 共享

循环架构推理时需要每步维护独立的 KV cache，带来 4 倍内存开销。实验发现：在 decoding 阶段只保留最后一步的 KV cache，性能几乎无损（GSM8K 78.85 vs 完整 cache 的 78.92），内存减少 4 倍。但 prefilling 阶段不能共享，否则性能大幅下降。

## LoopLM 的优势来自哪里：不是记更多，而是用更好

论文通过两组合成任务回答了一个关键问题：LoopLM 的性能提升到底来自更强的知识存储能力，还是更强的知识组合能力？

**知识存储实验**（Capo task）：训练不同大小的 GPT-2 模型去记忆合成人物传记，测量每参数存储的 bits 数。结果是：相同参数量下，LoopLM 和标准 transformer 的 knowledge capacity 几乎相同（都约 2 bits/parameter）。循环不增加存储容量。

**知识操作实验**（Mano task + Multi-hop QA）：

- Mano task 要求模型解决模运算的表达式树，需要组合多个算术规则。2 层 LoopLM 循环 6 次（$2 \otimes 6$）在最难的难度上达到 78.0% 准确率，而同参数量的 12 层标准 transformer（$12 \otimes 1$）只有 34.8%
- Multi-hop QA 要求组合多个事实进行推理。LoopLM 需要更少的训练样本就能学会 3-hop 推理任务

![Figure 7](/assets/images/looped-language-models/x8.png)
*Figure 7: Multi-hop QA 任务的样本效率对比。左图：横轴为训练样本量占比，纵轴为准确率，LoopLM 需要更少样本达到相同性能。右图：相同样本量下的训练曲线，循环更多的模型收敛更快。*

结论：循环带来的优势在于更强的知识组合和操作能力——模型可以在多次循环中反复调用同一组参数中编码的知识和规则，实现更深的推理链。论文还给出了理论支撑（Theorem 1）：对于知识图上的可达性问题，LoopLM 只需 $O(\log_2 D)$ 次循环（$D$ 为图直径），而 discrete CoT 需要 $O(n^2)$ 步，continuous CoT 需要 $O(D)$ 步。

## 思考与讨论

论文尝试了两种 RL 方案但都没有超过 SFT checkpoint。第一种是在 vLLM 中做完整 4 步 rollout 后模拟 early exit 截断 loss，但这造成 off-policy mismatch（token 在完整深度下生成，loss 在较浅深度计算），效果不行。第二种固定 4 步做 rollout 和更新，避免了 off-policy 问题，训练过程正常但性能没有提升——论文认为原因可能是模型规模较小加上 SFT 已经充分训练，RL 的提升空间有限。这两个失败的原因不同：前者是推理框架（vLLM/SGLang）不支持动态计算路径导致的工程限制，后者更像是小模型 RL headroom 不足的问题。这也意味着 LoopLM 要充分发挥潜力，既需要推理基础设施的适配，也需要在更大规模上验证 RL 的效果。

一个自然的延伸方向是循环步数的外推。目前模型只在 $T=4$ 上训练，超过训练深度后性能下降。如果能让模型在更多循环步数上保持甚至继续提升性能，那 LoopLM 就真正实现了一种 test-time compute scaling 的范式——不增加参数，不增加输出长度，只增加内部迭代次数。论文中安全指标在 $T>4$ 时继续改善的现象暗示某些能力确实可以外推，如何将这种外推能力拓展到通用任务上是个开放问题。

关于与 CoT 的关系，论文在 Section 9 中提到一个有趣的 probe 实验：LoopLM 内部隐状态中编码的答案与最终输出的一致性，高于 CoT 推理链中早期 token 与最终答案的一致性。换言之，LoopLM 的隐式推理过程比 CoT 的显式推理过程对最终答案的"承诺"更早、更一致。这对理解隐式推理与显式推理的差异提供了一个新的视角，不过这一结论的普适性还需要更大规模模型和更多任务类型的验证。
