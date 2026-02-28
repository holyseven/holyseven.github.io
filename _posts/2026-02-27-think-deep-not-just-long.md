---
layout: post
title: "【论文笔记】用 Deep-Thinking Tokens 度量 LLM 推理努力"
date: 2026-02-27
categories: [paper-notes]
tags: [reasoning, LLM, test-time-compute, interpretability, inference-scaling]
paper_title: "Think Deep, Not Just Long: Measuring LLM Reasoning Effort via Deep-Thinking Tokens"
paper_authors: "Wei-Lin Chen et al."
paper_link: "https://arxiv.org/abs/2602.13517"
---

# Think Deep, Not Just Long: 用 Deep-Thinking Tokens 度量 LLM 推理努力

> **论文**：[Think Deep, Not Just Long: Measuring LLM Reasoning Effort via Deep-Thinking Tokens](https://arxiv.org/abs/2602.13517)
> **作者**：Wei-Lin Chen, Liqian Peng, Tian Tan, Chao Zhao, Blake JianHang Chen, Ziqian Lin, Alec Go, Yu Meng
> **机构**：University of Virginia, Google
> **arXiv**：2602.13517v1, 2026-02-13

---

## 一句话总结

这篇论文发现，衡量 LLM 推理质量不应该看输出有多长（token 数量），而应该看每个 token 在模型内部经历了多深层的"思考修正"——作者据此提出 deep-thinking ratio (DTR)，并证明它与任务准确率的相关性远强于长度或置信度指标。

---

## 背景知识铺垫

理解这篇论文需要三个前置概念。

第一个是 **Chain-of-Thought (CoT)** 推理范式。自 2022 年 Wei et al. 的工作以来，让 LLM "一步步想"已经成为提升推理能力的标准做法。直觉上，更长的思考过程 = 更多的推理计算 = 更好的结果，这个假设驱动了 o1、DeepSeek-R1、Kimi k1.5 等一系列 reasoning model 的发展。

第二个是 **overthinking 现象**。近期多项研究（Gema et al., 2025; Wu et al., 2025）发现，CoT 长度和准确率之间的关系并非单调递增，而是呈倒 U 型曲线——超过某个甜蜜点后，更长的推理反而导致性能下降。模型可能在冗长的推理中不断放大错误启发式，或者纠结于无关细节。

第三个是 **logit lens** 技术。Nostalgebraist (2020) 发现，把 Transformer 中间层的隐藏状态直接乘上输出层的 unembedding 矩阵，可以得到有意义的 token 预测分布。也就是说，模型并非到最后一层才"突然知道答案"，而是逐层迭代地精化预测。这篇论文的核心方法正是建立在这一观察之上。

---

## 核心方法

### 从"想多久"到"想多深"

论文的出发点很直接：既然 token 数量不能反映推理质量，那什么能？作者的回答是——去看模型内部，每个 token 到底经历了多深层次的"预测修正"。

想象一个 36 层的 Transformer 在生成一个数学推理序列。当它输出"and"、"is"这样的功能词时，可能在前几层就已经确定了要输出什么，后面的层只是走个过场。但当它输出一个关键的计算结果（比如"13"）或选择答案符号（比如"(D)"）时，模型需要在更深的层中不断修正自己的预测，直到很晚才最终收敛。下图（论文 Figure 2）的热力图非常直观地展示了这一点——颜色越深代表中间层预测与最终层的 JSD 越大（分歧越大），可以看到功能词在浅层就已收敛（浅色），而关键数字和符号在深层仍有显著修正（深色）。

![Figure 2: Heatmap of thought — 展示每个 token 在不同层与最终层预测分布的 JSD 距离](/assets/images/think-deep/x2.png)

### DTR 的形式化定义

具体来说，对一个有 $L$ 层的自回归模型，在生成第 $t$ 个 token 时，每一层 $l$ 都会产生一个隐藏状态 $h_{t,l}$。将中间层的隐藏状态通过 unembedding 矩阵 $W_U$ 投影到词汇表空间，就得到了该层的预测分布：

$$p_{t,l} = \text{softmax}(W_U h_{t,l})$$

然后，用 Jensen-Shannon 散度 (JSD) 衡量每一层的预测分布与最终层分布之间的距离：

$$D_{t,l} = \text{JSD}(p_{t,L} \| p_{t,l})$$

其中 $D_{t,l}$ 表示第 $t$ 个 token 在第 $l$ 层时的预测分布距最终预测还有多远。当 $D_{t,l}$ 接近零时，说明该层的预测已经和最终结果一致了。JSD 被选中是因为它具有对称性和有界性（值域 $[0, 1]$），相比 KL 散度更稳定。

接下来定义 **settling depth**（收敛深度）$c_t$。为了避免 JSD 偶然降低后又反弹的情况，作者取了累积最小值：

$$\bar{D}_{t,l} = \min_{j \leq l} D_{t,j}$$

settling depth 就是 $\bar{D}_{t,l}$ 首次降到阈值 $g$ 以下的层：

$$c_t = \min\{l : \bar{D}_{t,l} \leq g\}$$

如果 $c_t$ 落在模型的"深层区域"（由深度分数 $\rho$ 定义，即 $c_t \geq \lceil \rho \times L \rceil$），这个 token 就被标记为 **deep-thinking token**。下图（论文 Figure 3 左侧）展示了这个判定流程：从底层到顶层逐层计算 JSD，当 JSD 在靠近顶层的位置才首次低于阈值时，该 token 就是 deep-thinking token。

![Figure 3: Deep-thinking token 的识别流程与 DTR 算法](/assets/images/think-deep/x3.png)

最后，对整个生成序列 $S$（长度 $T$），DTR 定义为 deep-thinking token 的比例：

$$\text{DTR}(S) = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}[c_t \in \mathcal{L}_{\text{deep-thinking}}]$$

直觉上，DTR 高意味着序列中有更大比例的 token 需要模型"深入思考"才能确定，这反映了推理过程中的实质性计算投入，而非纯粹的文本量堆砌。

### 超参数选择

DTR 有两个超参数：阈值 $g$ 和深度分数 $\rho$。论文统一设定 $g = 0.5$、$\rho = 0.85$。消融实验显示：$g$ 对相关性的影响比 $\rho$ 大——过于宽松的 $g = 0.25$ 会让太多 token 被识别为 deep-thinking，削弱区分力；而 $\rho$ 的变化主要只是平移了 DTR 的数值范围，正相关的趋势基本不变。

![Figure 4: 超参数消融 — (左) 不同阈值 g 的影响；(右) 不同深度分数 ρ 的影响](/assets/images/think-deep/x4.png)

### Think@n：基于 DTR 的 test-time scaling 策略

论文的第二个贡献是将 DTR 应用到推理时的样本选择中。Think@$n$ 的工作方式是：对每个问题采样 $n$ 个回答，只根据前 50 个 token 估算每个回答的 DTR，然后丢弃 DTR 排名后 50% 的回答，只对 DTR 最高的那一半进行 majority voting。

这个设计有两个关键点：一是只需要看前 50 个 token 就能估算 DTR（消融实验甚至显示 prefix=50 比更长的 prefix 效果更好）；二是通过 early rejection 可以在生成一小段后就终止不靠谱的回答，节省约 50% 的推理开销。

---

## 实验结果

实验覆盖了三个模型家族（GPT-OSS 20B/120B、DeepSeek-R1-70B、Qwen3-30B-Thinking）和四个高难度推理 benchmark（AIME 2024/2025、HMMT 2025、GPQA-Diamond）。每个问题采样 25 次回答，统计量在 30 个随机种子上取平均。

### DTR 与准确率的相关性

核心结果见下图（论文 Figure 1）。左图展示 token 长度与准确率的关系（平均相关 $r = -0.544$，负相关），右图展示 DTR 与准确率的关系（平均相关 $r = 0.828$，强正相关），对比非常鲜明。

![Figure 1: Token 长度 vs DTR 与准确率的相关性对比 (GPT-OSS-120B-medium)](/assets/images/think-deep/x1.png)

下表提取了 Table 1 中各指标在全部 32 个模型-benchmark 组合上的平均 Pearson 相关系数：

| 度量指标 | 平均 Pearson $r$ | 方向稳定性 |
|:---|:---:|:---|
| Token 长度 | -0.594 | 几乎一致负相关 |
| 反转 Token 长度 | +0.594 | 正相关但纯后验技巧 |
| Log Probability | +0.527 | 正相关但波动大 |
| Negative Perplexity | +0.219 | 弱且不稳定 |
| Negative Entropy | +0.571 | 中等正相关 |
| Self-Certainty | +0.605 | 较好但仍有负相关 case |
| **DTR (本文)** | **+0.683** | **32 组中仅 2 组负相关** |

### Think@n 的效果

下表提取 Table 2 关键数据（OSS-120B-medium，$n=48$）：

| 方法 | AIME 25 Acc | AIME 25 Cost | AIME 24 Acc | HMMT 25 Acc | GPQA-D Acc |
|:---|:---:|:---:|:---:|:---:|:---:|
| Cons@n (全量投票) | 92.7% | 307.6k (基准) | 92.7% | 80.0% | 73.8% |
| Mean@n (无筛选) | 80.0% | 307.6k | 81.6% | 62.6% | 69.9% |
| Short@n (选最短) | 87.3% | 255.7k (-17%) | 88.0% | 77.3% | 73.3% |
| Self-Certainty@n | 87.3% | 150.6k (-51%) | 91.3% | 78.0% | 76.0% |
| **Think@n (本文)** | **94.7%** | **155.4k (-49%)** | **93.3%** | **80.0%** | **74.7%** |

Think@n 在几乎所有 benchmark 上都匹配或超过全量投票（Cons@n），同时推理开销减少约 50%。下图（论文 Figure 5）直观展示了各方法在准确率-开销 Pareto 前沿上的位置：

![Figure 5: 准确率 vs 推理开销的 Pareto 前沿比较](/assets/images/think-deep/x6.png)

### Prefix 长度消融

关于"用多少 token 就能估算 DTR"，Table 3 的结果出人意料（AIME 2025，OSS-120B-medium）：

| Prefix 长度 | 准确率 | Cost (k tokens) |
|:---:|:---:|:---:|
| 50 | **94.7%** | 155.4k |
| 100 | 92.0% | 154.1k |
| 500 | 92.7% | 153.2k |
| 1000 | 92.7% | 177.4k |
| 全序列 | 94.0% | 307.6k |

仅用 50 个 token 估算的 DTR 就能达到甚至超过全序列估算的效果，这暗示推理质量的信号在生成的最初阶段就已经存在。

---

## 思考与讨论

**DTR 落地的一个现实挑战是对中间层隐藏状态的依赖。** 计算 DTR 需要获取模型每一层的隐藏状态做 unembedding 投影和 JSD 计算，这在开源模型上没有问题，但对于 API-only 的服务（如 GPT-4、Claude）就不适用了。一个自然的延伸方向是：模型服务方是否可以将 DTR 作为元数据随响应一起返回？如果 DTR 的信号确实如论文所示那样稳健，将它集成到 API 层面可能是一个有价值的功能。

**关于 DTR 计算本身的开销，值得进一步量化。** 对每个 token 的所有 $L$ 层做 softmax 投影和 JSD 计算，在词汇表达到 128K 的模型上不算轻量。Think@$n$ 报告的 50% 开销节省只计算了生成的 token 数，如果把 DTR 自身的计算成本算进去，实际的效率收益会更清晰。这一点在工程实践中值得关注。

**一个有趣的观察是 DTR 在不同模型间不可直接比较。** 附录 B 显示，同一个 GPT-OSS-120B 模型的不同 reasoning level 配置，DTR 数值差异显著——低 reasoning level 反而 DTR 更高。作者解释为高 reasoning level 把计算从"深度"重新分配到了"长度"。这意味着 DTR 更适合作为同一模型内部的质量排序信号，而非跨模型的能力度量。如果未来能找到某种归一化方案使其跨模型可比，DTR 的应用场景会更广。

**此外，Table 1 中有个别模型-benchmark 组合出现了 DTR 与准确率的负相关**（如 Qwen3-30B-Thinking 在 AIME 2024 上 $r=-0.657$），这提示 DTR 的适用性可能与模型架构或训练方式有关。进一步研究这些 case 背后的原因，可能有助于理解 DTR 在什么条件下最有效。
