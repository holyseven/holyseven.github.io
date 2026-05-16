---
layout: post
title: "【论文笔记】从宏观到微观：SFT 数据集多样性的三级分析框架"
date: 2026-05-13
categories: [paper-notes]
tags: [SFT, data-diversity, LLM, fine-tuning, token-level]
paper_title: "From Macro to Micro: Probing Dataset Diversity in Language Model Fine-Tuning"
paper_authors: "Haoyu Li et al."
paper_link: "https://arxiv.org/abs/2505.24768"
---

> **论文**：[From Macro to Micro: Probing Dataset Diversity in Language Model Fine-Tuning](https://arxiv.org/abs/2505.24768)
> **作者**：Haoyu Li, Xuhong Li, Yiming Dong, Kun Liu
> **机构**：北京理工大学, 百度

## 快速概览

SFT（Supervised Fine-Tuning）阶段的数据集多样性对 LLM 性能有显著影响，但之前的研究几乎只关注 instruction 端的多样性（比如让指令覆盖更多主题或标签）。这篇论文做了两件事：（1）提出了一个从宏观（语义聚类）、中观（标签分解）到微观（token 频率分布）的三级多样性分类体系；（2）发现在 response 端做 token 级别的多样性控制，与模型性能的相关性最强，且在最大多样性时性能最优。

核心直觉是：SFT 训练时模型只在 response token 上计算 loss，因此直接控制 response 的 token 多样性比间接通过 instruction 多样性来影响训练更有效。

## 问题背景：为什么关注数据集多样性

在 LLM 的 SFT 阶段，训练数据通常是 instruction-response 对。业界的共识是，SFT 数据集需要足够"多样"才能让模型泛化得好，但"多样"这个概念本身缺乏系统定义。

已有工作主要沿两个方向控制 instruction 的多样性：

- 宏观策略（BERTopic 等）：把 instruction embedding 后聚类，选择覆盖更多聚类的样本
- 中观策略（InsTag 等）：用 LLM 为每条 instruction 生成多个原子标签（如"代码生成""情感分析"），然后控制标签覆盖面

这些策略都作用在 instruction 上。但 LLM 在 SFT 时是以 response 为训练信号的——instruction 只是输入上下文，loss 只算在 response 的 token 上。这就产生了一个错位：我们控制的是 instruction 的多样性，期望的是 response 端的训练效果。

## 三级多样性框架

![Figure 1](/assets/images/dataset-diversity-sft/x1.png)
*Figure 1: 三种多样性控制策略示意。宏观层面对 instruction 做语义聚类；中观层面将 instruction 分解为原子标签；微观层面对 response 做 token 频率分析。*

论文把多样性策略按粒度分为三级，每级都可以分别作用在 instruction 或 response 上：

**宏观（Macroscopic）**：把文本 embedding 后用 HDBSCAN 聚类，每个簇代表一个语义主题。通过控制选取的簇数量来调节多样性。

**中观（Mesoscopic）**：用 LLM 为每条文本生成若干原子标签（如 InsTag 方法），标签进一步聚类。通过控制覆盖的标签类型数量来调节多样性。

**微观（Microscopic）**：用 LLM 的 tokenizer 对文本做 tokenization，统计 token 频率，然后按频率将 token 分为三档：

![Figure 2](/assets/images/dataset-diversity-sft/x2.png)
*Figure 2: SFT 数据集中 token 的频率分布。高频段（>500 次）是介词、冠词等功能词；低频段（<10 次）是人名、外来词或拼写错误；中频段（10-500 次）承载主要语义信息。*

- 高频段（>500 次出现）：介词、冠词、常见前后缀，语义信息少
- 低频段（<10 次）：人名、外来词、错误拼写，噪声大
- 中频段（10-500 次）：承载主要语义的 token

论文只关注中频段 token，通过控制数据集中覆盖的中频 token 类型数来调节多样性。

### 微观策略的算法细节

控制 token 级别的多样性比控制聚类/标签要复杂，因为一个样本平均含有 20+ 个中频 token（而宏观只有 1 个聚类、中观约 4 个标签）。论文用了两阶段算法：

1. **逆向贪心剪枝**（Algorithm 1）：给定目标 token 类型数 $k$，从完整数据集出发，反复移除当前贡献最少的样本，直到 token 类型数降到 $k$。这会留下一个中等规模的"候选池"。
2. **Token-aware 采样**（Algorithm 2）：从候选池中均匀采样，同时保证每种 token 类型都有足够的样本代表，最终得到固定大小（如 10K）的数据集。

所有策略最终构造的数据集形式相同：

$$\mathcal{D}_k = \bigcup_{i=1}^{k} \Pi(\mathcal{C}_i)$$

其中 $k$ 是类型数（topic/tag/token 类型），$\mathcal{C}_i$ 是第 $i$ 个类型对应的样本集合，$\Pi$ 是采样策略，确保各类型在最终数据集中均匀分布。

## 实验设置

- 数据源：117K 条开源 instruction，response 全部用 Llama-3.1-70B-Nemotron 重新生成（消除质量差异）
- 数据集大小：10K / 20K / 30K 三种规模
- 多样性梯度：每种策略构造 7 个数据集，多样性从 0% 到 100%
- 训练模型：Llama-2-7B（主实验）；Llama-2-70B 和 Llama-3-8B（消融实验）
- 评估方式：AlpacaEval 2.0 + Arena Hard 合并测试集，用 Llama-3.1-70B-Nemotron 做 pairwise 打分

## 主要结果

![Figure 3](/assets/images/dataset-diversity-sft/x3.png)
*Figure 3: 多样性百分比与模型性能的关系。每个子图中，不同颜色对应三个粒度：宏观（语义）、中观（标签）、微观（token）。虚线和阴影区域是贝叶斯线性回归及 1-sigma 不确定性。*

核心发现可以用下面这张表总结——表中数值是线性回归斜率（$\times 10^{-2}$），斜率越大说明增加多样性带来的性能提升越大：

| 数据集大小 | Instruction-Macro | Instruction-Meso | Instruction-Micro | Response-Macro | Response-Meso | Response-Micro |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 10K | 2.37 | 4.93 | -5.45 | 0.73 | 3.61 | 5.35 |
| 20K | 1.67 | 3.75 | -2.83 | 0.97 | 1.71 | 6.68 |
| 30K | 1.45 | 2.07 | -1.80 | 0.59 | 2.00 | 10.06 |

几个关键观察：

1. **Response-Micro 策略斜率最大**，且随数据集规模增大而增大（30K 时达到 10.06），说明在大数据集上 token 多样性的效应更加明显。
2. **Instruction-Micro 策略的斜率为负**。这说明在 instruction 上做 token 级别多样性控制反而有害——token 多样性不等于语义多样性，过于分散的 instruction token 可能破坏模型对常见指令模式的学习。
3. 中观策略在 instruction 和 response 两端都稳定正相关，但效果不如 Response-Micro。
4. 宏观策略有正效果但最弱，且范围窄。

## 多样性度量指标的有效性

![Figure 4](/assets/images/dataset-diversity-sft/x4.png)
*Figure 4: 各多样性指标与模型性能的 Pearson 相关系数。上半部分是 instruction 视角，下半部分是 response 视角。*

论文测试了 6 种后验多样性指标与模型性能的相关性：

| 指标 | Instruction 端相关性 | Response 端相关性 |
|:---|:---:|:---:|
| N-gram Ratio | 低 | 中 |
| Embedding Distance | 低 | 中 |
| Sequence Length | 低 | 0.73 |
| Compression Ratio | 低 | 中 |
| Self-BLEU | 0.42 | 中 |
| Information Entropy | 低 | 0.58 |

Response 端的序列长度和信息熵与性能相关性最高。但后续消融实验表明，长度本身不是因果因素（控制长度一致后，多样性-性能关系不变），信息熵才是更本质的指标——它直接衡量 token 分布的均匀程度，高熵意味着训练信号覆盖更广泛的 token 空间。

## 消融实验

### 跨模型验证

![Figure 5](/assets/images/dataset-diversity-sft/x6.png)
*Figure 5: 在 Llama-2-70B 和 Llama-3-8B 上的消融。Llama-3-8B 对 token 多样性更敏感。*

Llama-2-70B 和 Llama-3-8B 上的实验趋势与 Llama-2-7B 一致。Llama-3-8B 的斜率更大，可能因为其 tiktoken 分词器将词汇表从 32K 扩展到 128K，中频段 token 的覆盖范围更广，使得 token 多样性控制的操作空间更大。

### 长度控制

![Figure 6a](/assets/images/dataset-diversity-sft/x7.png)
*Figure 6(a): 控制 response 长度在 ~500 token 后，多样性-性能关系不变。上方为性能对比，下方为各数据集的平均 response 长度。*

由于 response 长度与性能相关性高达 0.73，需要排除"多样性高的数据集只是 response 更长"这个混淆因素。论文的做法不是截断，而是**按长度范围筛选**：从不同多样性梯度的子数据集中，只选取 response 长度在 ~500 token 附近的样本，使各数据集的平均 response 长度基本对齐。控制长度后，多样性与性能的正相关不变，说明长度只是伴随变量（incidental covariate），token 多样性才是驱动性能提升的本质因素。

### Tokenizer 的重要性

![Figure 6b](/assets/images/dataset-diversity-sft/x8.png)
*Figure 6(b): 用 word-based 分词替代 LLM tokenizer 后，正相关消失。*

用 word-based 分词（按空格切分）替代 LLM 自带的 tokenizer 来控制微观多样性后，正相关消失。这说明 tokenizer 的选择至关重要——需要与 LLM 训练时的 tokenization 方式一致，因为模型在训练时接收的输入就是按该 tokenizer 切分的。Word-based 分词破坏了这种对应关系。

## 思考与讨论

**为什么 response 比 instruction 更直接？** SFT 的 loss 只算在 response token 上。instruction 多样性是一种间接信号——你期望多样的 instruction 能"诱导"出多样的 response，但这个传导链条不一定可靠。直接控制 response 端 token 分布的均匀性，等于直接控制了训练信号的覆盖面。

**信息熵作为数据集质量指标的潜力。** 论文发现信息熵与模型性能相关性 0.58，且不依赖模型训练就能计算。如果这个结论在更多场景下成立，它可能成为一个廉价的数据集质量预估器——在花费大量算力训练模型之前，先看看 response 端的 token 熵是否足够高。

**局限性。** 论文的实验全部基于固定大小数据集从同一个 117K 语料池采样的设定。在实际场景中，数据来源更复杂，质量更参差。另外，论文只考虑了单轮 instruction-response 对，多轮对话场景下 token 多样性的定义和效果是否类似还不清楚。
