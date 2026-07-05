---
layout: post
title: "【论文笔记】CausalMix：用因果推断优化 LLM 训练数据配比"
date: 2026-07-05
categories: [paper-notes]
tags: [data-mixture, causal-inference, SFT, LLM-training, double-machine-learning]
paper_title: "CausalMix: Data Mixture as Causal Inference for Language Model Training"
paper_authors: "Zinan Tang, Yukun Zhang et al."
paper_link: "https://arxiv.org/abs/2607.01104"
---

> **论文**：[CausalMix: Data Mixture as Causal Inference for Language Model Training](https://arxiv.org/abs/2607.01104)
> **作者**：Zinan Tang, Yukun Zhang, Shaomian Zheng, Zhuoshi Pan, Qizhi Pei, Dingnan Jin, Jun Zhou, Yujun Wang, Biqing Huang
> **机构**：Tsinghua University, Ant Group, Renmin University of China

## 从"找最优配比"到"估计因果边际收益"

SFT 阶段的数据配比（coding、math、instruction following 等各类数据各占多少比例）对模型最终表现影响显著，但现有方法（如 RegMix）都在学一个静态映射：配比 → 验证 loss。问题在于，当数据池本身发生变化（质量、难度、规模不同），之前学到的映射就失效了，需要重新跑大量 proxy 实验。

CausalMix 换了一个视角：不再试图找到一个"万能最优配比"，而是问一个局部因果问题——**在当前数据状态下，增加某个 domain 的比例，对下游表现的因果边际收益是多少？** 这个问题天然考虑了数据状态（quality、difficulty、complexity）的影响，因此能在数据池变化时自适应调整。

## 方法框架

![Figure 1](/assets/images/causalmix/x1.png)
*Figure 1: CausalMix 流程总览。从历史 proxy 训练中提取数据状态协变量、配比分配和下游表现，通过正交因果学习估计状态条件下的边际数据收益。*

### 问题建模

给定 $K$ 个数据 domain，一次训练对应一个配比向量 $T = (T_1, \ldots, T_K)$，满足 $\sum_k T_k = 1$。对每次 proxy 训练，观测三元组 $(X_i, T_i, Y_i)$：

- $X_i$：数据状态协变量（训练前可获取的数据统计特征）
- $T_i$：该次训练使用的 domain 配比
- $Y_i$：训练后的下游评测表现

为捕捉配比的边际递减效应并处理组合数据的几何性质，对配比做 log 变换：$Z = \log(T + \varepsilon)$。

核心目标是估计 **state-conditioned marginal data return** $\theta_0(x) \in \mathbb{R}^K$，定义来自部分线性近似：

$$
\mu(x, Z) \approx g(x) + \theta_0(x)^\top Z
$$

其中 $g(x)$ 是状态依赖的 baseline 表现，$\theta_0(x)$ 是在状态 $x$ 下各 domain 的因果边际收益。$\theta_{0,k}(x) > 0$ 意味着增加 domain $k$ 的比例会提升表现，反之会引起负迁移。

### 正交化估计（Double Machine Learning）

直接回归会把数据状态的 baseline 效应和配比的因果效应混在一起。CausalMix 用 DML 框架将两者分离：

1. 学两个 nuisance 函数：$m_0(X) = \mathbb{E}[Y \mid X]$（给定状态预测表现）和 $e_0(X) = \mathbb{E}[Z \mid X]$（给定状态预测 log-配比）
2. 计算残差：$\widetilde{Y} = Y - m_0(X)$, $\widetilde{Z} = Z - e_0(X)$
3. 从残差关系估计因果效应：$\widetilde{Y} \approx \theta_0(X)^\top \widetilde{Z}$

这一步的直觉是：先去掉"数据状态本身就决定的那部分表现"和"数据状态本身就决定的那部分配比偏向"，剩下的残差关系才是配比变化对表现的因果影响。

实践中用 cross-fitting 避免过拟合：将 512 次 proxy 实验分折，每个残差由不包含该样本的模型预测得到。最终用 CausalForestDML 作为异质效应模型：

$$
\hat{\theta} = \arg\min_\theta \sum_i \left(\widetilde{Y}_i - \theta(X_i)^\top \widetilde{Z}_i\right)^2
$$

### 从边际收益到配比策略

估计出目标数据状态的边际收益 $\hat{\theta}(X_{\text{tar}})$ 后，转化为实际配比：

**解析策略（CausalMix-A）**：直接将正的边际收益归一化到 simplex：

$$
T_k^A = \frac{[\hat{\theta}_k(X_{\text{tar}})]_+}{\sum_{j=1}^K [\hat{\theta}_j(X_{\text{tar}})]_+}
$$

**搜索策略（CausalMix-S）**：从 Dirichlet 分布采样 100,000 个候选配比，用因果模型预测得分，取 top-100 的平均作为最终配比。相当于对高分候选做 local bagging，降低单点估计的噪声。

## 实验设计

- **数据**：tulu-3-sft-mixture，5 个 domain（Coding, IF, Math, Knowledge, Safety）
- **Proxy 实验**：512 次 Qwen2.5-0.5B 训练，每次 100K 样本
- **协变量**：从 OpenDataArena-scored-data 获取的三个指标——Normalized_Loss（数据可预测性/难度）、Writing_Style（文本质量）、HES（高熵 token 采样，反映推理复杂度）
- **因果模型**：EconML 框架，LightGBM 作为第一阶段预测器，CausalForestDML 作为因果估计器

## 主要结果

| Method | Knowledge | Reasoning | Math | Coding | IF | Safety | Avg_Dev |
|--------|-----------|-----------|------|--------|------|--------|---------|
| **Qwen2.5-0.5B, 800K** |
| Equal | 28.07 | 29.28 | 21.59 | 36.06 | 46.95 | 25.09 | 31.78 |
| RegMix | 24.90 | 29.94 | 30.25 | 25.46 | 25.32 | 22.52 | 26.40 |
| DoReMi | 27.70 | 29.66 | 20.58 | 34.53 | 42.88 | 32.93 | 31.38 |
| DMO | 28.07 | 31.88 | 22.42 | 26.90 | 41.59 | 41.37 | 32.04 |
| **CausalMix-A** | 27.93 | 29.68 | 23.56 | 27.76 | 43.81 | **50.92** | **33.94** |
| CausalMix-S | 28.31 | 30.96 | 27.64 | 30.97 | 42.51 | 36.47 | 32.81 |
| **Qwen2.5-7B, 800K** |
| Equal | 60.85 | 64.55 | 59.03 | 53.61 | 68.58 | 53.49 | 60.02 |
| DMO | 59.15 | 63.70 | 60.62 | 54.05 | 70.24 | 54.35 | 60.35 |
| **CausalMix-A** | 57.14 | 64.03 | 58.51 | **65.52** | 68.21 | 57.65 | **61.84** |
| **CausalMix-S** | 59.35 | 62.88 | 58.63 | 64.43 | 67.47 | 60.95 | **62.28** |

跨 100K、400K、800K 三种数据规模和 0.5B、7B 两种模型规模，CausalMix 均取得最优或次优的 Avg_Dev。从 0.5B proxy 学到的配比策略直接迁移到 7B 模型仍然有效，验证了 rank invariance 假设。

### LongCoT 数据迁移实验

更有意思的是迁移实验：用在 tulu-3-sft-mixture 上训练的因果模型，直接对全新的 AM-Thinking-v1-Distilled（长链式推理数据）推断最优配比，并用 Qwen3-4B（与 proxy 模型 Qwen2.5 不同系列）训练评估：

| Method | GSM8K | MATH | Avg_Math | HumanEval | MBPP | Avg_Code | Avg |
|--------|-------|------|----------|-----------|------|----------|-----|
| Equal | 90.45 | 56.78 | 73.62 | 59.76 | 48.20 | 53.98 | 63.80 |
| Grid | 87.34 | 61.20 | 74.27 | 62.80 | 47.60 | 55.20 | 64.74 |
| RegMix | 89.61 | 40.80 | 65.21 | 61.59 | 53.60 | 57.60 | 61.40 |
| DMO | 89.61 | 54.38 | 72.00 | 54.88 | 55.00 | 54.94 | 63.47 |
| **CausalMix** | 88.86 | **60.58** | **74.72** | 62.20 | **55.00** | **58.60** | **66.66** |

CausalMix 在跨数据集 + 跨模型系列的条件下依然取得最佳表现，说明它学到的不是某个特定数据集的 pattern，而是数据配比的内在因果规律。

## 消融与分析

### 关键组件的必要性

| Method | Avg (0.5B, 800K) | Avg (7B, 800K) |
|--------|-------------------|----------------|
| w/o X（去掉协变量） | 33.29 | 61.30 |
| w/o Orth.（去掉正交化） | 32.66 | 59.65 |
| CausalMix-A | **33.94** | 61.84 |
| CausalMix-S | 32.81 | **62.28** |

去掉协变量退化为 RegMix 式的全局映射，性能下降；去掉正交化直接拼 $(X, T)$ 预测 $Y$，性能更差（甚至不如不用 $X$），说明直接回归的正则化偏差会误导配比决策。

### 协变量选择

![Figure 2](/assets/images/causalmix/x2.png)
*Figure 2: 不同协变量组合下的 Spearman 等级相关系数。三个协变量（HES + Normalized_Loss + Writing_Style）的组合效果最佳。*

三个协变量分别对应数据的 Complexity、Difficulty、Quality 维度。过少的协变量无法刻画数据状态，过多则在 512 样本上受维度诅咒影响。

### CATE 解释器：技能冲突的量化证据

![Figure 3](/assets/images/causalmix/x3.png)
*Figure 3: CATE 模型树解释器的简化可视化。展示不同数据状态下各 domain 的边际收益方向。*

几个发现：

1. **IF 数据始终正收益**：instruction following 数据在所有特征子空间下都稳定贡献正向收益
2. **Knowledge 与推理的冲突**：当目标数据 Normalized_Loss 和 HES 都高（即数据难度大、推理复杂度高）时，Knowledge 数据产生负效应，证实了"技能冲突"的存在
3. **质量门槛效应**：当 Writing_Style 和 HES 较低（数据质量差）时，Math、Coding、Safety 这些复杂 domain 反而引入噪声降低表现；但质量达到一定水平后，它们会产生协同增益

## 讨论

CausalMix 的核心贡献是把数据配比问题从"超参数搜索"重新定义为"因果边际收益估计"，这个转换带来了三个实际好处：状态自适应（数据池变了不需要重新做 proxy 实验）、可解释性（可以分析哪些 domain 在什么条件下有用）、可迁移性（跨数据集和模型都能工作）。

一个自然的问题是 512 次 proxy 实验对于更复杂的场景是否足够。论文在协变量分析中也提到，受限于样本量，只能使用 3 个协变量，更多维度的数据特征可能需要更大规模的 proxy 实验来支撑。另外，当前方法假设配比在训练过程中是固定的，如何与动态调度（在训练过程中逐步调整配比）结合，可能是后续的延伸方向。

从工程角度看，512 次 0.5B 模型训练的计算成本相对可控，而学到的因果模型可以复用到 7B 甚至更大模型，这种"小模型探路，大模型受益"的 paradigm 在实际落地中有明确的成本优势。

## AI 犀利评判

**方法的"因果性"被高估了。** 论文把 data mixture 套上因果推断框架，听起来很 principled，但 proxy 实验本身就是随机采样配比的——treatment assignment 接近随机化实验，confounding 并不严重。DML 正交化在这里更像一个去噪技巧，而非解决了真正困难的因果识别问题。论文花大篇幅论证 ignorability、overlap 等因果识别条件，但在随机实验设计下这些条件天然满足，有过度 formalize 之嫌。

**512 个样本 + 3 个协变量，异质效应建模的上限很低。** 论文自己承认多加协变量反而降性能，说明数据量不够支撑更复杂的 CATE 建模。用 512 个点学一个从 3 维到 5 维 CATE 的映射，CausalForest 能捕获多少 heterogeneity 存疑——很可能最终生效的就是一个近似线性的全局趋势。

**实验提升幅度有限，且缺乏统计显著性验证。** Equal mixture（零成本 baseline）在多个设置下表现不差。7B 800K 场景下 Equal 60.02 vs CausalMix-S 62.28，1-2 个点的平均提升在没有误差棒的情况下无法确认为显著。拆开看各 domain，CausalMix 在 Knowledge（57.14 vs 60.85）和 Reasoning（62.88 vs 64.55）上还不如 Equal，主要靠 Coding 一个维度拉上去。论文对每个配置只跑了一次，SFT 训练本身的随机性（数据采样顺序等）可能就有 1-2 个点的波动。

**"迁移性"验证的场景过于简单。** 从 tulu-3 迁移到 AM-Thinking 的实验只涉及 2 个 domain（math + code），配比空间实质上是一维的。在一维上几乎任何方法都能给出合理建议。真正有说服力的迁移应该是 domain 划分方式不同、特征分布有明显 shift 的场景。

**Baseline 选择有取巧。** RegMix、DoReMi、ODM 都是为 pre-training 设计的方法，搬到 SFT 场景当 baseline 天然吃亏。SFT-specific 的对比只有 DMO 一个，且用的是 DMO 论文报告的比例而非重新适配到当前数据。

**CATE 解释器更像 post-hoc storytelling。** "IF 数据总是正收益""Knowledge 和推理冲突"这些发现，用简单的交叉实验或 feature importance 分析就能得到，不需要因果框架。因果框架本应给出 counterfactual 级别的洞察，但这里的解释粒度跟普通的特征重要性没有本质区别。

**总体判断：** 执行质量不错、实验覆盖面较广的工作，但 framing 有 overclaim。核心贡献更接近"用 DML 做了一个更 robust 的 data mixture 回归模型"，距离论文标题暗示的"揭示数据配比的因果机制"还有差距。
