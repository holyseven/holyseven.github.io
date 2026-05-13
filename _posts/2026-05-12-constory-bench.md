---
layout: post
title: "【论文笔记】LLM 长篇故事生成中的一致性 Bug 分析"
date: 2026-05-12
categories: [paper-notes]
tags: [LLM, story-generation, consistency, benchmark, evaluation]
paper_title: "Lost in Stories: Consistency Bugs in Long Story Generation by LLMs"
paper_authors: "Junjie Li et al."
paper_link: "https://arxiv.org/abs/2603.05890"
---

> **论文**：[Lost in Stories: Consistency Bugs in Long Story Generation by LLMs](https://arxiv.org/abs/2603.05890)
> **作者**：Junjie Li, Xinrui Guo, Yuhao Wu, Roy Ka-Wei Lee, Hongzhi Li, Yutao Xie
> **机构**：Microsoft（北京）, Singapore University of Technology and Design

## 快速概览

LLM 现在能生成数万字的长篇叙事，但在写长故事时经常"前后矛盾"——角色的名字、时间线、世界设定等会出现自相矛盾的情况。这篇论文做了三件事：（1）提出了 ConStory-Bench，一个包含 2000 条 prompt 的长篇叙事一致性评测基准；（2）设计了 ConStory-Checker，一个自动化的矛盾检测 pipeline；（3）系统地评估了主流 LLM（包括 GPT-5-Reasoning、Gemini-2.5-Pro、Claude-Sonnet-4.5 等）在长故事生成中的一致性表现，并从多个角度分析了错误产生的规律。

关键发现是：一致性错误在"事实细节"和"时间线逻辑"两个维度最为突出，错误倾向于出现在叙事中段（40%-60%位置），且在模型不确定性（entropy）更高的片段更容易发生。

## 问题定义：什么是叙事一致性错误

当 LLM 生成一篇 8000-10000 词的故事时，它需要在上下文窗口中追踪大量信息——人物外貌、名称、时间线、因果逻辑、世界规则等。当后文与前文建立的事实矛盾时，就产生了一致性错误。

论文定义了 5 大类、19 种细分的错误类型：

| 错误大类 | 细分类型（部分） |
|---------|--------------|
| Timeline & Plot Logic | 绝对时间矛盾、持续时间矛盾、因果逻辑违反、被遗弃的情节线 |
| Characterization | 记忆矛盾、知识矛盾、技能波动、遗忘能力 |
| World-building & Setting | 核心规则违反、社会规范违反、地理矛盾 |
| Factual & Detail Consistency | 外貌不匹配、名称混淆、数量不匹配 |
| Narrative & Style | 视角混淆、语调不一致、风格突变 |

![Figure 2](/assets/images/constory-bench/error_examples.png)
*Figure 2: 真实 LLM 生成故事中的一致性错误示例。高亮段展示了五大类错误的具体矛盾。*

## 方法：ConStory-Bench 与 ConStory-Checker

### 数据构建

论文从 7 个公开语料库（LongBench、WritingPrompts、WikiPlots 等）收集种子故事，然后用 LLM 将其改写为 4 种任务类型的 prompt：

- **Generation**（37.5%）：仅给出简短情节前提，自由创作
- **Continuation**（21.6%）：给出故事开头，续写为完整叙事
- **Expansion**（21.1%）：给出简洁但完整的情节大纲，展开细节
- **Completion**（19.8%）：给出开头和结尾，补写中间部分

所有 prompt 要求生成 8000-10000 词的故事。

### 自动评估 Pipeline

![Figure 1](/assets/images/constory-bench/x1.png)
*Figure 1: ConStory-Bench 整体框架。包含 2000 条 prompt 的基准、三阶段检测 pipeline、以及标准化的 CED/GRR 评分。*

ConStory-Checker 是一个四阶段的 LLM-as-judge pipeline（使用 o4-mini 作为评估模型）：

1. **Category-Guided Extraction**：按五个维度分别扫描叙事，提取可能矛盾的文本片段
2. **Contradiction Pairing**：对提取的片段做两两比对，判断"一致"或"矛盾"
3. **Evidence Chains**：为每个矛盾记录推理过程、引用文本证据、错误分类
4. **JSON Reports**：输出标准化 JSON，包含引文、位置偏移、配对关系和解释

### 评估指标

论文提出了两个互补的指标：

**Consistency Error Density (CED)**：每万词的错误数量，消除输出长度的偏差。

$$\text{CED}_{m,i} = \frac{e_{m,i}}{w_{m,i} / 10000}$$

其中 $e_{m,i}$ 是模型 $m$ 在故事 $i$ 上的错误计数，$w_{m,i}$ 是词数。

**Group Relative Rank (GRR)**：在同一 prompt 内对所有模型排名，控制 prompt 难度差异。先计算一个综合质量分：

$$Q_{m,i} = \frac{w_{m,i}}{1 + e_{m,i}}$$

然后在每个 prompt 内排名取平均。GRR 越低越好。

## 实验结果

### 总体性能对比

| 模型 | CED | GRR | 平均词数 | 平均错误数 |
|------|-----|-----|---------|----------|
| GPT-5-Reasoning | 0.113 | 3.05 | 9050 | 0.09 |
| Gemini-2.5-Pro | 0.305 | 7.79 | 5584 | 0.16 |
| Claude-Sonnet-4.5 | 0.520 | 4.90 | 8929 | 0.37 |
| GLM-4.6 | 0.528 | 8.45 | 4949 | 0.18 |
| Qwen3-32B | 0.537 | 6.39 | 6237 | 0.27 |
| DeepSeek-V3.2-Exp | 0.541 | 10.89 | 3724 | 0.15 |
| LongWriter-Zero | 0.669 | 5.45 | 13393 | 0.53 |
| SuperWriter (Agent) | 0.674 | 7.97 | 6036 | 0.38 |
| MiniMax-M1-80k | 3.447 | 18.07 | 1442 | 0.38 |

GPT-5-Reasoning 表现最优（CED 仅 0.113），不仅能写出长文（平均 9050 词），且几乎不犯一致性错误。开源模型中 GLM-4.6 和 Qwen3-32B 接近闭源水平。值得注意的是，Generation（开放式创作）任务的错误率普遍高于有上下文约束的任务类型。

从错误分布来看，Factual & Detail Consistency 和 Timeline & Plot Logic 是最主要的两类失败模式，说明实体追踪和时序推理仍然是核心挑战。

### 错误随长度线性增长

![Figure 3](/assets/images/constory-bench/length_distribution.png)
*Figure 3: 各模型的输出长度分布。*

![Figure 4](/assets/images/constory-bench/error_growth.png)
*Figure 4: 错误数量随故事长度的增长关系。*

错误数量与输出长度近似线性相关。Claude-Sonnet-4.5 的长度-错误相关系数 r=0.478（中等），DeepSeek-V3.2-Exp 达到 r=0.973（强相关）。这意味着更长的故事不可避免地会积累更多矛盾，但不同模型的"斜率"差异很大。

### 高 Entropy 区域更容易出错

论文用 Shannon entropy 衡量模型生成每个 token 时的不确定性：

$$H(P_t) = -\sum_{i=1}^{K} p_i \log_2 p_i$$

在 Qwen3-4B 上，错误片段的平均 entropy 比全文基线高 19.24%；Qwen3-30B 上高 12.03%。这说明模型并非"不知不觉"地犯错，而是在面对更大不确定性时更容易做出错误选择。

实际意义：entropy 可以作为一个"预警信号"——当局部 entropy 超过阈值时，触发验证或自检机制来防止一致性错误。

### 错误类型之间的共现关系

![Figure 5](/assets/images/constory-bench/causal_relationship.png)
*Figure 5: 错误类型之间的相关性矩阵。颜色越深表示共现越强。*

Factual & Detail Consistency 是一个"中枢"类错误，与 Characterization（r=0.304）、World-building（r=0.255）和 Timeline（r=0.176）都有显著相关。换言之，事实细节出错时，往往伴随着其他维度的连锁错误。

相比之下，Narrative & Style 错误与其他类型几乎零相关，说明风格不一致是通过独立机制产生的，可能与事实追踪能力无关。

### 错误的位置分布规律

![Figure 6](/assets/images/constory-bench/dumbbell.png)
*Figure 6: 错误位置分布的哑铃图。蓝点为事实首次建立位置，红点为矛盾出现位置，连线为两者间距。*

事实通常在叙事早期（15%-30%位置）建立，而矛盾集中出现在中段（40%-60%位置）。不同错误类型的"间距"差异很大：

- 地理矛盾的平均间距最大（31.0%），说明这是远程记忆失败
- 绝对时间矛盾紧随其后（29.7%）
- 视角混淆的间距最小（4.7%），说明这是局部生成问题

这提示我们：时间和地理类错误需要强大的远程记忆机制，而风格错误可以通过局部一致性检查解决。

## 思考与讨论

**Entropy 作为实时干预信号的可行性**。论文发现错误区域的 entropy 显著偏高，这为"生成时检测"提供了一条可操作的路径。可以想象一个系统：在长文生成过程中实时监控 entropy，一旦超过阈值就暂停生成、回查上下文中的关键事实，然后再继续。这比生成完再做后处理更高效。

**为什么开放式生成更难保持一致**。Generation 任务（无先验上下文约束）的错误率最高，这可能因为模型在无约束情况下倾向于"即兴发挥"，引入更多实体和情节线，而每多一个实体就多一个潜在的追踪负担。这也解释了为什么 agent-based 系统（如 SuperWriter）尽管引入了多步规划，CED 仍然不够低——规划解决了情节结构问题，但不一定能解决细粒度的事实追踪问题。

**评估方法的局限与启发**。论文坦承一致性是二元判断（矛盾/不矛盾），但文学创作中有些"矛盾"是有意为之的（如不可靠叙述者、叙事反转）。如何区分"bug"和"feature"是一个值得探索的方向。此外，当前 benchmark 仅覆盖英文西方叙事传统，不同文化对叙事一致性的期待可能不同。
