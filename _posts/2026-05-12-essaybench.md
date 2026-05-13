---
layout: post
title: "【论文笔记】EssayBench：面向多体裁中文作文的 LLM 写作评测基准"
date: 2026-05-12
categories: [paper-notes]
tags: [LLM-evaluation, Chinese-NLP, essay-writing, benchmark, LLM-as-judge]
paper_title: "EssayBench: Evaluating Large Language Models in Multi-Genre Chinese Essay Writing"
paper_authors: "Fan Gao et al."
paper_link: "https://arxiv.org/abs/2506.02596"
---

> **论文**：[EssayBench: Evaluating Large Language Models in Multi-Genre Chinese Essay Writing](https://arxiv.org/abs/2506.02596)
> **作者**：Fan Gao, Dongyuan Li, Ding Xia, Fei Mi, Yasheng Wang, Lifeng Shang, Baojun Wang
> **机构**：东京大学, 华为诺亚方舟实验室

## 快速概览

现有 LLM 写作评测基准大多采用粗粒度维度（流畅度、相关性、连贯性等）做直接打分，无法有效区分高质量和中等质量的文本，也忽略了中文作文在修辞传统和体裁规范上的独特要求。EssayBench 针对中文作文场景做了两件事：

1. 构建了一个覆盖议论文、记叙文、描写文、说明文四大体裁共 728 条 prompt 的基准数据集，并按约束程度分为 Open-Ended 和 Constrained 两类。
2. 提出了一套体裁导向、细粒度、层级依赖感知的评测框架，通过 LLM-as-a-judge 范式实现自动打分，与人类标注的排名相关性达到 Spearman $\rho = 0.816$（DeepSeek-R1 作为评判模型时）。

在 15 个主流 LLM 的基准测试中，Claude-3.7-sonnet 总分最高，中文模型中 Qwen-Max 排名第二；所有模型在记叙文和描写文上的表现明显弱于议论文和说明文。

## 评测框架设计

![Figure 1](/assets/images/essaybench/fig1.png)
*Figure 1: (a) 传统粗粒度评测方法 vs (b) EssayBench 的细粒度体裁导向层级评测框架。*

### 现有方法的问题

传统 LLM-as-a-judge 方法通常从 fluency、relevancy、coherency、readability 几个维度直接打 1-5 分或 1-10 分。问题在于：当前主流 LLM 在这些粗粒度维度上的得分已经普遍很高，难以区分不同水平的输出。此外，中文作文涉及表意文字、复杂句式和独特修辞传统，通用评测维度无法捕捉体裁特异性的写作要求。

### 三层设计

**体裁导向（Genre-Oriented）**：针对四种体裁分别定义 6 个评估维度，每个维度从基础要求到高级要求层级排列。例如议论文的层级从底到顶为：Argument（论点清晰度）→ Evidence（论据强度）→ Argumentation Methods（论证方法）→ Logical Development（逻辑发展）→ Expression（语言表达）→ Endings（结尾总结）。

**细粒度（Fine-Grained）**：在每个维度下设计多个子问题 $q_i$，使用 CoT 策略引导评判模型为每个子问题打 1-10 分，维度得分为子问题分数之和：

$$S_t = \sum_i q_i$$

**层级依赖感知（Dependency-Aware）**：不同层级的 trait 对最终评分的贡献不同。基础层级（depth $d=0$）比高级层级（depth $d=3$）权重更大。权重计算公式：

$$W_t = \alpha^d$$

其中 $\alpha$ 是超参数（实验中取 $\alpha = 3$），$d$ 是该维度在层级中的深度。最终总分为所有维度得分的加权和。直觉是：基础维度（如论点是否清晰）是高级维度（如结尾是否有力）的前提条件；如果基础都没做好，高级维度表现再好也意义有限。

## 数据集构建

![Figure 2](/assets/images/essaybench/fig2.png)
*Figure 2: 数据集构建流程（数据收集→过滤→分类）、prompt 示例和评测框架总览。*

数据来源包括线上聊天机器人的真实用户查询和教育考试材料。经过规则过滤、K-means 聚类去重、ROUGE-L < 0.7 的相似度筛选后，保留约 1000 条 prompt，再经人工审核最终得到 728 条。

prompt 按约束程度分为两类：
- **Open-Ended**：仅指定体裁和主题
- **Constrained**：额外包含字数限制、内容要求、目标读者等约束

![Figure 3](/assets/images/essaybench/fig3.png)
*Figure 3: 数据集在四种体裁和两类 prompt 上的分布。*

## 人类一致性验证

论文从四种体裁的两类难度中各随机采 10 条 prompt（共 80 条），用 7 个 LLM 生成作文，由 14 名中文语言学背景标注员做 pairwise 比较（每对 3 人标注，共 5040 条标注）。Fleiss' Kappa 为 0.469，属于中等一致性水平。

### 排名一致性

| 评判模型 | 方法 | Overall $\rho$ | Overall $\tau$ |
|---------|------|--------------|--------------|
| DeepSeek-V3 | Align-Score | 0.674 | 0.599 |
| DeepSeek-V3 | Ours | 0.667 | 0.549 |
| GPT-4o | Align-Score | 0.628 | 0.546 |
| GPT-4o | Ours | 0.733 | 0.627 |
| DeepSeek-R1 | Align-Score | 0.749 | 0.667 |
| DeepSeek-R1 | Ours | **0.816** | **0.704** |

关键发现：
1. **评判模型越强，EssayBench 框架优势越大**。R1 作为评判时，EssayBench 比 Align-Score 的 $\rho$ 高出 9%。这是因为框架需要模型在单轮内理解维度定义、分析文本并为多个子问题打分，对模型能力要求更高。
2. **依赖权重机制贡献约 2% 的提升**，说明层级加权确实有助于更准确地反映作文质量。
3. 框架在记叙文和描写文上的优势更明显——这两种体裁强调意象、修辞和文化表达，正是通用评测方法覆盖不到的地方。

### 敏感度分析

将标注数据按质量分高/中/低三档，用 Mann-Whitney U 检验和 Mean Difference 检验框架区分不同质量档次的能力。EssayBench 在 high vs. medium 的区分上比 Align-Score 提升 2%-10%，说明它特别擅长区分"好"和"很好"这种细微差异。

## 15 个 LLM 的基准测试结果

使用 GPT-4o 作为评判模型，temperature=0.2，分数归一化到 100 分制。

| 模型 | Overall | Argumentative | Narrative | Descriptive | Expository |
|------|---------|---------------|-----------|-------------|------------|
| Claude-3.7-sonnet | **76.6** | 77.7 / 78.8 | 75.7 / 75.3 | 74.6 / 73.6 | 77.5 / 79.0 |
| Qwen-Max | 75.6 | 74.5 / 78.7 | 73.5 / 74.7 | 74.1 / 72.6 | 77.1 / 77.6 |
| Claude-3.5-sonnet | 75.4 | 73.4 / 73.8 | 75.3 / 73.6 | 74.8 / 73.4 | 77.1 / 80.4 |
| DeepSeek-V3 | 75.1 | 77.2 / 77.9 | 71.2 / 71.8 | 72.7 / 67.8 | 80.4 / 79.4 |
| GPT-4o | 74.2 | 74.8 / 76.9 | 72.8 / 72.4 | 70.5 / 71.7 | 75.8 / 76.7 |
| GPT-3.5-turbo | 51.5 | 49.4 / 51.4 | 56.5 / 53.1 | 51.1 / 46.8 | 50.0 / 52.9 |
| LLaMA-3.1-70B | 40.5 | 37.6 / 46.6 | 35.1 / 28.6 | 45.0 / 42.2 | 39.6 / 44.8 |

*表中 Open / Cons. 分别对应 Open-Ended 和 Constrained 两种 prompt 类型的得分。*

![Figure 5](/assets/images/essaybench/fig5.png)
*Figure 5: 各模型在不同体裁上的得分箱线图。议论文和说明文得分明显高于记叙文和描写文。*

### 关键观察

**体裁差异**：所有模型在议论文和说明文上表现更好，在记叙文和描写文上明显下降。前者强调结构化、逻辑性和清晰度——这些是 LLM 的强项；后者需要创意、情感表达和文化语境意识，LLM 尚有欠缺。

**Open-Ended vs. Constrained**：有约束的 prompt 反而让模型表现更好。这不难理解——约束提供了明确的写作方向和结构指导，减少了模型在自由发挥时的规划负担。

**中文模型的竞争力**：Qwen-Max 总分排第二，DeepSeek-V3 超过 Grok-3 和 GPT-4o，Qwen-2.5-72B 超过同体量的 LLaMA-3.1-70B，说明中文优化确实带来了写作质量的提升。

## 讨论

EssayBench 的设计思路值得关注：不是简单地增加评测维度数量，而是引入层级结构和依赖权重，让基础写作能力在总分中占更大比重。这与教育实践中的作文评分逻辑一致——一篇论点混乱的议论文，即使语言华丽也不会得高分。

不过论文也承认几个局限：评测维度主要关注段落和整体层面的表达与结构，缺少词汇和句子级别的细粒度分析；没有单独评估 instruction-following 能力（如是否严格遵守字数限制）；数据集和评测目前局限于中文，虽然框架本身可以扩展到其他语言。

从实用角度看，EssayBench 为中文教育场景下的 LLM 选型提供了有价值的参考，也为未来改进 LLM 的中文叙事和描写能力指明了方向。
