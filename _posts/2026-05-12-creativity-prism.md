---
layout: post
title: "【论文笔记】CreativityPrism：LLM 创造力的全方位评测框架"
date: 2026-05-12
categories: [paper-notes]
tags: [LLM, creativity, benchmark, evaluation, divergent-thinking]
paper_title: "CreativityPrism: A Holistic Benchmark for Large Language Model Creativity"
paper_authors: "Zhaoyi Joey Hou et al."
paper_link: "https://arxiv.org/abs/2510.20091"
---

> **论文**：[CreativityPrism: A Holistic Benchmark for Large Language Model Creativity](https://arxiv.org/abs/2510.20091)
> **作者**：Zhaoyi Joey Hou, Bowei Alvin Zhang, Yining Lu, Bhiman Kumar Baghel, Anneliese Brei, Ximing Lu, Meng Jiang, Faeze Brahman, Snigdha Chaturvedi, Haw-Shiuan Chang, Daniel Khashabi, Xiang Lorraine Li
> **机构**：University of Pittsburgh, Johns Hopkins, Notre Dame, UNC Chapel Hill, UW, UMass Amherst, Allen AI

## 快速概览

现有 LLM 创造力评测碎片化严重——有的只看词汇多样性、有的只看生成内容与训练数据的 n-gram 重叠率、有的只看是否能想出不寻常的物品用途。这些任务各自只捕捉了创造力的一个切面，难以给出全景评价。

CreativityPrism 做的事情：把创造力拆解为 **quality（质量）、novelty（新颖性）、diversity（多样性）** 三个维度，跨越 divergent thinking、creative writing、logical reasoning 三个领域，包含 9 个任务和 20 个评测指标，对 17 个 SOTA LLM 进行系统评测。核心发现是——在某一维度或领域表现好，不代表其他维度也好；闭源模型在整体创造力上仍显著领先开源模型。

## 框架设计

![Figure 1](/assets/images/creativity-prism/x1.png)
*Figure 1: CreativityPrism 框架全景。上方为三个领域（发散思维、创意写作、逻辑推理）的 9 个任务，下方将 20 个评测指标归入质量、新颖性、多样性三个维度。*

### 三个领域、九个任务

| 领域 | 任务 | 简要描述 |
|------|------|----------|
| Divergent Thinking | AUT (Alternative Uses Test) | 给定常见物品，生成非常规用途 |
| | DAT (Divergent Association Task) | 生成 10 个尽可能不相关的名词 |
| | TTCT (Torrance Tests) | 回答经典心理学创造力测试题 |
| Creative Writing | TTCW (Torrance Creative Writing) | 根据摘要生成 New Yorker 风格文章 |
| | Creative Short Story | 给定 3 个关键词，写 ≤5 句短故事 |
| | Creativity Index | 给定小说/诗歌/演讲前缀，续写段落 |
| | CS4 | 在不断增加的约束下修改基础故事 |
| Logical Reasoning | NeoCoder | 在受限技术条件下写代码解题 |
| | Creative Math | 给定参考解法后，生成不同的数学解法 |

### 三个评测维度

- **Quality**：生成内容是否满足任务的基本功能需求。如 NeoCoder 看代码能否通过测试、CS4 看故事是否连贯且满足约束。
- **Novelty**：生成内容与已有内容的差异程度。如 Creative Math 要求解法不同于参考解法、Creativity Index 测量与预训练语料的 n-gram 重叠。
- **Diversity**：多次生成之间的变化幅度。如 Creative Short Story 的词汇多样性、DAT 中生成名词的语义距离。

每个任务至少覆盖一个维度，没有哪个任务能同时完美覆盖三个维度，这正说明了全方位评测框架的必要性。

## 评测方法

### 分数聚合

1. 每个指标做 min-max 归一化到 [0, 1]
2. 同一任务在同一维度有多个指标时先取任务内平均，避免某个任务因指标多而权重过大
3. 每个维度的分数 = 该维度下所有任务平均分
4. Overall = Quality + Novelty + Diversity 三个维度的简单平均

### LLM-as-a-Judge 可靠性

9 个任务中有 6 个使用 LLM 作为自动评测的一部分（默认 Qwen2.5-72B）。论文对每个任务报告了人类-LLM 评分一致性：

- AUT：Pearson r = 0.70
- NeoCoder：技术检测 recall = 0.94
- CS4：Pearson r = 0.55 (p<0.01)
- Creative Math：novelty 准确率 0.78，correctness 准确率 0.94
- TTCW：只保留人机一致性 ≥ 0.69 的四个子指标

## 实验结果

### 整体排名

| 模型 | Overall | Quality | Novelty | Diversity |
|------|---------|---------|---------|-----------|
| DeepSeek-V3 | .739 | .716 | .720 | .854 |
| GPT-4.1 | .721 | .697 | .692 | .871 |
| GPT-4.1-mini | .695 | .681 | .678 | .774 |
| Claude3-Sonnet | .697 | .672 | .663 | .835 |
| Gemini-2.0-Flash | .677 | .645 | .654 | .822 |
| DeepSeek-R1 | .638 | .573 | .600 | .710 |
| Qwen2.5-72B | .596 | .581 | .595 | .674 |
| Llama3.3-70B | .541 | .533 | .574 | .562 |
| Mistral-7B | .522 | .376 | .558 | .649 |

DeepSeek-V3 是综合冠军（.739），在三个维度上均领先。开源模型中 Qwen2.5-72B 表现最好（.596），但与闭源顶级模型差距超过 20%。

### 闭源 vs 开源差距

![Figure 2](/assets/images/creativity-prism/plot-perf-overall.png)
*各模型在三个维度上的归一化表现。*

差距在不同领域分布不均：

- **Logical reasoning 差距最大**：闭源模型在编程和数学推理上投入了大量训练资源
- **Creative writing 差距次之**：闭源模型通常使用了角色扮演或创意写作数据做后训练
- **Divergent thinking 差距最小**：这类心理学任务在任何模型的训练中都没有被特别重视

在维度层面，quality 和 diversity 的差距大于 novelty，可能因为闭源模型有更多高质量私有数据，且在推理任务上进行了专项优化。

### 相关性分析

![Figure 3](/assets/images/creativity-prism/x2.png)
*按任务和领域分组的模型表现相关性矩阵。对角线块为同任务内指标间相关性。*

![Figure 4](/assets/images/creativity-prism/x3.png)
*按创造力维度分组的相关性矩阵（左）及各模型在三个维度的雷达图（右）。*

关键发现：

1. **同任务内高相关**：TTCW、TTCT 等任务内部的各指标间 Pearson r > 0.85，说明任务内不同指标测的是同一能力
2. **Diversity 和 Quality 跨任务也高相关**：在这两个维度上某个任务表现好的模型，在其他任务上往往也好
3. **Novelty 跨任务相关性很低**：不同任务对"新颖性"的定义差异巨大。比如 Creative Short Story 的 Surprises（句间语义跃迁）和 NeoCoder 的 Divergence@0（解法不同于参考）甚至呈负相关 (-0.25)
4. **跨领域相关性弱**：divergent thinking 表现好不代表 logical reasoning 也好

这些发现支持了论文的核心立论：创造力是多面的，单一任务或单一维度的评测无法给出全面结论。

## 讨论

**为什么 novelty 维度相关性低？** 不同任务中"新颖"的操作定义差异太大——有的是与训练数据的 n-gram 距离、有的是与参考解法的编辑距离、有的是读者感知到的情节反转程度。这些定义之间几乎没有内在联系。

**时间趋势**：论文发现模型创造力随发布时间线性提升（三个维度均是如此）。但作者也指出，部分指标（如 Creativity Index 的 L-uniqueness）天然偏爱知识截止日期更新的模型——更晚发布的模型训练数据更新，因此生成的内容与旧语料的 n-gram 重叠更低。

**局限性**：(1) 仅覆盖英文，创造力具有文化依赖性；(2) 6/9 的任务依赖 LLM-as-a-Judge，存在评估偏差；(3) 仅关注文本模态；(4) 任务选取受限于自动化评测的可行性，排除了一些高层次的创造力概念。

整体而言，CreativityPrism 提供了一个结构清晰、可扩展的创造力评测体系。它最有价值的贡献不是"哪个模型最有创造力"这个排名，而是揭示了创造力各维度之间的低相关性——提醒我们不要用单一任务的表现去推断模型的整体创造力水平。
